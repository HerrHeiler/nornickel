"""Memory-bounded sliding-window inference for arbitrarily large images.

Strategy: overlap-tile with Gaussian blending of per-class probabilities.
Peak memory ≈ H×W×num_classes float16 accumulator + one batch of tiles,
so a 10k×10k image with 4 classes needs ~800 MB RAM and <4 GB VRAM.
"""
from typing import Callable, Iterator

import numpy as np
import torch


def _gaussian_weight(tile: int, sigma_frac: float = 0.35) -> np.ndarray:
    ax = np.linspace(-1, 1, tile)
    g = np.exp(-(ax ** 2) / (2 * sigma_frac ** 2))
    w = np.outer(g, g).astype(np.float32)
    return np.clip(w, 1e-3, None)


def iter_tiles(h: int, w: int, tile: int, stride: int) -> Iterator[tuple[int, int]]:
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y in ys:
        for x in xs:
            yield y, x


@torch.inference_mode()
def predict_large_image(
    img: np.ndarray,
    forward: Callable[[torch.Tensor], torch.Tensor],
    num_classes: int,
    tile: int = 1024,
    overlap: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
    use_amp: bool = True,
    progress_cb: Callable[[float], None] | None = None,
) -> np.ndarray:
    """img: HxWx3 uint8 RGB -> HxW uint8 class-index mask."""
    h, w = img.shape[:2]
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W = img.shape[:2]
    stride = tile - overlap

    prob = np.zeros((num_classes, H, W), dtype=np.float16)
    weight = np.zeros((H, W), dtype=np.float32)
    gw = _gaussian_weight(tile)

    coords = list(iter_tiles(H, W, tile, stride))
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        tiles = np.stack([img[y:y + tile, x:x + tile] for y, x in batch_coords])
        t = (tiles.astype(np.float32) / 255.0 - mean) / std
        t = torch.from_numpy(t).permute(0, 3, 1, 2).to(device, non_blocking=True)

        amp_ctx = torch.autocast(device_type="cuda", enabled=use_amp and device == "cuda")
        with amp_ctx:
            logits = forward(t)
            p = torch.softmax(logits.float(), dim=1)
        p = p.cpu().numpy().astype(np.float16)

        for j, (y, x) in enumerate(batch_coords):
            prob[:, y:y + tile, x:x + tile] += p[j] * gw
            weight[y:y + tile, x:x + tile] += gw

        if progress_cb:
            progress_cb(min(1.0, (i + len(batch_coords)) / len(coords)))

    mask = np.empty((H, W), np.uint8)
    conf = np.empty((H, W), np.float16)
    step = 512
    for y0 in range(0, H, step):
        sl = prob[:, y0:y0 + step]
        mask[y0:y0 + step] = np.argmax(sl, axis=0).astype(np.uint8)
        conf[y0:y0 + step] = (sl.max(axis=0)
                              / np.maximum(weight[y0:y0 + step], 1e-6)
                              ).astype(np.float16)
    del prob, weight
    return mask[:h, :w], conf[:h, :w].astype(np.float32)
