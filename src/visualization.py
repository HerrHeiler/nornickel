import cv2
import numpy as np

from .config import CLASS_COLORS_RGB


def make_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = img.copy()
    color_layer = np.zeros_like(img)
    hit = np.zeros(mask.shape, bool)
    for cls, rgb in CLASS_COLORS_RGB.items():
        m = mask == cls
        color_layer[m] = rgb
        hit |= m
    overlay[hit] = (img[hit] * (1 - alpha) + color_layer[hit] * alpha).astype(np.uint8)
    return overlay


def confidence_heatmap(conf: np.ndarray) -> np.ndarray:
    c8 = np.clip(conf * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(255 - c8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def downscale_for_display(img: np.ndarray, max_side: int = 2048) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def cleanup_mask(mask: np.ndarray, min_px: int = 64) -> np.ndarray:
    """Drop connected components below min_px per foreground class."""
    out = mask.copy()
    for cls in CLASS_COLORS_RGB:
        binary = (mask == cls).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_px:
                out[labels == i] = 0
    return out
