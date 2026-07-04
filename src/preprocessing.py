from pathlib import Path

import cv2
import numpy as np
import tifffile


def load_image(path: str | Path) -> np.ndarray:
    """Load TIFF/PNG/JPEG/BMP of arbitrary bit depth -> uint8 RGB.

    Uses np.fromfile + imdecode instead of cv2.imread: the dataset lives in
    folders with Cyrillic names, which cv2.imread cannot open on Windows.
    """
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(str(path))
    else:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read {path}")
        if img.ndim == 3 and img.shape[2] >= 3:
            img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        lo, hi = np.percentile(img, (0.5, 99.5))
        img = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def normalize_illumination(img: np.ndarray, clip_limit: float = 2.0,
                           tile_grid: int = 16) -> np.ndarray:
    """CLAHE on L channel — evens out uneven panoramic stitching illumination."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)


def preprocess(img: np.ndarray, downscale: float = 1.0,
               illum: bool = True, do_denoise: bool = False) -> np.ndarray:
    if downscale != 1.0:
        img = cv2.resize(img, None, fx=downscale, fy=downscale,
                         interpolation=cv2.INTER_AREA)
    if illum:
        img = normalize_illumination(img)
    if do_denoise:
        img = denoise(img)
    return img


def sample_region_mask(img: np.ndarray) -> np.ndarray:
    """Binary mask of the actual polished section vs. empty scan padding.

    The provided panoramas are legitimately very dark (under-exposed ore
    matrix), so an Otsu split would throw away most of the specimen. Instead
    we only exclude *saturated* padding — near-black or near-white regions
    connected to the image border — which is what real scan margins look like.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pad = ((gray < 8) | (gray > 247)).astype(np.uint8)
    pad = cv2.morphologyEx(pad, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(pad, connectivity=8)
    h, w = gray.shape
    border = np.zeros_like(pad, bool)
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        big = stats[i, cv2.CC_STAT_AREA] > 0.002 * h * w
        if touches and big:
            border |= labels == i
    return (~border).astype(np.uint8)
