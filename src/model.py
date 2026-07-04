from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import (CLASS_BACKGROUND, CLASS_FINE, CLASS_REGULAR, CLASS_TALC,
                     InferenceConfig)


def build_model(arch: str, encoder: str, num_classes: int,
                pretrained: bool = True):
    import segmentation_models_pytorch as smp
    factory = {
        "unet": smp.Unet,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
    }[arch]
    return factory(encoder_name=encoder,
                   encoder_weights="imagenet" if pretrained else None,
                   in_channels=3, classes=num_classes)


def load_model(cfg: InferenceConfig):
    """Returns None when no checkpoint exists — pipeline then uses the CV baseline.

    torch is imported lazily so the classical baseline (and dataset prep)
    works on machines without a torch install.
    """
    if not Path(cfg.weights_path).exists():
        return None
    import torch
    device = cfg.device if torch.cuda.is_available() else "cpu"
    state = torch.load(cfg.weights_path, map_location=device,
                       weights_only=False)
    saved = state.get("cfg", {})
    model = build_model(saved.get("arch", cfg.arch),
                        saved.get("encoder", cfg.encoder),
                        saved.get("num_classes", cfg.num_classes),
                        pretrained=False)
    model.load_state_dict(state.get("model", state))
    model.eval().to(device)
    return model


class ClassicalSegmenter:
    """Threshold + morphology segmentation encoding the geological priors:

    IMPORTANT: expects the *raw* (non-CLAHE) image. CLAHE manufactures
    local texture in smooth matrix and destroys the talc/host separation
    (measured on the 42 annotated pairs: at equal talc recall the false-
    positive rate on host matrix roughly doubles with CLAHE).

    * Sulfides are the bright reflective phase. The threshold adapts to
      exposure via the matrix median, so the same code works on normally
      exposed grade photos and on the very dark panoramas. Percentile
      thresholds would hallucinate a constant phase share on every image
      and make grade classification impossible.
    * Talc is the dark *dispersed* phase in the matrix: not just dark pixels
      (magnetite crystals and holes are dark too) but a high local density
      of fine dark specks — measured in a sliding window, calibrated on the
      42 expert-annotated talc pairs.
    * Regular vs. fine intergrowths are separated per connected sulfide grain
      by (a) grain size and (b) internal replacement ratio — the fraction of
      dark pixels inside the grain's convex hull (magnetite/gangue lamellae).
    """

    def __init__(self,
                 sulfide_rel: float = 1.8,
                 sulfide_abs_range: tuple = (120, 200),
                 dark_rel: float = 0.6,
                 speck_min_size: int = 5,
                 speck_max_size: int = 21,
                 density_window: int = 61,
                 density_thr: float = 0.05,
                 replacement_thr: float = 0.35,
                 min_regular_area: int = 2500,
                 min_grain_area: int = 64):
        self.sulfide_rel = sulfide_rel
        self.sulfide_abs_range = sulfide_abs_range
        self.dark_rel = dark_rel
        self.speck_min_size = speck_min_size
        self.speck_max_size = speck_max_size
        self.density_window = density_window
        self.density_thr = density_thr
        self.replacement_thr = replacement_thr
        self.min_regular_area = min_regular_area
        self.min_grain_area = min_grain_area

    def __call__(self, img: np.ndarray, sample_mask: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if sample_mask is None:
            sample_mask = np.ones_like(gray, np.uint8)

        med = float(np.median(gray[sample_mask > 0]))
        thr_sulf = float(np.clip(self.sulfide_rel * med, *self.sulfide_abs_range))
        sulfide = ((gray >= thr_sulf) & (sample_mask > 0)).astype(np.uint8)
        sulfide = cv2.morphologyEx(sulfide, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        matrix_med = float(np.median(gray[(sample_mask > 0) & (sulfide == 0)]))
        dark = ((gray <= self.dark_rel * matrix_med) & (sample_mask > 0)
                & (sulfide == 0)).astype(np.uint8)
        k = np.ones((self.speck_max_size, self.speck_max_size), np.uint8)
        massive = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
        specks = dark & (massive == 0)
        if self.speck_min_size > 1:
            specks = cv2.morphologyEx(
                specks, cv2.MORPH_OPEN,
                np.ones((self.speck_min_size, self.speck_min_size), np.uint8))
        density = cv2.boxFilter(specks.astype(np.float32), -1,
                                (self.density_window, self.density_window))
        talc = ((density >= self.density_thr) & (sample_mask > 0)
                & (sulfide == 0)).astype(np.uint8)
        talc = cv2.morphologyEx(talc, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        talc = cv2.morphologyEx(talc, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

        mask = np.full(gray.shape, CLASS_BACKGROUND, np.uint8)
        mask[talc > 0] = CLASS_TALC

        n, labels, stats, _ = cv2.connectedComponentsWithStats(sulfide, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_grain_area:
                continue
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            grain = (labels[y:y + h, x:x + w] == i).astype(np.uint8)

            hull = cv2.morphologyEx(grain, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            hull_area = int(hull.sum())
            replacement = 1.0 - area / max(hull_area, 1)

            cls = CLASS_REGULAR if (replacement < self.replacement_thr
                                    and area >= self.min_regular_area) else CLASS_FINE
            region = mask[y:y + h, x:x + w]
            region[grain > 0] = cls

        conf = np.where(mask != CLASS_BACKGROUND, 0.75, 0.9).astype(np.float32)
        return mask, conf
