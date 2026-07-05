import json
import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import InferenceConfig
from .metrics import ClassificationResult, compute_metrics
from .model import ClassicalSegmenter, load_model
from .preprocessing import load_image, preprocess, sample_region_mask
from .visualization import cleanup_mask, make_overlay

log = logging.getLogger("ore")


class OrePipeline:
    """load → preprocess → tiled segmentation → cleanup → metrics → overlay."""

    def __init__(self, cfg: InferenceConfig | None = None):
        self.cfg = cfg or InferenceConfig()
        try:
            import torch
            if not torch.cuda.is_available():
                self.cfg.device = "cpu"
        except Exception:
            self.cfg.device = "cpu"
        self.model = load_model(self.cfg)
        self.baseline = ClassicalSegmenter()
        self.calib = self._load_calibration()
        log.info("Backend: %s", "DL model" if self.model else "classical baseline")

    @staticmethod
    def _load_calibration() -> dict:
        from .config import CALIB_PATH, DEFAULT_CALIB
        p = Path(CALIB_PATH)
        if p.exists():
            try:
                return {**DEFAULT_CALIB, **json.loads(p.read_text(encoding="utf-8"))}
            except (json.JSONDecodeError, OSError) as e:
                log.warning("calibration.json unreadable (%s), using defaults", e)
        return dict(DEFAULT_CALIB)

    def process(self, image_path: str | Path,
                px_size_um: float | None = None,
                progress_cb: Callable[[float, str], None] | None = None
                ) -> dict:
        t0 = time.time()
        cb = progress_cb or (lambda p, s: None)

        cb(0.05, "Загрузка изображения")
        raw = load_image(image_path)
        raw = preprocess(raw, downscale=self.cfg.downscale_factor, illum=False)
        mpx = raw.shape[0] * raw.shape[1] / 1e6
        if mpx > self.cfg.max_mpx:
            s = (self.cfg.max_mpx / mpx) ** 0.5
            raw = preprocess(raw, downscale=s, illum=False)
            log.info("auto-downscale ×%.2f (%.0f→%.0f Mpx)", s, mpx, self.cfg.max_mpx)
        img = preprocess(raw, illum=True)

        cb(0.10, "Определение области шлифа")
        smask = sample_region_mask(raw)

        cb(0.15, "Сегментация")
        cv_talc = None
        if self.model is not None:
            from .tiling import predict_large_image
            mask, conf = predict_large_image(
                img, self.model, self.cfg.num_classes,
                tile=self.cfg.tile_size, overlap=self.cfg.overlap,
                batch_size=self.cfg.batch_size, device=self.cfg.device,
                use_amp=self.cfg.use_amp,
                progress_cb=lambda p: cb(0.15 + 0.6 * p, "Сегментация"),
            )
            mask[smask == 0] = 0
            # second leg of the hybrid talc measure (see compute_metrics)
            cv_talc = self.baseline.talc_share(raw, smask)
        else:
            mask, conf = self.baseline(raw, smask)

        cb(0.80, "Постобработка")
        mask = cleanup_mask(mask, self.cfg.min_object_px)

        cb(0.85, "Расчёт метрик")
        result = compute_metrics(mask, smask, px_size_um, calib=self.calib,
                                 cv_talc_pct=cv_talc)

        cb(0.92, "Построение визуализации")
        overlay = make_overlay(img, mask)

        elapsed = time.time() - t0
        log.info("Processed %s in %.1fs -> %s", image_path, elapsed, result.ore_type)
        cb(1.0, "Готово")

        return {
            "image": img,
            "mask": mask,
            "confidence": conf,
            "overlay": overlay,
            "result": result,
            "elapsed_s": elapsed,
        }

    def export(self, out_dir: str | Path, name: str, artifacts: dict) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        res: ClassificationResult = artifacts["result"]
        paths = {}

        paths["mask"] = out / f"{name}_mask.png"
        cv2.imencode(".png", artifacts["mask"])[1].tofile(str(paths["mask"]))

        paths["overlay"] = out / f"{name}_overlay.jpg"
        cv2.imencode(".jpg",
                     cv2.cvtColor(artifacts["overlay"], cv2.COLOR_RGB2BGR),
                     [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(paths["overlay"]))

        paths["metrics"] = out / f"{name}_metrics.csv"
        res.metrics_df.to_csv(paths["metrics"], index=False)

        paths["report"] = out / f"{name}_report.json"
        paths["report"].write_text(json.dumps({
            "ore_type": res.ore_type,
            "conclusion": res.conclusion,
            "talc_pct": round(res.talc_pct, 2),
            "sulfide_total_pct": round(res.sulfide_total_pct, 2),
            "regular_of_sulfides_pct": round(res.regular_of_sulfides_pct, 2),
            "fine_of_sulfides_pct": round(res.fine_of_sulfides_pct, 2),
            "elapsed_s": round(artifacts["elapsed_s"], 1),
            "config": vars(self.cfg),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return paths


def batch_process(pipeline: OrePipeline, input_dir: str | Path,
                  out_dir: str | Path) -> None:
    from .config import IMAGE_EXTS
    files = sorted(p for p in Path(input_dir).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    for p in files:
        artifacts = pipeline.process(p)
        pipeline.export(out_dir, p.stem, artifacts)
