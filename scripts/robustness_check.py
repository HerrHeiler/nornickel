"""Robustness harness for the spec's «сложные случаи»: uneven illumination,
scratches, dirt/contamination, blur, exposure shifts.

Applies synthetic perturbations to grade-labeled photos and reports whether
the ore verdict survives and how far the measured talc share drifts.

Usage:
  python scripts/robustness_check.py --raw data/raw --per-class 4
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import GRADE_FOLDERS, IMAGE_EXTS, InferenceConfig
from src.pipeline import OrePipeline
from src.preprocessing import load_image

RNG = np.random.default_rng(7)


def uneven_illumination(img):
    """Strong diagonal illumination gradient ×[0.55 .. 1.35]."""
    h, w = img.shape[:2]
    gy = np.linspace(0.55, 1.35, h)[:, None]
    gx = np.linspace(0.9, 1.1, w)[None, :]
    return np.clip(img.astype(np.float32) * (gy * gx)[..., None],
                   0, 255).astype(np.uint8)


def scratches(img):
    """Bright thin polishing scratches across the section."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(12):
        p1 = (int(RNG.integers(0, w)), int(RNG.integers(0, h)))
        ang = RNG.uniform(0, np.pi)
        length = int(RNG.integers(w // 4, w))
        p2 = (int(p1[0] + length * np.cos(ang)), int(p1[1] + length * np.sin(ang)))
        color = tuple(int(v) for v in RNG.integers(190, 240, 3))
        cv2.line(out, p1, p2, color, int(RNG.integers(1, 4)), cv2.LINE_AA)
    return out


def dirt(img):
    """Dark dust blobs and smudges on the section surface."""
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(60):
        c = (int(RNG.integers(0, w)), int(RNG.integers(0, h)))
        r = int(RNG.integers(2, 12))
        shade = int(RNG.integers(5, 40))
        cv2.circle(out, c, r, (shade, shade, shade), -1, cv2.LINE_AA)
    return out


def blur(img):
    return cv2.GaussianBlur(img, (7, 7), 2.0)


def underexposed(img):
    return np.clip(img.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)


PERTURBATIONS = {
    "illumination": uneven_illumination,
    "scratches": scratches,
    "dirt": dirt,
    "blur": blur,
    "underexposed": underexposed,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--out", default="reports/robustness.csv")
    args = ap.parse_args()

    pipeline = OrePipeline(InferenceConfig())
    rows = []
    tmpdir = Path(tempfile.mkdtemp(prefix="robust_"))

    for rel, label in GRADE_FOLDERS.items():
        if "ч1" not in rel:
            continue
        folder = Path(args.raw) / Path(rel)
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        for p in files[:args.per_class]:
            base = pipeline.process(p)["result"]
            img = load_image(p)
            for name, fn in PERTURBATIONS.items():
                pert = fn(img)
                tmp = tmpdir / f"{name}.png"
                cv2.imencode(".png",
                             cv2.cvtColor(pert, cv2.COLOR_RGB2BGR))[1].tofile(str(tmp))
                res = pipeline.process(tmp)["result"]
                rows.append({
                    "file": p.name, "true": label, "perturbation": name,
                    "verdict_base": base.ore_type, "verdict_pert": res.ore_type,
                    "verdict_stable": base.ore_type == res.ore_type,
                    "talc_base": round(base.talc_pct, 2),
                    "talc_pert": round(res.talc_pct, 2),
                    "talc_drift": round(abs(res.talc_pct - base.talc_pct), 2),
                })
                print(f"{p.name} [{name}]: stable={rows[-1]['verdict_stable']} "
                      f"talc drift={rows[-1]['talc_drift']:.1f}пп")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("\nPer-perturbation summary:")
    print(df.groupby("perturbation").agg(
        stable=("verdict_stable", "mean"),
        mean_talc_drift=("talc_drift", "mean"),
        max_talc_drift=("talc_drift", "max"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
