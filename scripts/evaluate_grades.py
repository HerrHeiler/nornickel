"""End-to-end grade-classification evaluation on the folder-labeled photos.

The dataset has no pixel masks for sulfides, but it has ~1200 photos labeled
by ore grade at the image level (ч1: 42/68/68, ч2: 87/497/418). Running the
full pipeline on them and comparing the predicted grade with the folder label
gives the honest business metric (accuracy / macro-F1) and a way to tune the
segmenter without pixel annotations.

Usage:
  python scripts/evaluate_grades.py --raw data/raw --limit 30
  python scripts/evaluate_grades.py --raw data/raw --out reports/eval.csv
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (GRADE_FOLDERS, IMAGE_EXTS, InferenceConfig,
                        ORE_REFRACTORY, ORE_REGULAR, ORE_TALC)
from src.pipeline import OrePipeline

LABELS = [ORE_REGULAR, ORE_REFRACTORY, ORE_TALC]


def collect_files(raw: Path, limit: int | None, offset: int = 0,
                  ch2_only: bool = False):
    for rel, label in GRADE_FOLDERS.items():
        if ch2_only and "ч2" not in rel:
            continue
        folder = raw / Path(rel)
        if not folder.exists():
            print(f"skip missing folder: {folder}")
            continue
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        n = limit or len(files)
        if offset and len(files) > offset + n:
            files = files[offset:offset + n]
        else:
            files = files[-n:]
        for p in files:
            yield p, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--limit", type=int, default=None,
                    help="max images per folder (quick sweep)")
    ap.add_argument("--downscale", type=float, default=1.0,
                    help="keep 1.0: speck-size thresholds and U-Net training "
                         "crops are calibrated at native resolution")
    ap.add_argument("--out", default="reports/eval.csv")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip first N files per folder (avoid train overlap)")
    ap.add_argument("--ch2-only", action="store_true",
                    help="only ч2 folders (ч1 fully participates in training)")
    ap.add_argument("--weights", default=None,
                    help="checkpoint path override")
    args = ap.parse_args()

    cfg = InferenceConfig(downscale_factor=args.downscale)
    if args.weights:
        cfg.weights_path = args.weights
    pipeline = OrePipeline(cfg)

    rows = []
    t0 = time.time()
    for i, (path, label) in enumerate(collect_files(Path(args.raw), args.limit,
                                                    args.offset, args.ch2_only)):
        try:
            art = pipeline.process(path)
        except Exception as e:
            print(f"ERROR {path.name}: {e}")
            continue
        res = art["result"]
        rows.append({
            "file": str(path.relative_to(args.raw)),
            "true": label,
            "pred": res.ore_type,
            "talc_pct": round(res.talc_pct, 2),
            "sulfide_pct": round(res.sulfide_total_pct, 2),
            "regular_of_sulf": round(res.regular_of_sulfides_pct, 2),
            "fine_of_sulf": round(res.fine_of_sulfides_pct, 2),
            "sec": round(art["elapsed_s"], 2),
        })
        if (i + 1) % 25 == 0:
            print(f"{i + 1} images, {time.time() - t0:.0f}s elapsed")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n{len(df)} images evaluated -> {out}")
    cm = pd.crosstab(df["true"], df["pred"]).reindex(
        index=LABELS, columns=LABELS, fill_value=0)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)

    f1s = {}
    for c in LABELS:
        tp = int(((df["true"] == c) & (df["pred"] == c)).sum())
        fp = int(((df["true"] != c) & (df["pred"] == c)).sum())
        fn = int(((df["true"] == c) & (df["pred"] != c)).sum())
        f1s[c] = 2 * tp / max(2 * tp + fp + fn, 1)
    acc = float((df["true"] == df["pred"]).mean())
    print(f"\nAccuracy: {acc:.3f}")
    for c, v in f1s.items():
        print(f"F1 [{c}]: {v:.3f}")
    print(f"Macro-F1: {sum(f1s.values()) / len(f1s):.3f}")

    b = df[df["true"] != ORE_TALC]
    tp = int(((b["true"] == ORE_REFRACTORY) & (b["pred"] == ORE_REFRACTORY)).sum())
    fp = int(((b["true"] == ORE_REGULAR) & (b["pred"] == ORE_REFRACTORY)).sum())
    fn = int(((b["true"] == ORE_REFRACTORY) & (b["pred"] != ORE_REFRACTORY)).sum())
    f1_fine = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f"Intergrowth-type F1 (fine vs regular, non-talc photos): {f1_fine:.3f}")

    print("\nMean talc% by true class:")
    print(df.groupby("true")["talc_pct"].describe()[["mean", "50%", "min", "max"]])

    stats = defaultdict(float)
    stats["mean_sec"] = df["sec"].mean()
    print(f"\nMean processing time: {stats['mean_sec']:.2f}s per photo")


if __name__ == "__main__":
    main()
