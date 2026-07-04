"""Measurement calibration against expert data. Two independent parts:

1. --talc: predicted talc share vs expert markup on the 42 annotated pairs
   -> linear bias correction (gain/bias) + MAE / ±3% report.
   The expert 10% rule is never touched; we correct the *measurement*.

2. --rules <eval.csv>: sweep the fine-share threshold (what "преобладают
   тонкие" means in measured terms) on a grade-labeled evaluation CSV
   produced by scripts/evaluate_grades.py with identity calibration.

Both write into models/calibration.json, which the pipeline picks up.

Usage:
  python scripts/calibrate.py --talc
  python scripts/calibrate.py --rules reports/eval_dl3.csv
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (CALIB_PATH, DEFAULT_CALIB, ORE_REFRACTORY,
                        ORE_REGULAR, ORE_TALC, TALC_THRESHOLD_PCT, TrainConfig)

LABELS = [ORE_REGULAR, ORE_REFRACTORY, ORE_TALC]


def load_calib() -> dict:
    p = Path(CALIB_PATH)
    if p.exists():
        return {**DEFAULT_CALIB, **json.loads(p.read_text(encoding="utf-8"))}
    return dict(DEFAULT_CALIB)


def save_calib(calib: dict) -> None:
    p = Path(CALIB_PATH)
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    print(f"saved {p}: {calib}")


def val_stems() -> set[str]:
    """Reproduce the train/val split of src.train to tag in-sample photos."""
    cfg = TrainConfig()
    imgs = sorted((Path(cfg.data_root) / cfg.img_dir).glob("*.png"))
    stems = sorted({p.stem.rsplit("_", 1)[0] for p in imgs})
    random.seed(cfg.seed)
    random.shuffle(stems)
    return set(stems[:max(1, int(len(stems) * cfg.val_fraction))])


def calibrate_talc(raw_dir: str, limit: int | None = None) -> None:
    import cv2
    from src.config import InferenceConfig
    from src.model import ClassicalSegmenter
    from src.pipeline import OrePipeline
    from src.preprocessing import load_image
    from scripts.prepare_dataset import (annotation_line, fill_talc_regions,
                                         iter_annotated_pairs)

    pipeline = OrePipeline(InferenceConfig())
    pipeline.calib = dict(DEFAULT_CALIB)
    seg = ClassicalSegmenter()
    vstems = val_stems()

    rows = []
    pairs = list(iter_annotated_pairs(Path(raw_dir)))
    if limit:
        pairs = pairs[::max(1, len(pairs) // limit)][:limit]
    for annot_p, orig_p in pairs:
        annot = load_image(annot_p)
        zone = fill_talc_regions(annotation_line(annot))
        orig = load_image(orig_p)
        weak, _ = seg(orig)
        gt = 100.0 * float(((zone > 0) & (weak == 0)).mean())

        art = pipeline.process(orig_p)
        rows.append({
            "file": annot_p.name,
            "gt_talc": gt,
            "pred_talc": art["result"].talc_pct,
            "split": "val" if annot_p.stem in vstems else "train",
        })
        print(f"{annot_p.name}: gt={gt:.1f} pred={rows[-1]['pred_talc']:.1f} "
              f"[{rows[-1]['split']}]")

    df = pd.DataFrame(rows)
    df.to_csv("reports/talc_calibration.csv", index=False, encoding="utf-8-sig")

    gain, bias = np.polyfit(df["pred_talc"], df["gt_talc"], 1)
    corr = np.clip(gain * df["pred_talc"] + bias, 0, 100)

    def report(tag, gt, pred):
        err = np.abs(pred - gt)
        print(f"{tag}: MAE={err.mean():.2f}пп  median={np.median(err):.2f}пп  "
              f"within ±3пп: {(err <= 3).mean() * 100:.0f}%")

    print(f"\nfit: corrected = {gain:.3f} * measured + {bias:.2f}")
    report("raw        (all)", df["gt_talc"], df["pred_talc"])
    report("calibrated (all)", df["gt_talc"], corr)
    v = df[df["split"] == "val"]
    if len(v):
        report("calibrated (val)", v["gt_talc"],
               np.clip(gain * v["pred_talc"] + bias, 0, 100))

    calib = load_calib()
    calib["talc_gain"], calib["talc_bias"] = round(float(gain), 4), round(float(bias), 3)
    save_calib(calib)


def calibrate_rules(csv_path: str) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    calib = load_calib()
    talc_corr = np.clip(calib["talc_gain"] * df["talc_pct"] + calib["talc_bias"],
                        0, 100)

    def macro_f1(pred):
        f1s = []
        for c in LABELS:
            tp = int(((df["true"] == c) & (pred == c)).sum())
            fp = int(((df["true"] != c) & (pred == c)).sum())
            fn = int(((df["true"] == c) & (pred != c)).sum())
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        return float(np.mean(f1s)), f1s

    best = None
    for thr in np.arange(20, 72.5, 2.5):
        pred = np.where(talc_corr > TALC_THRESHOLD_PCT, ORE_TALC,
                        np.where(df["fine_of_sulf"] <= thr,
                                 ORE_REGULAR, ORE_REFRACTORY))
        mf1, f1s = macro_f1(pd.Series(pred))
        acc = float((pred == df["true"]).mean())
        if best is None or mf1 > best[1]:
            best = (thr, mf1, acc, f1s)
        print(f"thr={thr:5.1f}  macroF1={mf1:.3f}  acc={acc:.3f}")

    thr, mf1, acc, f1s = best
    print(f"\nbest fine_ratio_thr={thr}: macro-F1={mf1:.3f} acc={acc:.3f} "
          f"per-class={['%.3f' % v for v in f1s]}")
    calib["fine_ratio_thr"] = float(thr)
    save_calib(calib)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--talc", action="store_true")
    ap.add_argument("--rules", metavar="EVAL_CSV")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only every k-th annotated pair (quick run)")
    args = ap.parse_args()
    Path("reports").mkdir(exist_ok=True)
    if args.talc:
        calibrate_talc(args.raw, args.limit)
    if args.rules:
        calibrate_rules(args.rules)
    if not args.talc and not args.rules:
        print("nothing to do: pass --talc and/or --rules <csv>")
