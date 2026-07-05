"""Score an evaluation CSV: best macro-F1 over the calibration grid
(talc gain x fine-share threshold). Fair A/B comparison of model variants -
each is judged at its own optimal measurement calibration.

Usage: python scripts/score_experiment.py reports/exp_e0.csv [more.csv ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (ORE_REFRACTORY, ORE_REGULAR, ORE_TALC,
                        TALC_THRESHOLD_PCT)

LABELS = [ORE_REGULAR, ORE_REFRACTORY, ORE_TALC]


def macro_f1(true, pred):
    f1s = []
    for c in LABELS:
        tp = int(((true == c) & (pred == c)).sum())
        fp = int(((true != c) & (pred == c)).sum())
        fn = int(((true == c) & (pred != c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s)), f1s


def score(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    best = None
    for gain in np.arange(0.25, 1.55, 0.05):
        talc = np.clip(gain * df["talc_pct"], 0, 100)
        for thr in np.arange(15, 72.5, 2.5):
            pred = pd.Series(np.where(
                talc > TALC_THRESHOLD_PCT, ORE_TALC,
                np.where(df["fine_of_sulf"] <= thr, ORE_REGULAR,
                         ORE_REFRACTORY)))
            mf1, f1s = macro_f1(df["true"], pred)
            acc = float((pred.values == df["true"].values).mean())
            if best is None or mf1 > best[0]:
                best = (mf1, acc, gain, thr, f1s, pred)
    mf1, acc, gain, thr, f1s, pred = best
    name = Path(csv_path).stem
    print(f"{name}: n={len(df)}  best macroF1={mf1:.3f} acc={acc:.3f} "
          f"(gain={gain:.2f}, fine_thr={thr:.1f}) "
          f"per-class={['%.2f' % v for v in f1s]}")
    cm = pd.crosstab(df["true"], pred).reindex(index=LABELS, columns=LABELS,
                                               fill_value=0)
    print(cm.to_string().encode("ascii", "replace").decode())
    return mf1


if __name__ == "__main__":
    for p in sys.argv[1:]:
        score(p)
        print()
