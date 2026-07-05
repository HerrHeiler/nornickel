"""Self-training on the unused ч2 оталькованные photos: the model's own
confident predictions become training crops. These photos have no expert
masks, come from a different camera than ч1, and are the dominant error
mode (talc missed) — pseudo-labels close exactly this domain gap.

Usage:
  python scripts/pseudo_label.py --weights models/exp5/best.pt \
      --out data/processed_v3 --limit 47 --min-conf 0.7
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import IMAGE_EXTS, InferenceConfig
from src.pipeline import OrePipeline
from scripts.prepare_dataset import tile_pair

SRC_DIR = "Фото руд по сортам. ч2/оталькованные"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="data/processed_v3")
    ap.add_argument("--limit", type=int, default=47,
                    help="first N photos (eval uses the folder tail)")
    ap.add_argument("--min-conf", type=float, default=0.7,
                    help="pixels below this confidence become background")
    ap.add_argument("--tile", type=int, default=512)
    args = ap.parse_args()

    cfg = InferenceConfig()
    cfg.weights_path = args.weights
    pipeline = OrePipeline(cfg)

    out_img = Path(args.out) / "images"
    out_mask = Path(args.out) / "masks"
    folder = Path(args.raw) / Path(SRC_DIR)
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)[:args.limit]

    total = 0
    for p in files:
        art = pipeline.process(p)
        mask = art["mask"].copy()
        # low-confidence pixels don't teach anything reliable
        mask[art["confidence"] < args.min_conf] = 0
        img = art["image"]        # CLAHE version = what the model trains on
        talc_frac = float((mask == 3).mean())
        total += tile_pair(img, mask, args.tile, out_img, out_mask,
                           f"pl_{p.stem}", min_fg_frac=0.02)
        print(f"{p.name}: talc={100 * talc_frac:.1f}%  crops so far={total}")
    print(f"Pseudo-label crops: {total}")


if __name__ == "__main__":
    main()
