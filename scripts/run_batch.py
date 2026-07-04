"""Batch mode: python scripts/run_batch.py --input data/raw --output outputs"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import OrePipeline, batch_process

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs")
    args = ap.parse_args()
    batch_process(OrePipeline(), args.input, args.output)
