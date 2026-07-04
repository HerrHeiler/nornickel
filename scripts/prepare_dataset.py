"""Convert the provided Yandex Disk dataset into (image, mask) training crops.

Actual data layout (verified on the real data):

  data/raw/
    Панорамы/                                  14 unlabeled panoramas ~14000x10800
    Фото руд по сортам. ч1/
      Оталькованные руды/                      42 originals 2272x1704
        Области оталькования/                  42 same-named copies with talc zones
                                               outlined by a pure-blue line
                                               (HSV H=120 S=255 V=254, ~5-7 px)
      Рядовые руды/                            68 image-level labeled
      Труднообогатимые руды/                   68 image-level labeled
    Фото руд по сортам. ч2/                    ~1000 image-level labeled photos

Pixel-accurate talc masks are recovered from the blue outlines. Outlines are
often OPEN curves that terminate on the image border, so we treat the border
as part of the barrier: regions are the connected components of the complement
of (blue line ∪ border), and a region is talc when most of its boundary
contact is with the annotation line rather than with the raw border.

Sulfide classes (regular/fine) have no pixel labels; they are weak-labeled
with the ClassicalSegmenter on the line-inpainted image.

Usage:
  python scripts/prepare_dataset.py --raw data/raw --out data/processed --tile 512
  python scripts/prepare_dataset.py --debug   # writes fill-QA overlays only
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ANNOT_HSV_HI, ANNOT_HSV_LO, CLASS_TALC
from src.model import ClassicalSegmenter
from src.preprocessing import load_image, preprocess

TALC_SUBDIR = "Области оталькования"
TALC_DIR = "Фото руд по сортам. ч1/Оталькованные руды"
NEG_DIRS = [
    "Фото руд по сортам. ч1/Рядовые руды",
    "Фото руд по сортам. ч1/Труднообогатимые руды",
]


def annotation_line(img_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    line = cv2.inRange(hsv, np.array(ANNOT_HSV_LO), np.array(ANNOT_HSV_HI))
    line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return (line > 0).astype(np.uint8)


def close_open_curves(line: np.ndarray) -> np.ndarray:
    """Extend open stroke ends along their direction until they hit the
    image border or another stroke.

    Annotators rely on the viewer mentally continuing a stroke to the edge
    (corner cut-offs, corridors between two curves); without this the
    complement stays one connected region and no talc can be recovered.
    """
    from skimage.morphology import skeletonize

    skel = skeletonize(line > 0).astype(np.uint8)
    k8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    for _ in range(12):
        nb = cv2.filter2D(skel, -1, k8)
        skel[(skel == 1) & (nb <= 1)] = 0
    nb = cv2.filter2D(skel, -1, k8)
    ends = np.argwhere((skel == 1) & (nb == 1))

    H, W = line.shape
    out = line.copy()
    for y, x in ends:
        y0, x0 = max(0, y - 30), max(0, x - 30)
        win = skel[y0:y + 31, x0:x + 31]
        pts = np.argwhere(win == 1)
        if len(pts) < 5:
            continue
        cy, cx = pts.mean(axis=0) + (y0, x0)
        d = np.array([y - cy, x - cx], float)
        norm = float(np.hypot(*d))
        if norm < 2:
            continue
        d /= norm
        p = np.array([y, x], float)
        max_extend = 350
        path = []
        committed = False
        for step in range(max_extend):
            p += d
            iy, ix = int(round(p[0])), int(round(p[1]))
            if not (0 <= iy < H and 0 <= ix < W):
                committed = True
                break
            if step > 40 and line[iy, ix]:
                committed = True
                break
            path.append((ix, iy))
        if committed:
            for pt in path:
                cv2.circle(out, pt, 3, 1, -1)
    return out


def fill_talc_regions(line: np.ndarray, min_region_px: int = 2000
                      ) -> np.ndarray:
    """Recover talc zones from the boundary curves drawn by the geologist.

    Verified convention across the 42 annotated pairs: each curve separates
    a talc zone from the host field — closed loops around talc patches,
    corner cut-offs closed by the image border, corridors between two
    curves. The host field is the single largest region of the partition;
    everything else is talc.
    """
    closed = close_open_curves(line)
    barrier = closed.copy()
    t = 4
    barrier[:t, :] = barrier[-t:, :] = 1
    barrier[:, :t] = barrier[:, -t:] = 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (barrier == 0).astype(np.uint8), connectivity=4)
    talc = np.zeros_like(line)
    if n <= 2:
        return talc
    areas = stats[1:, cv2.CC_STAT_AREA]
    host = 1 + int(np.argmax(areas))
    for i in range(1, n):
        if i == host or stats[i, cv2.CC_STAT_AREA] < min_region_px:
            continue
        talc[labels == i] = 1

    talc = cv2.morphologyEx(talc, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    talc[cv2.dilate(talc, np.ones((3, 3), np.uint8)).astype(bool)
         & (closed > 0)] = 1
    return talc


def extract_talc_from_markup(annot_rgb: np.ndarray,
                             orig_rgb: np.ndarray | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (clean_image, talc_mask). Prefers the pixel-identical original
    from the parent folder over inpainting when it is available."""
    line = annotation_line(annot_rgb)
    talc = fill_talc_regions(line)
    if orig_rgb is not None and orig_rgb.shape == annot_rgb.shape:
        return orig_rgb, talc
    clean = cv2.inpaint(cv2.cvtColor(annot_rgb, cv2.COLOR_RGB2BGR),
                        (line * 255).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(clean, cv2.COLOR_BGR2RGB), talc


def _write_png(path: Path, arr: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise IOError(f"PNG encode failed for {path}")
    buf.tofile(str(path))


def tile_pair(img, mask, tile, out_img_dir, out_mask_dir, stem,
              min_fg_frac=0.0):
    h, w = img.shape[:2]
    k = 0
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            m = mask[y:y + tile, x:x + tile]
            if min_fg_frac and (m > 0).mean() < min_fg_frac:
                continue
            _write_png(out_img_dir / f"{stem}_{k:05d}.png",
                       cv2.cvtColor(img[y:y + tile, x:x + tile],
                                    cv2.COLOR_RGB2BGR))
            _write_png(out_mask_dir / f"{stem}_{k:05d}.png", m)
            k += 1
    return k


def iter_annotated_pairs(raw: Path):
    talc_dir = raw / Path(TALC_DIR)
    annot_dir = talc_dir / TALC_SUBDIR
    origs = {p.name.lower(): p for p in talc_dir.iterdir() if p.is_file()}
    for annot_p in sorted(annot_dir.iterdir()):
        if not annot_p.is_file():
            continue
        yield annot_p, origs.get(annot_p.name.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--neg-limit", type=int, default=30,
                    help="photos per talc-free folder used as negatives")
    ap.add_argument("--debug", action="store_true",
                    help="only write talc-fill QA overlays to <out>/debug")
    args = ap.parse_args()

    raw = Path(args.raw)
    out_img = Path(args.out) / "images"
    out_mask = Path(args.out) / "masks"
    dbg_dir = Path(args.out) / "debug"
    for d in ([dbg_dir] if args.debug else [out_img, out_mask]):
        d.mkdir(parents=True, exist_ok=True)

    seg = ClassicalSegmenter()
    total = 0
    for annot_p, orig_p in iter_annotated_pairs(raw):
        annot = load_image(annot_p)
        orig = load_image(orig_p) if orig_p else None
        clean, talc = extract_talc_from_markup(annot, orig)

        if args.debug:
            vis = clean.copy()
            vis[talc > 0] = (0.5 * vis[talc > 0]
                             + 0.5 * np.array([0, 80, 255])).astype(np.uint8)
            cv2.imwrite(str(dbg_dir / f"{annot_p.stem}_talc.jpg"),
                        cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{annot_p.name}: talc={100 * talc.mean():.1f}%")
            continue

        weak_mask, _ = seg(clean)
        weak_mask[(talc > 0) & (weak_mask == 0)] = CLASS_TALC

        model_input = preprocess(clean, illum=True)
        total += tile_pair(model_input, weak_mask, args.tile, out_img, out_mask,
                           annot_p.stem, min_fg_frac=0.01)
        print(f"{annot_p.name}: talc={100 * talc.mean():.1f}%  crops so far={total}")

    if args.debug:
        return

    from src.config import IMAGE_EXTS
    for rel in NEG_DIRS:
        folder = raw / Path(rel)
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        for p in files[:args.neg_limit]:
            img = load_image(p)
            weak_mask, _ = seg(img)
            weak_mask[weak_mask == CLASS_TALC] = 0
            model_input = preprocess(img, illum=True)
            total += tile_pair(model_input, weak_mask, args.tile, out_img,
                               out_mask, f"neg_{p.stem}", min_fg_frac=0.01)
        print(f"{rel}: negatives done, crops so far={total}")

    print(f"Total crops: {total}")


if __name__ == "__main__":
    main()
