"""Train segmentation model on tiled crops.

Expected layout (produced by scripts/prepare_dataset.py):
  data/processed/images/*.png   RGB crops
  data/processed/masks/*.png    single-channel class-index masks (0..3)

Run:  python -m src.train [--data ...] [--out ...] [--epochs N]
                          [--aug base|strong] [--fine-weight W] [--resume]
"""
import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import TrainConfig
from .model import build_model


class OreDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_p, mask_p = self.pairs[i]
        img = cv2.imdecode(np.fromfile(str(img_p), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imdecode(np.fromfile(str(mask_p), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].long()


def get_transforms(crop: int, train: bool, aug: str = "base"):
    norm = [A.Normalize(), ToTensorV2()]
    if not train:
        return A.Compose([A.PadIfNeeded(crop, crop), A.CenterCrop(crop, crop), *norm])
    geometric = [
        A.PadIfNeeded(crop, crop),
        A.RandomCrop(crop, crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    if aug == "base":
        return A.Compose([
            *geometric,
            A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
            A.GaussNoise(p=0.3),
            A.ElasticTransform(alpha=40, sigma=6, p=0.2),
            *norm,
        ])
    return A.Compose([
        *geometric,
        A.RandomBrightnessContrast(0.35, 0.3, p=0.85),
        A.RandomGamma(gamma_limit=(55, 165), p=0.5),
        A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.4),
        A.OneOf([A.MotionBlur(blur_limit=5),
                 A.GaussianBlur(blur_limit=(3, 5)),
                 A.Defocus(radius=(2, 4))], p=0.3),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2),
                       shadow_dimension=5, p=0.25),
        A.ElasticTransform(alpha=40, sigma=6, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        *norm,
    ])


class DiceCE(torch.nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32))
        import segmentation_models_pytorch as smp
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for c in range(num_classes):
            p, t = pred == c, y == c
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()
    iou = inter / union.clamp(min=1)
    return iou.mean().item(), iou.tolist()


def main(cfg: TrainConfig | None = None, aug: str = "base",
         resume: bool = False, oversample_talc: float = 1.0):
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    root = Path(cfg.data_root)
    imgs = sorted((root / cfg.img_dir).glob("*.png"))
    pairs = [(p, root / cfg.mask_dir / p.name) for p in imgs]
    stems = sorted({p.stem.rsplit("_", 1)[0] for p, _ in pairs})
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * cfg.val_fraction))
    val_stems = set(stems[:n_val])
    val_pairs = [pr for pr in pairs if pr[0].stem.rsplit("_", 1)[0] in val_stems]
    train_pairs = [pr for pr in pairs if pr[0].stem.rsplit("_", 1)[0] not in val_stems]

    train_ds = OreDataset(train_pairs, get_transforms(cfg.crop_size, True, aug))
    val_ds = OreDataset(val_pairs, get_transforms(cfg.crop_size, False))
    if oversample_talc > 1.0:
        from torch.utils.data import WeightedRandomSampler
        w = [oversample_talc if not p.stem.startswith(("neg_", "ch2_"))
             else 1.0 for p, _ in train_pairs]
        sampler = WeightedRandomSampler(w, num_samples=len(train_pairs),
                                        replacement=True)
        train_dl = DataLoader(train_ds, cfg.batch_size, sampler=sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, cfg.batch_size, num_workers=0)

    model = build_model(cfg.arch, cfg.encoder, cfg.num_classes).to(device)
    criterion = DiceCE(cfg.class_weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_miou = 0.0
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        best_miou = state.get("miou", 0.0)
        print(f"resumed from best.pt (mIoU={best_miou:.4f})")

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()

        miou, per_class = evaluate(model, val_dl, cfg.num_classes, device)
        # select by mean IoU of the ORE classes: the easy background class
        # otherwise dominates the average and masks a talc collapse
        fg_miou = float(sum(per_class[1:]) / (cfg.num_classes - 1))
        print(f"  val mIoU={miou:.4f} fg={fg_miou:.4f} "
              f"per-class={[f'{v:.3f}' for v in per_class]}")
        if fg_miou > best_miou:
            best_miou = fg_miou
            miou = fg_miou
            torch.save({"model": model.state_dict(),
                        "miou": miou,
                        "cfg": vars(cfg)}, out_dir / "best.pt")
            print(f"  saved best.pt (mIoU={miou:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--aug", choices=["base", "strong"], default="base")
    ap.add_argument("--fine-weight", type=float, default=None)
    ap.add_argument("--talc-weight", type=float, default=None)
    ap.add_argument("--oversample-talc", type=float, default=1.0,
                    help="sampling boost for talc-annotated source crops")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.data:
        cfg.data_root = args.data
    if args.out:
        cfg.out_dir = args.out
    if args.epochs:
        cfg.epochs = args.epochs
    if args.encoder:
        cfg.encoder = args.encoder
    if args.fine_weight is not None:
        w = list(cfg.class_weights)
        w[2] = args.fine_weight
        cfg.class_weights = tuple(w)
    if args.talc_weight is not None:
        w = list(cfg.class_weights)
        w[3] = args.talc_weight
        cfg.class_weights = tuple(w)
    main(cfg, aug=args.aug, resume=args.resume,
         oversample_talc=args.oversample_talc)
