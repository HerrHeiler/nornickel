"""Train segmentation model on tiled crops.

Expected layout (produced by scripts/prepare_dataset.py):
  data/processed/images/*.png   RGB crops
  data/processed/masks/*.png    single-channel class-index masks (0..3)

Run:  python -m src.train
"""
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


def get_transforms(crop: int, train: bool):
    norm = [A.Normalize(), ToTensorV2()]
    if not train:
        return A.Compose([A.PadIfNeeded(crop, crop), A.CenterCrop(crop, crop), *norm])
    return A.Compose([
        A.PadIfNeeded(crop, crop),
        A.RandomCrop(crop, crop),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(alpha=40, sigma=6, p=0.2),
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


def main(cfg: TrainConfig | None = None):
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

    train_ds = OreDataset(train_pairs, get_transforms(cfg.crop_size, True))
    val_ds = OreDataset(val_pairs, get_transforms(cfg.crop_size, False))
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
    if ckpt_path.exists():
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
        print(f"  val mIoU={miou:.4f}  per-class={[f'{v:.3f}' for v in per_class]}")
        if miou > best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict(),
                        "miou": miou,
                        "cfg": vars(cfg)}, out_dir / "best.pt")
            print(f"  saved best.pt (mIoU={miou:.4f})")


if __name__ == "__main__":
    main()
