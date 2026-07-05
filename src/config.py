from dataclasses import dataclass, field

CLASS_BACKGROUND = 0
CLASS_REGULAR = 1
CLASS_FINE = 2
CLASS_TALC = 3

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_REGULAR: "sulfide_regular",
    CLASS_FINE: "sulfide_fine",
    CLASS_TALC: "talc",
}

CLASS_NAMES_RU = {
    CLASS_REGULAR: "Обычные срастания",
    CLASS_FINE: "Тонкие срастания",
    CLASS_TALC: "Тальк",
}

CLASS_COLORS_RGB = {
    CLASS_REGULAR: (0, 200, 0),
    CLASS_FINE: (220, 0, 0),
    CLASS_TALC: (0, 80, 255),
}

TALC_THRESHOLD_PCT = 10.0

CALIB_PATH = "models/calibration.json"
DEFAULT_CALIB = {
    "talc_gain": 1.0,
    "talc_bias": 0.0,
    "cv_talc_gain": 0.0,      # hybrid: weight of the classical talc measure
    "fine_ratio_thr": 50.0,
}

ANNOT_HSV_LO = (110, 150, 120)
ANNOT_HSV_HI = (130, 255, 255)

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

ORE_TALC = "оталькованная руда"
ORE_REGULAR = "рядовая руда"
ORE_REFRACTORY = "труднообогатимая руда"

GRADE_FOLDERS = {
    "Фото руд по сортам. ч1/Оталькованные руды": ORE_TALC,
    "Фото руд по сортам. ч1/Рядовые руды": ORE_REGULAR,
    "Фото руд по сортам. ч1/Труднообогатимые руды": ORE_REFRACTORY,
    "Фото руд по сортам. ч2/оталькованные": ORE_TALC,
    "Фото руд по сортам. ч2/рядовые": ORE_REGULAR,
    "Фото руд по сортам. ч2/тонкие": ORE_REFRACTORY,
}


@dataclass
class InferenceConfig:
    tile_size: int = 1024
    overlap: int = 256
    batch_size: int = 8
    use_amp: bool = False
    tta_hflip: bool = False
    device: str = "cuda"
    num_classes: int = 4
    encoder: str = "efficientnet-b3"
    arch: str = "unet"
    weights_path: str = "models/best.pt"
    downscale_factor: float = 1.0
    max_mpx: float = 100.0
    min_object_px: int = 64


@dataclass
class TrainConfig:
    data_root: str = "data/processed"
    img_dir: str = "images"
    mask_dir: str = "masks"
    crop_size: int = 384
    batch_size: int = 4
    epochs: int = 25
    lr: float = 3e-4
    use_amp: bool = False
    encoder: str = "efficientnet-b0"
    arch: str = "unet"
    num_classes: int = 4
    val_fraction: float = 0.15
    out_dir: str = "models"
    seed: int = 42
    class_weights: tuple = (0.5, 1.0, 1.5, 1.5)
