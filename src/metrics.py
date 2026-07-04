from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (CLASS_FINE, CLASS_NAMES_RU, CLASS_REGULAR, CLASS_TALC,
                     DEFAULT_CALIB, ORE_REFRACTORY, ORE_REGULAR, ORE_TALC,
                     TALC_THRESHOLD_PCT)


@dataclass
class ClassificationResult:
    ore_type: str
    talc_pct: float
    sulfide_total_pct: float
    regular_pct: float
    fine_pct: float
    regular_of_sulfides_pct: float
    fine_of_sulfides_pct: float
    metrics_df: pd.DataFrame
    conclusion: str


def compute_metrics(mask: np.ndarray, sample_mask: np.ndarray | None = None,
                    px_size_um: float | None = None,
                    calib: dict | None = None) -> ClassificationResult:
    calib = {**DEFAULT_CALIB, **(calib or {})}
    valid = np.ones(mask.shape, bool) if sample_mask is None else sample_mask.astype(bool)
    total = int(valid.sum())

    counts = {c: int(((mask == c) & valid).sum())
              for c in (CLASS_REGULAR, CLASS_FINE, CLASS_TALC)}
    pct = {c: 100.0 * v / max(total, 1) for c, v in counts.items()}

    pct[CLASS_TALC] = float(np.clip(
        calib["talc_gain"] * pct[CLASS_TALC] + calib["talc_bias"], 0.0, 100.0))

    sulf_px = counts[CLASS_REGULAR] + counts[CLASS_FINE]
    sulf_pct = 100.0 * sulf_px / max(total, 1)
    reg_of_sulf = 100.0 * counts[CLASS_REGULAR] / max(sulf_px, 1)
    fine_of_sulf = 100.0 * counts[CLASS_FINE] / max(sulf_px, 1)

    if pct[CLASS_TALC] > TALC_THRESHOLD_PCT:
        ore_type = ORE_TALC
    elif fine_of_sulf <= calib["fine_ratio_thr"]:
        ore_type = ORE_REGULAR
    else:
        ore_type = ORE_REFRACTORY

    rows = []
    for c in (CLASS_REGULAR, CLASS_FINE, CLASS_TALC):
        row = {
            "Фаза": CLASS_NAMES_RU[c],
            "Площадь, пикс.": counts[c],
            "Доля от площади шлифа, %": round(pct[c], 2),
        }
        if c in (CLASS_REGULAR, CLASS_FINE):
            row["Доля среди сульфидов, %"] = round(
                (reg_of_sulf if c == CLASS_REGULAR else fine_of_sulf), 2)
        else:
            row["Доля среди сульфидов, %"] = None
        if px_size_um:
            row["Площадь, мм²"] = round(counts[c] * (px_size_um / 1000.0) ** 2, 4)
        rows.append(row)
    rows.append({
        "Фаза": "Сульфиды (всего)",
        "Площадь, пикс.": sulf_px,
        "Доля от площади шлифа, %": round(sulf_pct, 2),
        "Доля среди сульфидов, %": 100.0,
        **({"Площадь, мм²": round(sulf_px * (px_size_um / 1000.0) ** 2, 4)}
           if px_size_um else {}),
    })
    df = pd.DataFrame(rows)

    conclusion = _build_conclusion(ore_type, pct[CLASS_TALC], reg_of_sulf, fine_of_sulf)

    return ClassificationResult(
        ore_type=ore_type,
        talc_pct=pct[CLASS_TALC],
        sulfide_total_pct=sulf_pct,
        regular_pct=pct[CLASS_REGULAR],
        fine_pct=pct[CLASS_FINE],
        regular_of_sulfides_pct=reg_of_sulf,
        fine_of_sulfides_pct=fine_of_sulf,
        metrics_df=df,
        conclusion=conclusion,
    )


def _build_conclusion(ore_type: str, talc: float, reg: float, fine: float) -> str:
    if ore_type == ORE_TALC:
        return (f"Руда классифицирована как {ore_type}: содержание талька — "
                f"{talc:.0f}% (порог {TALC_THRESHOLD_PCT:.0f}%).")
    if ore_type == ORE_REGULAR:
        return (f"Руда классифицирована как {ore_type}: содержание талька — "
                f"{talc:.0f}%, преобладание обычных срастаний — {reg:.0f}% "
                f"от общей площади сульфидов.")
    return (f"Руда классифицирована как {ore_type}: содержание талька — "
            f"{talc:.0f}%, преобладание тонких срастаний — {fine:.0f}% "
            f"от общей площади сульфидов.")
