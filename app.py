import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ORE_REFRACTORY, ORE_TALC, InferenceConfig
from src.metrics import compute_metrics
from src.pipeline import OrePipeline
from src.visualization import confidence_heatmap, downscale_for_display

st.set_page_config(page_title="Скажи мне, кто твой шлиф", layout="wide", page_icon="🪨")


@st.cache_resource
def get_pipeline(
    tile: int, overlap: int, downscale: float, max_mpx: float
) -> OrePipeline:
    cfg = InferenceConfig(
        tile_size=tile, overlap=overlap, downscale_factor=downscale, max_mpx=max_mpx
    )
    return OrePipeline(cfg)


def zoomable(img: np.ndarray, title: str):
    fig = px.imshow(downscale_for_display(img))
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=30, b=0), dragmode="pan", height=560
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def correction_block(pipeline: OrePipeline, artifacts: dict, stem: str):
    st.markdown(
        "1. Скачайте индексную маску (0 — фон, 1 — обычные срастания, "
        "2 — тонкие, 3 — тальк).\n"
        "2. Исправьте ошибочные участки в любом редакторе "
        "(GIMP/Photoshop/ImageJ, кисть значением класса).\n"
        "3. Загрузите исправленную маску — метрики будут пересчитаны, "
        "пара (снимок, маска) сохранится для дообучения."
    )
    ok_m, mask_png = cv2.imencode(".png", artifacts["mask"])
    st.download_button(
        "⬇Индексная маска (PNG)",
        mask_png.tobytes(),
        file_name=f"{stem}_mask.png",
        mime="image/png",
    )

    fixed = st.file_uploader(
        "Исправленная маска (PNG, значения 0–3)", type=["png"], key="fixed_mask"
    )
    if fixed is None:
        return
    buf = np.frombuffer(fixed.getvalue(), np.uint8)
    fixed_mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if fixed_mask is None or fixed_mask.shape != artifacts["mask"].shape:
        st.error(
            "Маска не читается или не совпадает по размеру "
            f"({artifacts['mask'].shape[1]}×{artifacts['mask'].shape[0]} ожидается)."
        )
        return
    if fixed_mask.max() > 3:
        st.error("Маска содержит значения вне диапазона 0–3.")
        return
    res_fixed = compute_metrics(fixed_mask, calib=pipeline.calib)
    changed = 100.0 * float((fixed_mask != artifacts["mask"]).mean())
    st.info(f"Исправлено {changed:.1f}% пикселей.")
    st.success(f"Пересчитанное заключение: {res_fixed.conclusion}")
    st.dataframe(res_fixed.metrics_df, use_container_width=True, hide_index=True)
    if st.button("Сохранить пару для дообучения"):
        corr = Path("data/corrections")
        (corr / "images").mkdir(parents=True, exist_ok=True)
        (corr / "masks").mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", cv2.cvtColor(artifacts["image"], cv2.COLOR_RGB2BGR))[
            1
        ].tofile(str(corr / "images" / f"{stem}.png"))
        cv2.imencode(".png", fixed_mask)[1].tofile(str(corr / "masks" / f"{stem}.png"))
        st.success(
            "Сохранено в data/corrections — добавьте папку в "
            "data_root при следующем запуске обучения."
        )


def main():
    with st.sidebar:
        st.header("Параметры анализа")
        tile = st.select_slider("Размер тайла", [512, 768, 1024, 1536], value=1024)
        overlap = st.select_slider("Перекрытие", [128, 192, 256, 384], value=256)
        downscale = st.slider("Масштаб изображения", 0.25, 1.0, 1.0, 0.25)
        px_size = st.number_input(
            "Размер пикселя, мкм (0 = неизвестен)", min_value=0.0, value=0.0, step=0.1
        )
        max_mpx = st.select_slider(
            "Лимит, Мпикс (панорамы даунскейлятся)", [25, 50, 100], value=100
        )
        show_conf = st.checkbox("Карта уверенности модели", value=False)
        st.divider()
        st.caption(
            "Зелёный — обычные срастания · Красный — тонкие срастания · Синий — тальк"
        )

    st.title("Автоматическая классификация руд по панорамным OM-снимкам шлифов")

    uploaded = st.file_uploader(
        "Загрузите панорамное изображение шлифа",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp"],
    )
    if uploaded is None:
        st.info("Загрузите TIFF/PNG/JPEG/BMP для запуска анализа.")
        return

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded.name).suffix
    ) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    pipeline = get_pipeline(tile, overlap, downscale, float(max_mpx))

    progress = st.progress(0.0, text="Запуск анализа…")
    artifacts = pipeline.process(
        tmp_path,
        px_size_um=px_size or None,
        progress_cb=lambda p, s: progress.progress(p, text=s),
    )
    progress.empty()

    res = artifacts["result"]
    stem = Path(uploaded.name).stem

    badge = {ORE_TALC: "🟦", ORE_REFRACTORY: "🟥"}.get(res.ore_type, "🟩")
    st.subheader(f"{badge} Заключение")
    st.success(res.conclusion)
    st.caption(
        f"Время обработки: {artifacts['elapsed_s']:.1f} с · "
        f"Разрешение: {artifacts['image'].shape[1]}×{artifacts['image'].shape[0]} пикс."
    )

    col1, col2 = st.columns(2)
    with col1:
        zoomable(artifacts["image"], "Исходное изображение")
    with col2:
        zoomable(artifacts["overlay"], "Маска классификации")

    if show_conf:
        zoomable(
            confidence_heatmap(artifacts["confidence"]),
            "Карта уверенности (красный = спорные участки)",
        )

    st.subheader("Количественные метрики")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Сульфиды, всего", f"{res.sulfide_total_pct:.1f}%")
    mcol2.metric("Обычные срастания", f"{res.regular_of_sulfides_pct:.1f}% сульфидов")
    mcol3.metric("Тонкие срастания", f"{res.fine_of_sulfides_pct:.1f}% сульфидов")
    mcol4.metric("Тальк", f"{res.talc_pct:.1f}%", delta="порог 10%", delta_color="off")

    st.dataframe(res.metrics_df, use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    res.metrics_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇Экспорт метрик (CSV)",
        csv_buf.getvalue(),
        file_name=f"{stem}_metrics.csv",
        mime="text/csv",
    )

    ok, jpg = cv2.imencode(
        ".jpg",
        cv2.cvtColor(artifacts["overlay"], cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    st.download_button(
        "⬇Экспорт маски (JPEG)",
        jpg.tobytes(),
        file_name=f"{stem}_overlay.jpg",
        mime="image/jpeg",
    )

    with st.expander("Экспертная коррекция маски (active learning)"):
        correction_block(pipeline, artifacts, stem)


main()
