from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.detection import detect_and_crop_board
from src.digit_recognition import load_digit_model, predict_sudoku_grid
from src.preprocessing import board_to_mnist_cells
from src.solver import solve_board, validate_sudoku_matrix, validate_sudoku_rules


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

YOLO_MODEL_PATH = MODELS_DIR / "best.pt"
DIGIT_MODEL_PATH = MODELS_DIR / "digit_cnn.keras"
SAMPLE_IMAGE_PATH = DATA_DIR / "img.png"


st.set_page_config(
    page_title="Sudoku DL Solver",
    page_icon="S",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #4b5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: #f8fafc;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_digit_model():
    return load_digit_model(DIGIT_MODEL_PATH)


def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def grid_to_dataframe(grid):
    return pd.DataFrame(
        grid.astype(int),
        columns=[f"C{i}" for i in range(1, 10)],
        index=[f"F{i}" for i in range(1, 10)],
    )


def dataframe_to_grid(df):
    grid = df.to_numpy(dtype=int)
    return np.clip(grid, 0, 9)


def render_sudoku_grid(grid, original_grid=None):
    html = """
    <style>
    .sudoku-board {
        border-collapse: collapse;
        margin-top: 0.5rem;
    }
    .sudoku-board td {
        width: 42px;
        height: 42px;
        text-align: center;
        vertical-align: middle;
        border: 1px solid #94a3b8;
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        background: #ffffff;
    }
    .sudoku-board .given {
        background: #e0f2fe;
        color: #0f172a;
    }
    .sudoku-board .empty {
        color: transparent;
    }
    .sudoku-board tr:nth-child(3n) td {
        border-bottom: 3px solid #0f172a;
    }
    .sudoku-board tr:first-child td {
        border-top: 3px solid #0f172a;
    }
    .sudoku-board td:nth-child(3n) {
        border-right: 3px solid #0f172a;
    }
    .sudoku-board td:first-child {
        border-left: 3px solid #0f172a;
    }
    </style>
    <table class="sudoku-board">
    """

    for row in range(9):
        html += "<tr>"
        for col in range(9):
            value = int(grid[row, col])
            is_given = original_grid is not None and int(original_grid[row, col]) != 0
            css_class = "given" if is_given else ""
            if value == 0:
                css_class = f"{css_class} empty".strip()
                text = "0"
            else:
                text = str(value)
            html += f'<td class="{css_class}">{text}</td>'
        html += "</tr>"

    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


def run_prediction(image_path):
    board_bgr, box, _ = detect_and_crop_board(YOLO_MODEL_PATH, image_path)
    mnist_cells, occupancy_matrix, _, board_resized = board_to_mnist_cells(board_bgr)
    digit_model = get_digit_model()
    predicted_grid, confidence_grid = predict_sudoku_grid(digit_model, mnist_cells)

    return {
        "board_bgr": board_bgr,
        "board_resized": board_resized,
        "mnist_cells": mnist_cells,
        "occupancy_matrix": occupancy_matrix,
        "predicted_grid": predicted_grid,
        "confidence_grid": confidence_grid,
        "box": box,
    }


st.markdown('<div class="main-title">Sudoku DL Solver</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Deteccion del tablero con YOLO, reconocimiento de digitos con CNN y resolucion con backtracking.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Entrada")
    input_source = st.radio(
        "Fuente de imagen",
        ["Subir imagen", "Imagen de ejemplo"],
        horizontal=False,
    )

    uploaded_file = None
    if input_source == "Subir imagen":
        uploaded_file = st.file_uploader(
            "Sube un archivo PNG/JPG del Sudoku",
            type=["png", "jpg", "jpeg"],
        )
    else:
        st.caption("Se usara la imagen de ejemplo incluida en el proyecto.")

    process_clicked = st.button("Procesar y resolver", type="primary")

    st.divider()
    st.caption("Modelos configurados")
    st.write(f"YOLO: `{YOLO_MODEL_PATH.name}`")
    st.write(f"CNN: `{DIGIT_MODEL_PATH.name}`")


if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "original_bgr" not in st.session_state:
    st.session_state.original_bgr = None
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "image_key" not in st.session_state:
    st.session_state.image_key = None

if input_source == "Subir imagen" and uploaded_file is not None:
    selected_image_path = save_uploaded_file(uploaded_file)
    selected_image_key = f"upload:{uploaded_file.name}:{uploaded_file.size}"
elif input_source == "Imagen de ejemplo" and SAMPLE_IMAGE_PATH.exists():
    selected_image_path = SAMPLE_IMAGE_PATH
    selected_image_key = f"sample:{SAMPLE_IMAGE_PATH.stat().st_mtime}"
else:
    selected_image_path = None
    selected_image_key = None

if selected_image_path is None:
    st.info("Sube una imagen PNG/JPG de un Sudoku o usa la imagen de ejemplo para comenzar.")
    st.stop()

if not YOLO_MODEL_PATH.exists():
    st.error(f"No se encontro el modelo YOLO: {YOLO_MODEL_PATH}")
    st.stop()

if not DIGIT_MODEL_PATH.exists():
    st.error(f"No se encontro el modelo CNN: {DIGIT_MODEL_PATH}")
    st.stop()

image_changed = selected_image_key != st.session_state.image_key

if process_clicked or image_changed or st.session_state.prediction_result is None:
    try:
        original_bgr = cv2.imread(str(selected_image_path))
        if original_bgr is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {selected_image_path}")

        with st.spinner("Procesando imagen..."):
            result = run_prediction(selected_image_path)

        st.session_state.prediction_result = result
        st.session_state.original_bgr = original_bgr
        st.session_state.image_path = selected_image_path
        st.session_state.image_key = selected_image_key
    except Exception as exc:
        st.error("No se pudo procesar la imagen.")
        st.exception(exc)
        st.stop()

result = st.session_state.prediction_result
original_bgr = st.session_state.original_bgr

predicted_grid = result["predicted_grid"]
confidence_grid = result["confidence_grid"]
occupancy_matrix = result["occupancy_matrix"]

col_img, col_board = st.columns([1, 1])

with col_img:
    st.subheader("Imagen y tablero detectado")
    st.image(bgr_to_rgb(original_bgr), caption="Imagen original", use_container_width=True)
    st.image(bgr_to_rgb(result["board_bgr"]), caption="Tablero recortado", use_container_width=True)

with col_board:
    st.subheader("Matriz detectada")
    render_sudoku_grid(predicted_grid, original_grid=predicted_grid)

    detected_cells = int(occupancy_matrix.sum())
    nonzero_confidences = confidence_grid[confidence_grid > 0]
    mean_confidence = float(nonzero_confidences.mean()) if len(nonzero_confidences) else 0.0

    c1, c2 = st.columns(2)
    c1.metric("Celdas detectadas", detected_cells)
    c2.metric("Confianza media", f"{mean_confidence:.2%}")


auto_solution, auto_solved, auto_message = solve_board(predicted_grid)

if auto_solved:
    st.divider()
    st.subheader("Solucion automatica")
    st.success("La matriz detectada se pudo resolver automaticamente.")
    render_sudoku_grid(auto_solution, original_grid=predicted_grid)

    solution_csv = pd.DataFrame(auto_solution).to_csv(index=False, header=False)
    st.download_button(
        "Descargar solucion CSV",
        data=solution_csv,
        file_name="sudoku_solution.csv",
        mime="text/csv",
        key="auto_solution_download",
    )
else:
    st.divider()
    st.warning(
        "La matriz detectada necesita revision antes de resolver: "
        f"{auto_message}"
    )


st.divider()
st.subheader("Revision manual antes de resolver")
st.markdown(
    '<div class="small-note">Si algun digito fue reconocido de forma incorrecta, corrigelo aqui. Usa 0 para celdas vacias.</div>',
    unsafe_allow_html=True,
)

edited_df = st.data_editor(
    grid_to_dataframe(predicted_grid),
    use_container_width=True,
    num_rows="fixed",
    column_config={col: st.column_config.NumberColumn(col, min_value=0, max_value=9, step=1) for col in [f"C{i}" for i in range(1, 10)]},
    key="sudoku_editor",
)

corrected_grid = dataframe_to_grid(edited_df)

format_valid, format_message = validate_sudoku_matrix(corrected_grid)
rules_valid, rules_message = validate_sudoku_rules(corrected_grid)

if not format_valid:
    st.error(format_message)
elif not rules_valid:
    st.warning(rules_message)
else:
    st.success("La matriz esta lista para resolverse.")


if st.button("Resolver Sudoku", type="primary", disabled=not (format_valid and rules_valid)):
    solution, solved, message = solve_board(corrected_grid)

    if solved:
        st.subheader("Sudoku resuelto")
        render_sudoku_grid(solution, original_grid=corrected_grid)
        st.success(message)

        solution_csv = pd.DataFrame(solution).to_csv(index=False, header=False)
        st.download_button(
            "Descargar solucion CSV",
            data=solution_csv,
            file_name="sudoku_solution.csv",
            mime="text/csv",
        )
    else:
        st.error(message)
