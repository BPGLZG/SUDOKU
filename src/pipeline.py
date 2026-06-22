from pathlib import Path

import numpy as np

from .detection import detect_and_crop_board
from .digit_recognition import load_digit_model, predict_sudoku_grid
from .preprocessing import board_to_mnist_cells
from .solver import solve_board


def process_sudoku_image(yolo_model_path, digit_model_path, image_path):
    """Run the full Sudoku image pipeline up to prediction and solving."""
    board_bgr, box, _ = detect_and_crop_board(yolo_model_path, image_path)
    mnist_cells, occupancy_matrix, cells, board_resized = board_to_mnist_cells(board_bgr)

    digit_model = load_digit_model(digit_model_path)
    predicted_grid, confidence_grid = predict_sudoku_grid(digit_model, mnist_cells)

    solution, solved, message = solve_board(predicted_grid)

    return {
        "board_bgr": board_bgr,
        "board_resized": board_resized,
        "cells": cells,
        "mnist_cells": mnist_cells,
        "occupancy_matrix": occupancy_matrix,
        "predicted_grid": predicted_grid,
        "confidence_grid": confidence_grid,
        "solution": solution,
        "solved": solved,
        "message": message,
        "box": box,
    }


def save_grid(grid, output_path):
    """Save a Sudoku grid as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, grid, fmt="%d", delimiter=",")
