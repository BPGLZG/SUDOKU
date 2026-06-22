from pathlib import Path

import numpy as np
from tensorflow import keras


def load_digit_model(model_path):
    """Load the trained digit recognition CNN."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Digit model not found: {model_path}")

    return keras.models.load_model(model_path)


def predict_digit(model, mnist_cell):
    """Predict one MNIST-like digit image."""
    image = mnist_cell.reshape(1, 28, 28, 1)
    probabilities = model.predict(image, verbose=0)[0]

    return int(np.argmax(probabilities)), float(np.max(probabilities))


def predict_sudoku_grid(model, mnist_cells):
    """Predict all occupied Sudoku cells and return grid plus confidences."""
    predicted_grid = np.zeros((9, 9), dtype=int)
    confidence_grid = np.zeros((9, 9), dtype=float)

    for row in range(9):
        for col in range(9):
            if mnist_cells[row][col] is None:
                continue

            digit, confidence = predict_digit(model, mnist_cells[row][col])
            predicted_grid[row, col] = digit
            confidence_grid[row, col] = confidence

    return predicted_grid, confidence_grid
