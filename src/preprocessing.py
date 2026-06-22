import cv2
import numpy as np


BOARD_SIZE = 450
EMPTY_THRESHOLD = 65


def resize_board(board_bgr, board_size=BOARD_SIZE):
    """Resize the detected board to a square image."""
    return cv2.resize(board_bgr, (board_size, board_size))


def split_board_into_cells(board_image, board_size=BOARD_SIZE):
    """Split a square Sudoku board into 81 cells."""
    cell_size = board_size // 9
    cells = []

    for row in range(9):
        row_cells = []
        for col in range(9):
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            row_cells.append(board_image[y1:y2, x1:x2])
        cells.append(row_cells)

    return cells


def crop_cell_margin(cell, margin=10):
    """Remove inner borders from a cell."""
    h, w = cell.shape[:2]
    return cell[margin:h - margin, margin:w - margin]


def preprocess_cell_for_digit(cell, margin=10):
    """Convert a cell to a binary image with a white digit over black background."""
    cell_crop = crop_cell_margin(cell, margin=margin)
    gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )


def is_empty_cell(processed_cell, threshold=EMPTY_THRESHOLD):
    """Return True when a preprocessed cell has too few white pixels."""
    white_pixels = cv2.countNonZero(processed_cell)
    return white_pixels < threshold


def extract_digit_region(processed_cell):
    """Extract the smallest rectangle containing the digit pixels."""
    coords = cv2.findNonZero(processed_cell)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    return processed_cell[y:y + h, x:x + w]


def keep_digit_component(processed_cell, min_area=15):
    """Keep the largest digit-like component and remove thin grid artifacts."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        processed_cell,
        connectivity=8,
    )

    cleaned = np.zeros_like(processed_cell)
    components = []

    for label in range(1, num_labels):
        _, _, w, h, area = stats[label]

        if area < min_area:
            continue

        is_thin_vertical_line = w <= 3 and h > 10
        is_thin_horizontal_line = h <= 3 and w > 10

        if is_thin_vertical_line or is_thin_horizontal_line:
            continue

        components.append((label, area))

    if len(components) == 0:
        return cleaned

    best_label = max(components, key=lambda item: item[1])[0]
    cleaned[labels == best_label] = 255

    return cleaned


def prepare_digit_for_mnist(processed_cell, output_size=28, digit_size=20):
    """Center a cleaned digit in a 28x28 MNIST-like image."""
    cleaned_cell = keep_digit_component(processed_cell)
    digit = extract_digit_region(cleaned_cell)

    if digit is None:
        return np.zeros((output_size, output_size), dtype=np.float32)

    h, w = digit.shape

    if h == 0 or w == 0:
        return np.zeros((output_size, output_size), dtype=np.float32)

    scale = digit_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit_resized

    return canvas.astype("float32") / 255.0


def prepare_cells_for_prediction(cells, empty_threshold=EMPTY_THRESHOLD, margin=10):
    """Prepare all occupied cells for CNN prediction and return occupancy matrix."""
    mnist_cells = [[None for _ in range(9)] for _ in range(9)]
    occupancy_matrix = np.zeros((9, 9), dtype=int)

    for row in range(9):
        for col in range(9):
            processed = preprocess_cell_for_digit(cells[row][col], margin=margin)

            if is_empty_cell(processed, threshold=empty_threshold):
                mnist_cells[row][col] = None
                occupancy_matrix[row, col] = 0
            else:
                mnist_cells[row][col] = prepare_digit_for_mnist(processed)
                occupancy_matrix[row, col] = 1

    return mnist_cells, occupancy_matrix


def board_to_mnist_cells(board_bgr, board_size=BOARD_SIZE, empty_threshold=EMPTY_THRESHOLD):
    """Resize, split, and preprocess a detected Sudoku board."""
    board_resized = resize_board(board_bgr, board_size=board_size)
    cells = split_board_into_cells(board_resized, board_size=board_size)
    mnist_cells, occupancy_matrix = prepare_cells_for_prediction(
        cells,
        empty_threshold=empty_threshold,
    )

    return mnist_cells, occupancy_matrix, cells, board_resized
