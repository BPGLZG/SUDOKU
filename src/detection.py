from pathlib import Path

import cv2
from ultralytics import YOLO


def load_yolo_model(model_path):
    """Load a YOLO model from disk."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    return YOLO(str(model_path))


def detect_board(model, image_path):
    """Run YOLO detection and return the first prediction result."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(str(image_path))
    return results[0]


def get_best_box(result):
    """Select the detected box with the highest confidence."""
    boxes = result.boxes

    if len(boxes) == 0:
        raise ValueError("YOLO did not detect a Sudoku board.")

    best_box = max(boxes, key=lambda box: float(box.conf[0]))
    x1, y1, x2, y2 = best_box.xyxy[0].tolist()

    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "confidence": float(best_box.conf[0]),
        "class_id": int(best_box.cls[0]),
    }


def load_image(image_path):
    """Load an image with OpenCV in BGR format."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def crop_board_from_box(image_bgr, box, padding=7):
    """Crop the Sudoku board using YOLO box coordinates."""
    h, w = image_bgr.shape[:2]

    x1 = max(box["x1"] - padding, 0)
    y1 = max(box["y1"] - padding, 0)
    x2 = min(box["x2"] + padding, w)
    y2 = min(box["y2"] + padding, h)

    return image_bgr[y1:y2, x1:x2]


def detect_and_crop_board(model_path, image_path, padding=7):
    """Complete YOLO step: load model, detect board, and crop it."""
    model = load_yolo_model(model_path)
    image_bgr = load_image(image_path)
    result = detect_board(model, image_path)
    box = get_best_box(result)
    board_bgr = crop_board_from_box(image_bgr, box, padding=padding)

    return board_bgr, box, result
