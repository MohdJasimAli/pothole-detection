"""
Utility functions for the Road Pothole Detection System.

Provides helper functions for:
- File I/O and path management
- Image conversion and visualization
- Coordinate transformations
- Logging setup
- Data serialization
"""

import os
import cv2
import json
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

import config


# ============================================================
# Logging Setup
# ============================================================
def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# Image Utilities
# ============================================================
def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format) or None if loading fails
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load an image from raw bytes.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Image as numpy array (BGR format) or None
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to file.

    Args:
        image: Image as numpy array
        output_path: Output file path

    Returns:
        True if saved successfully, False otherwise
    """
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    return cv2.imwrite(output_path, image)


def image_to_base64(image: np.ndarray) -> str:
    """Convert a numpy image array to base64 string."""
    import base64
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert a base64 string to numpy image array."""
    import base64
    image_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def resize_image(image: np.ndarray, max_dim: int = 640) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: Input image
        max_dim: Maximum dimension (height or width)

    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ============================================================
# Coordinate & Bounding Box Utilities
# ============================================================
def extract_bounding_boxes(results, image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Extract bounding boxes from YOLO detection results.

    Args:
        results: YOLO detection results
        image_shape: (height, width) of the source image

    Returns:
        List of bounding box dictionaries
    """
    boxes = []
    img_h, img_w = image_shape[:2]

    if results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu().numpy()[0])
            cls_id = int(box.cls.cpu().numpy()[0])

            x1, y1, x2, y2 = xyxy
            width_px = int(x2 - x1)
            height_px = int(y2 - y1)

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w - 1, int(x2))
            y2 = min(img_h - 1, int(y2))

            class_name = config.CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": class_name,
                "width_px": width_px,
                "height_px": height_px,
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            })

    return boxes


def pixels_to_cm(pixel_width: float, pixel_height: float,
                 reference_width_px: float = None,
                 reference_width_cm: float = None,
                 distance_m: float = 2.0) -> Tuple[float, float]:
    """Convert pixel measurements to real-world centimeters."""
    if reference_width_px and reference_width_cm and reference_width_px > 0:
        pixel_to_cm = reference_width_cm / reference_width_px
        width_cm = pixel_width * pixel_to_cm
        height_cm = pixel_height * pixel_to_cm
    else:
        f = config.FOCAL_LENGTH_PIXELS
        pixel_to_cm = (distance_m * 100) / f
        width_cm = pixel_width * pixel_to_cm
        height_cm = pixel_height * pixel_to_cm
    return width_cm, height_cm


def draw_bounding_boxes(image: np.ndarray, detections: List[Dict[str, Any]],
                        thickness: int = 2) -> np.ndarray:
    """Draw bounding boxes with labels on image."""
    img = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        label = det.get("class_name", "Object")
        severity = det.get("severity", "")
        size_cat = det.get("size_category", "")

        if severity == "Critical":
            color = (0, 0, 255)
        elif severity == "High":
            color = (0, 165, 255)
        elif severity == "Medium":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{label}: {conf:.2f}"
        if size_cat:
            label_text += f" [{size_cat}]"

        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if severity:
            sev_text = f"Severity: {severity}"
            cv2.rectangle(img, (x1, y2), (x1 + 120, y2 + 20), color, -1)
            cv2.putText(img, sev_text, (x1 + 5, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img


# ============================================================
# Report Generation
# ============================================================
def generate_report_json(detections: List[Dict[str, Any]],
                         image_path: str = "",
                         gps_coords: Tuple[float, float] = None) -> Dict[str, Any]:
    """Generate a JSON-serializable report from detections."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "image_path": image_path,
        "gps_coordinates": {"lat": gps_coords[0], "lng": gps_coords[1]} if gps_coords else None,
        "total_potholes_detected": len(detections),
        "potholes": detections,
        "summary": {
            "critical_count": sum(1 for d in detections if d.get("severity") == "Critical"),
            "high_count": sum(1 for d in detections if d.get("severity") == "High"),
            "medium_count": sum(1 for d in detections if d.get("severity") == "Medium"),
            "low_count": sum(1 for d in detections if d.get("severity") == "Low"),
        }
    }
    return report


def format_severity_report(report: Dict[str, Any]) -> str:
    """Format a report dictionary into a human-readable string."""
    lines = [
        "=" * 60,
        "POTHOLE DETECTION & SEVERITY REPORT",
        "=" * 60,
        f"Timestamp: {report['timestamp']}",
        f"Image: {report.get('image_path', 'N/A')}",
        f"GPS: {report.get('gps_coordinates', 'N/A')}",
        f"Total Potholes Detected: {report['total_potholes_detected']}",
        "",
        "SEVERITY SUMMARY:",
        f"  [CRITICAL]: {report['summary']['critical_count']}",
        f"  [HIGH]:      {report['summary']['high_count']}",
        f"  [MEDIUM]:    {report['summary']['medium_count']}",
        f"  [LOW]:       {report['summary']['low_count']}",
        "",
        "DETAILED FINDINGS:",
        "-" * 60,
    ]

    for i, p in enumerate(report["potholes"], 1):
        lines.append(f"\nPothole #{i}:")
        lines.append(f"  Location (px): bbox={p['bbox']}")
        lines.append(f"  Confidence: {p['confidence']:.4f}")
        lines.append(f"  Size: {p.get('size_cm2', 'N/A')} cm2 ({p.get('size_category', 'N/A')})")
        lines.append(f"  Dimensions: {p.get('width_cm', 'N/A')}cm x {p.get('height_cm', 'N/A')}cm")
        lines.append(f"  Depth Est.: {p.get('depth_cm', 'N/A')} cm ({p.get('depth_category', 'N/A')})")
        lines.append(f"  Severity: {p.get('severity', 'Unknown')}")
        lines.append(f"  Priority: {p.get('priority', 'N/A')}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as JSON file."""
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file as dictionary."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        return json.load(f)


# ============================================================
# Performance Metrics
# ============================================================
def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])

    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# Validation Utilities
# ============================================================
def is_valid_image(filepath: str) -> bool:
    """Check if file is a valid image."""
    try:
        img = load_image(filepath)
        return img is not None
    except Exception:
        return False


def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS