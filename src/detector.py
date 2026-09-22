"""
Pothole Detection Module
========================
Uses YOLOv8 (Ultralytics) for real-time pothole detection from road images.

Usage:
    from src.detector import PotholeDetector

    detector = PotholeDetector(model_path="data/models/best.pt")
    results = detector.detect(image)
    detections = detector.extract_detections(results)
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


def _torch_usable():
    """Return True only if torch is installed AND its native libs load.

    The probe runs in a subprocess because a broken torch install can raise
    beyond ImportError/OSError (segfault / access violation), which would
    otherwise crash this interpreter during a plain `import torch`.
    """
    try:
        from importlib import metadata as _metadata
        _metadata.version("torch")
    except Exception:
        return False
    try:
        import subprocess as _sp
        import sys as _sys
        r = _sp.run(
            [_sys.executable, "-c",
             "import torch; "
             "assert hasattr(torch, 'cuda'); "
             "torch.cuda.is_available()"],
            capture_output=True, timeout=120,
        )
        return r.returncode == 0
    except Exception:
        return False


try:
    _TORCH_OK = _torch_usable()
except Exception:
    _TORCH_OK = False

if _TORCH_OK:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
else:
    TORCH_AVAILABLE = False

import config
from . import utils
from . import preprocessing
from .utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class DetectionResult:
    """Data class representing a single pothole detection."""
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    width_px: int
    height_px: int
    center: Tuple[int, int]
    size_cm2: float = 0.0
    size_category: str = ""
    width_cm: float = 0.0
    height_cm: float = 0.0
    depth_cm: float = 0.0
    depth_category: str = ""
    severity: str = ""
    priority: int = 4
    response_time_hours: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "center": list(self.center),
            "size_cm2": round(self.size_cm2, 2),
            "size_category": self.size_category,
            "width_cm": round(self.width_cm, 2),
            "height_cm": round(self.height_cm, 2),
            "depth_cm": round(self.depth_cm, 2),
            "depth_category": self.depth_category,
            "severity": self.severity,
            "priority": self.priority,
            "response_time_hours": self.response_time_hours,
        }


class PotholeDetector:
    """Pothole detection using YOLOv8."""

    def __init__(self, model_path=None, device=None,
                 conf_thres=None, iou_thres=None):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and Ultralytics YOLO are required. "
                "Install with: pip install torch ultralytics"
            )
        if model_path is None:
            model_path = config.YOLO_MODEL_PATH
        self.conf_thres = conf_thres or config.CONFIDENCE_THRESHOLD
        self.iou_thres = iou_thres or config.IOU_THRESHOLD
        self.device = device or config.DEFAULT_DEVICE
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        logger.info(f"PotholeDetector initialized on device: {self.device}")

    def _load_model(self, model_path):
        if os.path.exists(model_path):
            logger.info(f"Loading custom model from: {model_path}")
            return YOLO(model_path)
        for name in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]:
            try:
                logger.info(f"Trying model: {name}")
                return YOLO(name)
            except Exception:
                continue
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please download a YOLOv8 model or provide a valid model path."
        )

    def detect(self, image, preprocess_input=False):
        if preprocess_input:
            img = preprocessing.enhance_contrast(image)
        else:
            img = image
        results = self.model(
            img, conf=self.conf_thres, iou=self.iou_thres,
            verbose=False, device=self.device,
        )
        return results[0] if len(results) > 0 else results

    def detect_and_extract(self, image):
        results = self.detect(image)
        return self.extract_detections(results)

    def detect_from_path(self, image_path):
        image = utils.load_image(image_path)
        detections = self.detect_and_extract(image)
        return image, detections

    def extract_detections(self, results):
        """Extract structured detection results from YOLO output."""
        detections = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections
        if hasattr(results, 'orig_shape'):
            image_shape = results.orig_shape
        else:
            image_shape = (720, 1280)
        img_h, img_w = image_shape[:2]
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu().numpy()[0])
            cls_id = int(box.cls.cpu().numpy()[0])
            x1, y1, x2, y2 = xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            width_px = x2 - x1
            height_px = y2 - y1
            class_name = config.CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            detection = DetectionResult(
                bbox=[x1, y1, x2, y2], confidence=conf, class_id=cls_id,
                class_name=class_name, width_px=width_px,
                height_px=height_px,
                center=((x1 + x2) // 2, (y1 + y2) // 2),
            )
            detections.append(detection)
        logger.info(f"Extracted {len(detections)} detections")
        return detections

    def visualize_detections(self, image, detections, show_labels=True):
        """Draw detection results on the image."""
        img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                label = f"Pothole: {det.confidence:.2f}"
                (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - h - 4),
                              (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1)
        return img

    def get_model_info(self):
        return {
            "model_path": config.YOLO_MODEL_PATH,
            "device": self.device,
            "conf_threshold": self.conf_thres,
            "iou_threshold": self.iou_thres,
            "class_names": config.CLASS_NAMES,
        }
