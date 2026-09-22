"""
Classical Computer-Vision Pothole Detector (no deep-learning dependency)
========================================================================
Fallback detector using traditional image processing:

    1. Grayscale + CLAHE contrast enhancement
    2. GLOBAL (Otsu) + LOCAL (adaptive) thresholding combined - Otsu catches
       the full body of large dark blobs, adaptive catches edges/subtle
       regions; neither alone is sufficient
    3. Brightness verification against a LARGE-window local mean (window
       scales with image size) so big uniform potholes are not rejected
    4. Morphological open/close/dilate clean-up
    5. Contour extraction with shape/area filtering
    6. Nearby-box merging so one pothole fragmented into several contours
       reports as a single bounding box

Usage:
    from src.classical_detector import ClassicalPotholeDetector
    detector = ClassicalPotholeDetector()
    detections = detector.detect_and_extract(image)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

import config
from .utils import setup_logger

logger = setup_logger(__name__)


class ClassicalPotholeDetector:
    """
    Pothole detector using classical image processing.

    Detects potholes as dark, roughly elliptical regions on the road surface.
    Suitable as a CPU-only, dependency-light fallback to YOLOv8.

    Attributes:
        min_area_px: Minimum contour area (pixels) to count as a pothole
        max_area_ratio: Maximum contour area as a fraction of the image
        darkness_threshold: How much darker than the local mean a region must be
        merge_gap_px: Max pixel gap between fragments merged into one pothole
    """

    def __init__(self,
                 min_area_px: int = 300,
                 max_area_ratio: float = 0.30,
                 darkness_threshold: float = 0.80,
                 block_size: int = 51,
                 constant_c: int = 10,
                 merge_gap_px: int = 40):
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio
        self.darkness_threshold = darkness_threshold
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.constant_c = constant_c
        self.merge_gap_px = merge_gap_px
        logger.info(
            f"ClassicalPotholeDetector initialized "
            f"(min_area={min_area_px}px, block={self.block_size}, "
            f"C={constant_c}, merge_gap={merge_gap_px}px)"
        )

    def get_model_info(self) -> dict:
        """Return detector information (mirrors PotholeDetector API)."""
        return {
            "model_path": "classical-cv (no weights)",
            "device": "cpu",
            "conf_threshold": self.darkness_threshold,
            "iou_threshold": None,
            "class_names": config.CLASS_NAMES,
            "method": "otsu+adaptive threshold + morphology + contours",
        }

    def detect(self, image: np.ndarray, preprocess_input: bool = False):
        """
        Run classical detection.

        Args:
            image: BGR image
            preprocess_input: Accepted for API compatibility (unused)

        Returns:
            List of (contour_or_None, bbox, score, area) tuples
        """
        return self._find_pothole_contours(image)

    def _find_pothole_contours(self, image: np.ndarray):
        """Core classical detection routine."""
        img_h, img_w = image.shape[:2]
        img_area = float(img_h * img_w)

        # 1. Grayscale + contrast enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                               tileGridSize=config.CLAHE_TILE_GRID_SIZE)
        gray = clahe.apply(gray)

        # 2. Mild blur to suppress texture noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # 3a. GLOBAL threshold (Otsu): captures the FULL body of large
        #     uniform dark blobs, which adaptive thresholding alone misses
        #     (inside a blob bigger than the window, the local neighborhood
        #     is equally dark so nothing looks "darker than local mean").
        _, otsu_mask = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3b. LOCAL adaptive threshold: catches edges and subtle regions
        adaptive_mask = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.constant_c,
        )

        combined = cv2.bitwise_or(otsu_mask, adaptive_mask)

        # 4. Morphological clean-up + slight dilation to bridge gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel,
                                    iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel,
                                    iterations=3)

        # 5. Brightness verification against a LARGE-window local mean.
        #    The window must be much bigger than the biggest expected
        #    pothole (~1/3 of image dims) so the interior of a large blob
        #    is still "darker than its surroundings".
        win = max(151, (max(img_h, img_w) // 3) | 1)  # force odd
        local_mean = cv2.blur(blurred.astype(np.float32), (win, win))
        brightness_ok = (blurred.astype(np.float32) <
                         local_mean * self.darkness_threshold)
        mask = cv2.bitwise_and(combined,
                               (brightness_ok * 255).astype(np.uint8))

        # 6. Extract contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px or area > img_area * self.max_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0 or w == 0:
                continue

            # Shape filters: potholes are roughly elliptical, not thin lines
            aspect = w / float(h)
            if aspect < 0.25 or aspect > 4.0:
                continue

            # Circularity: 4*pi*area / perimeter^2
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.15:
                continue

            # Fill ratio: how much of the bbox the blob occupies
            fill_ratio = area / float(w * h)
            if fill_ratio < 0.30:
                continue

            # Confidence: combination of circularity, fill and darkness
            region = gray[y:y + h, x:x + w]
            local_ref = local_mean[y:y + h, x:x + w]
            darkness = 1.0 - (float(np.mean(region)) /
                              max(float(np.mean(local_ref)), 1.0))
            darkness = float(np.clip(darkness * 3.0, 0.0, 1.0))

            score = float(np.clip(
                0.4 * min(circularity / 0.8, 1.0) +
                0.3 * min(fill_ratio / 0.7, 1.0) +
                0.3 * darkness,
                0.05, 0.99,
            ))

            results.append((cnt, [x, y, x + w, y + h], score, area))

        # 7. Merge nearby fragments into single pothole boxes
        results = self._merge_results(results, self.merge_gap_px)
        return results

    @staticmethod
    def _merge_results(results, gap_px: int = 40):
        """
        Merge detection fragments whose bounding boxes are within `gap_px`
        of each other into a single detection (union box, max score,
        summed area). A single pothole often fragments into several
        contours after thresholding.
        """
        if not results:
            return results
        boxes = [list(r[1]) for r in results]
        scores = [r[2] for r in results]
        areas = [r[3] for r in results]
        used = [False] * len(boxes)
        merged = []
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, x2, y2 = boxes[i]
            best = scores[i]
            tot = areas[i]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2 = boxes[j]
                    if not (bx1 - gap_px > x2 or bx2 + gap_px < x1 or
                            by1 - gap_px > y2 or by2 + gap_px < y1):
                        x1 = min(x1, bx1)
                        y1 = min(y1, by1)
                        x2 = max(x2, bx2)
                        y2 = max(y2, by2)
                        best = max(best, scores[j])
                        tot += areas[j]
                        used[j] = True
                        changed = True
            merged.append((None, [x1, y1, x2, y2], best, tot))
        merged.sort(key=lambda r: r[3], reverse=True)
        return merged

    def extract_detections(self, classical_results):
        """Convert classical results into DetectionResult objects."""
        from .detector import DetectionResult

        detections: List = []
        for _cnt, bbox, score, _area in classical_results:
            x1, y1, x2, y2 = bbox
            detections.append(DetectionResult(
                bbox=[x1, y1, x2, y2],
                confidence=round(score, 4),
                class_id=0,
                class_name=config.CLASS_NAMES.get(0, "Pothole"),
                width_px=x2 - x1,
                height_px=y2 - y1,
                center=((x1 + x2) // 2, (y1 + y2) // 2),
            ))
        logger.info(f"Classical detector found {len(detections)} candidate(s)")
        return detections

    def detect_and_extract(self, image: np.ndarray):
        """Convenience: detect + convert in one step."""
        return self.extract_detections(self.detect(image))

    def detect_from_path(self, image_path: str):
        """Load image from disk, detect, and return (image, detections)."""
        from .utils import load_image
        image = load_image(image_path)
        return image, self.detect_and_extract(image)

    def visualize_detections(self, image: np.ndarray, detections,
                             show_labels: bool = True) -> np.ndarray:
        """Draw detections (mirrors PotholeDetector.visualize_detections)."""
        img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if show_labels:
                label = f"Pothole: {det.confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label,
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img