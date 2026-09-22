"""
Size Estimation Module
======================
Estimates pothole size (length, width, area) in real-world units
using pixel measurements and camera calibration parameters.

Classes:
    - SizeEstimator: Estimates real-world pothole dimensions

Usage:
    from src.size_estimator import SizeEstimator

    estimator = SizeEstimator()
    width_cm, height_cm, area_cm2 = estimator.estimate_size(bbox, image_width)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

import config
from .utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class SizeResult:
    """Container for size estimation results."""
    width_px: int = 0
    height_px: int = 0
    width_cm: float = 0.0
    height_cm: float = 0.0
    area_cm2: float = 0.0
    size_category: str = ""
    pixel_density_cm_per_px: float = 0.0


class SizeEstimator:
    """
    Real-world size estimation for potholes.

    Uses camera parameters and reference measurements to convert
    pixel coordinates to physical dimensions (cm).
    """

    def __init__(self,
                 focal_length: float = None,
                 camera_height_cm: float = None,
                 reference_width_cm: float = None):
        """
        Initialize size estimator.

        Args:
            focal_length: Camera focal length in pixels
            camera_height_cm: Camera height from road surface (cm)
            reference_width_cm: Known reference object width (cm)
        """
        self.focal_length = focal_length or config.FOCAL_LENGTH_PIXELS
        self.camera_height_cm = camera_height_cm or config.CAMERA_HEIGHT_CM
        self.reference_width_cm = reference_width_cm or config.REFERENCE_OBJECT_SIZE_CM

    def estimate_size(self, bbox: Tuple[int, int, int, int],
                      image_width: int,
                      reference_bbox: Tuple[int, int, int, int] = None) -> SizeResult:
        """
        Estimate real-world pothole size from bounding box.

        Args:
            bbox: (x1, y1, x2, y2) bounding box in pixels
            image_width: Width of the source image in pixels
            reference_bbox: Optional reference object bbox for calibration

        Returns:
            SizeResult with all size measurements
        """
        x1, y1, x2, y2 = bbox
        width_px = x2 - x1
        height_px = y2 - y1
        area_px = width_px * height_px

        # Calculate pixel-to-cm conversion factor
        pixel_density = self._calculate_pixel_density(
            image_width, reference_bbox
        )

        width_cm = width_px * pixel_density
        height_cm = height_px * pixel_density
        area_cm2 = width_cm * height_cm

        size_category = self._classify_size(area_cm2)

        logger.info(
            f"Size: {width_cm:.1f}x{height_cm:.1f}cm, "
            f"Area: {area_cm2:.1f}cm2, Category: {size_category}"
        )

        return SizeResult(
            width_px=width_px,
            height_px=height_px,
            width_cm=round(width_cm, 2),
            height_cm=round(height_cm, 2),
            area_cm2=round(area_cm2, 2),
            size_category=size_category,
            pixel_density_cm_per_px=round(pixel_density, 6),
        )

    def _calculate_pixel_density(self, image_width: int,
                                 reference_bbox: Tuple[int, int, int, int] = None
                                 ) -> float:
        """
        Calculate pixel-to-cm density.

        Uses reference object if provided, otherwise estimates
        based on camera parameters and geometry.
        """
        if reference_bbox is not None:
            ref_x1, ref_y1, ref_x2, ref_y2 = reference_bbox
            ref_width_px = abs(ref_x2 - ref_x1)
            if ref_width_px > 0:
                return self.reference_width_cm / ref_width_px

        # Fallback: calibrated baseline estimate.
        # IMPORTANT: For accurate measurements, pass a reference_bbox with a
        # known object width, or calibrate config.PIXEL_TO_CM_BASELINE for
        # your specific camera and mounting height.
        baseline_640 = config.PIXEL_TO_CM_BASELINE
        # Same scene at higher resolution => each pixel covers LESS ground.
        # scale_factor = W / 640, so cm-per-pixel = baseline / scale_factor.
        scale_factor = image_width / 640.0 if image_width > 0 else 1.0
        pixel_to_cm = baseline_640 / scale_factor

        return pixel_to_cm

    def _classify_size(self, area_cm2: float) -> str:
        """Classify pothole size category based on area."""
        if area_cm2 < config.SIZE_SMALL_MAX_CM2:
            return "Small"
        elif area_cm2 < config.SIZE_MEDIUM_MAX_CM2:
            return "Medium"
        else:
            return "Large"

    def estimate_depth_from_size_ratio(self, bbox: Tuple[int, int, int, int],
                                       depth_value: float) -> float:
        """
        Rough depth estimate based on depth map value and size ratio.

        Args:
            bbox: Bounding box
            depth_value: Depth map value at pothole center

        Returns:
            Estimated depth in cm
        """
        # This is a supplementary method for cross-validation
        x1, y1, x2, y2 = bbox
        width_px = x2 - x1
        height_px = y2 - y1

        # Larger potholes tend to be deeper (empirical observation)
        area_factor = min(width_px * height_px / 10000.0, 5.0)
        depth_estimate = depth_value * 5.0 * area_factor  # Scale to cm

        return round(depth_estimate, 2)

    def batch_estimate(self, bboxes: list, image_width: int) -> list:
        """Estimate sizes for a batch of bounding boxes."""
        results = []
        for bbox in bboxes:
            result = self.estimate_size(bbox, image_width)
            results.append(result)
        return results
