"""
Depth Estimation Module
=======================
Uses MiDaS (via PyTorch Hub) for monocular depth estimation
from pothole images.

Classes:
    - DepthEstimator: Main depth estimation class

Usage:
    from src.depth_estimator import DepthEstimator

    estimator = DepthEstimator()
    depth_map = estimator.estimate_depth(image)
    pothole_depth = estimator.get_pothole_depth(depth_map, bbox)
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional
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
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
else:
    TORCH_AVAILABLE = False

import config
from .utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class DepthResult:
    """Container for depth estimation results."""
    depth_map: np.ndarray       # Full image depth map (H, W)
    relative_depth: np.ndarray  # Normalized depth [0, 1]
    pothole_depth_cm: float = 0.0
    depth_category: str = ""
    depth_percentile: float = 0.0


class DepthEstimator:
    """
    Monocular depth estimation using MiDaS.

    Attributes:
        model: MiDaS model instance
        device: Device for inference
        model_type: MiDaS model variant
    """

    def __init__(self, model_type: str = None, device: str = None):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for depth estimation. "
                "Install with: pip install torch"
            )

        self.model_type = model_type or config.MIDAS_MODEL_TYPE
        self.device = device or config.DEFAULT_DEVICE

        if self.model_type == "DPT_Hybrid":
            model_repo = "isl-org/MiDaS"
            model_name = "DPT_Hybrid"
        elif self.model_type == "DPT_Large":
            model_repo = "isl-org/MiDaS"
            model_name = "DPT_Large"
        else:
            model_repo = "isl-org/MiDaS"
            model_name = "DPT_Hybrid"

        self._load_model(model_repo, model_name)

        logger.info(f"DepthEstimator initialized with {self.model_type} on {self.device}")

    def _load_model(self, repo: str, model_name: str):
        """Load MiDaS model from PyTorch Hub or cache."""
        try:
            self.model = torch.hub.load(repo, model_name, pretrained=True)
            self.model.eval()
            self.model.to(self.device)

            # Load transforms
            self.transforms = torch.hub.load(repo, "transforms")[0]

            logger.info(f"MiDaS {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MiDaS: {e}")
            logger.info("Will use synthetic depth estimation as fallback")
            self.model = None

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for an input image.

        Args:
            image: Input image (BGR format)

        Returns:
            Depth map (H, W) with normalized depth values
        """
        if self.model is None:
            return self._synthetic_depth(image)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transforms
        input_batch = self.transforms(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.model(input_batch)

        # Resize to original image size
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = depth.cpu().numpy()
        return depth_map

    def _synthetic_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback: Generate a synthetic depth map based on brightness/shadows.
        This is used when MiDaS is not available.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32) / 255.0

        # Darker areas = deeper (shadows in potholes)
        # Invert: brighter = closer, darker = farther
        depth_map = 1.0 - gray_f

        # Apply Gaussian blur to smooth
        depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)

        # Normalize to [0, 1]
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map.astype(np.float32)

    def get_pothole_depth(self, depth_map: np.ndarray,
                          bbox: Tuple[int, int, int, int],
                          reference_depth: float = None) -> Tuple[float, str]:
        """
        Estimate pothole depth from the depth map within the bounding box.

        Args:
            depth_map: Full image depth map
            bbox: (x1, y1, x2, y2) bounding box
            reference_depth: Depth of surrounding road (meters)

        Returns:
            Tuple of (depth_in_cm, depth_category)
        """
        x1, y1, x2, y2 = bbox
        region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            return 0.0, ""

        # Pothole depth = surrounding road depth - pothole depth (inverted)
        # Since MiDaS gives relative depth, deeper = higher value
        # In road images, potholes appear darker/deeper

        mean_depth = float(np.mean(region))
        min_depth = float(np.min(region))

        # Estimate depth difference from surrounding road
        # Use a ring around the bounding box as reference
        h, w = depth_map.shape
        pad = 20
        x1_ref = max(0, x1 - pad)
        y1_ref = max(0, y1 - pad)
        x2_ref = min(w, x2 + pad)
        y2_ref = min(h, y2 + pad)

        # Extract reference region (excluding pothole area)
        ref_region = depth_map[y1_ref:y2_ref, x1_ref:x2_ref].copy()
        ref_h, ref_w = ref_region.shape

        # Mask out the pothole area from reference
        pothole_h = y2 - y1
        pothole_w = x2 - x1
        py_offset = y1 - y1_ref
        px_offset = x1 - x1_ref

        if px_offset >= 0 and py_offset >= 0 and \
           px_offset + pothole_w <= ref_w and py_offset + pothole_h <= ref_h:
            ref_region[py_offset:py_offset + pothole_h,
                       px_offset:px_offset + pothole_w] = 0

        ref_region = ref_region[ref_region > 0]
        ref_mean = float(np.mean(ref_region)) if len(ref_region) > 0 else mean_depth

        # Depth difference (pothole is deeper = lower relative depth value)
        depth_diff = abs(ref_mean - mean_depth)
        depth_diff = float(min(max(depth_diff, 0.0), 1.0))

        # Map the normalized difference to a plausible physical pothole depth.
        # Relative / synthetic depth maps are dimensionless, so this is a
        # calibrated estimate: full contrast (1.0) -> 25 cm, which still lands
        # in the Deep category but stays within realistic pothole geometry.
        # NOTE: For survey-grade accuracy, calibrate this against measured
        # ground-truth depths or use stereo/LiDAR (see docs).
        max_physical_cm = getattr(config, "DEPTH_MAX_CM_ESTIMATE", 25.0)
        depth_cm = depth_diff * max_physical_cm

        # Geometric plausibility bound: a routine pothole is wider than it is
        # deep. Cap the estimate at ~50% of the pothole's minor real-world
        # dimension (derived from the bbox using the same calibrated baseline
        # as size estimation). Requires config.PIXEL_TO_CM_BASELINE.
        try:
            minor_px = max(1, min(abs(x2 - x1), abs(y2 - y1)))
            map_w = depth_map.shape[1] if depth_map.ndim == 2 else 1
            dens = (getattr(config, "PIXEL_TO_CM_BASELINE", 0.35) *
                    (640.0 / map_w if map_w else 1.0))
            minor_cm = minor_px * dens
            depth_cm = min(depth_cm, round(0.5 * minor_cm, 2))
        except Exception:
            pass
        depth_cm = round(max(depth_cm, 0.0), 2)

        # Classify depth
        depth_category = self._classify_depth(depth_cm)

        return depth_cm, depth_category

    def _classify_depth(self, depth_cm: float) -> str:
        """Classify depth into categories."""
        if depth_cm < config.DEPTH_SHALLOW_MAX_CM:
            return "Shallow"
        elif depth_cm < config.DEPTH_MODERATE_MAX_CM:
            return "Moderate"
        else:
            return "Deep"

    def create_depth_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """Create a color-mapped visualization of the depth map."""
        depth_normalized = depth_map.copy()
        depth_min = depth_normalized.min()
        depth_max = depth_normalized.max()
        if depth_max > depth_min:
            depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        return depth_colored

    def estimate_depth_for_pothole(self, image, bbox):
        """Full pipeline: estimate depth map and extract pothole depth."""
        depth_map = self.estimate_depth(image)
        depth_cm, category = self.get_pothole_depth(depth_map, bbox)
        return DepthResult(
            depth_map=depth_map,
            relative_depth=(depth_map - depth_map.min()) /
                            (depth_map.max() - depth_map.min() + 1e-8),
            pothole_depth_cm=depth_cm,
            depth_category=category,
        )
