"""
Configuration module for Road Pothole Detection and Severity Analysis System.
Contains all configuration constants, paths, and settings.
"""

import os

# ============================================================
# Device Detection
# ============================================================

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

    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_AVAILABLE = True
else:
    TORCH_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "frontend", "uploads")

# Model paths
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
MIDAS_MODEL_PATH = os.path.join(MODELS_DIR, "midas")

# Detection backend mode. When True (default), the pipeline falls back to the
# classical image-processing detector (src/classical_detector.py) on machines
# where PyTorch/Ultralytics cannot be installed - for example sandboxed VMs
# where the torch native DLLs cannot initialize. When False, missing torch
# raises ImportError instead.
ALLOW_CLASSICAL_FALLBACK = True

# ============================================================
# Detection Settings
# ============================================================
# YOLO detection thresholds
CONFIDENCE_THRESHOLD = 0.4      # Minimum confidence for detection
IOU_THRESHOLD = 0.45            # NMS threshold
IMAGE_SIZE = 640                # Input image size for YOLO

# Classes to detect (0 = pothole in our trained model)
CLASS_NAMES = {0: "Pothole"}

# ============================================================
# Image Processing Settings
# ============================================================
# Camera calibration parameters (for size estimation)
# These should be calibrated based on your camera setup
CAMERA_HEIGHT_CM = 80           # Height of camera from ground (cm)
FOCAL_LENGTH_PIXELS = 800       # Focal length in pixels
SENSOR_WIDTH_MM = 36            # Sensor width in mm
REFERENCE_OBJECT_SIZE_CM = 30   # Known reference (e.g., road marking width in cm)

# Baseline pixel-to-cm scale used when no reference object is supplied.
# Assumes a 640 px wide image of a pothole captured from ~1-1.5 m away,
# covering roughly 220 cm of ground (~0.35 cm per pixel).
#
# WARNING: This is an ESTIMATE ONLY. Real-world accuracy requires
# calibration with a reference object of known size. Adjust this value
# for your specific camera setup and mounting height.
PIXEL_TO_CM_BASELINE = 0.35

# Preprocessing settings
CLAHE_CLIP_LIMIT = 3.0          # Contrast enhancement clip limit
CLAHE_TILE_GRID_SIZE = (8, 8)   # CLAHE tile grid size
GAUSSIAN_BLUR_KERNEL = (3, 3)   # Gaussian blur kernel size

# ============================================================
# Depth Estimation Settings
# ============================================================
# MiDaS model settings
MIDAS_MODEL_TYPE = "DPT_Hybrid"  # "DPT_Hybrid", "DPT_Large", "DPT_Small"
MIDAS_TARGET_SIZE = (384, 384)   # Input size for MiDaS

# Depth range for normalization (in meters, relative)
MIN_DEPTH = 0.1
MAX_DEPTH = 10.0

# ============================================================
# Severity Classification Thresholds
# ============================================================
# Size thresholds in cm²
SIZE_SMALL_MAX_CM2 = 500        # < 500 cm² → Small
SIZE_MEDIUM_MAX_CM2 = 2000      # 500–2000 cm² → Medium
# > 2000 cm² → Large

# Depth thresholds in cm (estimated)
DEPTH_SHALLOW_MAX_CM = 2.0      # < 2 cm → Shallow
DEPTH_MODERATE_MAX_CM = 5.0     # 2–5 cm → Moderate
# > 5 cm → Deep

# Upper bound used to map normalized relative-depth differences to a
# plausible physical pothole depth in cm. Calibrate against measured
# ground truth for survey-grade accuracy.
DEPTH_MAX_CM_ESTIMATE = 25.0

# Severity priority mapping
SEVERITY_LEVELS = {
    "Critical": {"color": "red", "priority": 1, "response_time_hours": 24},
    "High": {"color": "orange", "priority": 2, "response_time_hours": 168},
    "Medium": {"color": "yellow", "priority": 3, "response_time_hours": 720},
    "Low": {"color": "green", "priority": 4, "response_time_hours": 2160},
}

# Size categories
SIZE_CATEGORIES = {
    "Small": {"min": 0, "max": SIZE_SMALL_MAX_CM2},
    "Medium": {"min": SIZE_SMALL_MAX_CM2, "max": SIZE_MEDIUM_MAX_CM2},
    "Large": {"min": SIZE_MEDIUM_MAX_CM2, "max": float("inf")},
}

# Depth categories
DEPTH_CATEGORIES = {
    "Shallow": {"min": 0, "max": DEPTH_SHALLOW_MAX_CM},
    "Moderate": {"min": DEPTH_SHALLOW_MAX_CM, "max": DEPTH_MODERATE_MAX_CM},
    "Deep": {"min": DEPTH_MODERATE_MAX_CM, "max": float("inf")},
}

# ============================================================
# API Settings
# ============================================================
API_HOST = os.environ.get("HOST", "0.0.0.0")
# Render / Railway / Heroku inject $PORT at runtime; fall back to 5000 locally.
API_PORT = int(os.environ.get("PORT", "5000"))
# Debug must be OFF in production (never expose the Werkzeug debugger).
# Set API_DEBUG=true explicitly for local development only.
API_DEBUG = os.environ.get("API_DEBUG", "false").lower() == "true"

# Maximum upload size (MB)
MAX_CONTENT_LENGTH_MB = 50

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}

# ============================================================
# Web Frontend Settings
# ============================================================
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000

# Mapbox / Leaflet settings
MAP_DEFAULT_LAT = 37.7749    # Default center latitude
MAP_DEFAULT_LNG = -122.4194  # Default center longitude
MAP_DEFAULT_ZOOM = 15

# ============================================================
# Flask Secret Key (for session management)
# ============================================================
SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "pothole-detection-secret-key-2024")

# ============================================================
# Database Settings (SQLite for development)
# ============================================================
DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "pothole_reports.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# ============================================================
# Logging Settings
# ============================================================
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "app.log")

# ============================================================
# Utility Functions
# ============================================================
def ensure_directories():
    """Create all required directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, SAMPLES_DIR, UPLOADS_DIR, os.path.dirname(LOG_FILE)]:
        os.makedirs(d, exist_ok=True)

# Ensure directories exist on import
ensure_directories()