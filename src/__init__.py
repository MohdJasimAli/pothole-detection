"""
Road Pothole Detection and Severity Analysis System
=====================================================

Core computer vision modules for automated pothole detection,
size estimation, and depth/severity classification.

Modules:
    - preprocessing: Image preprocessing and enhancement
    - detector: YOLOv8-based pothole detection
    - depth_estimator: MiDaS-based monocular depth estimation
    - size_estimator: Pothole size estimation in real-world units
    - severity: Severity classification based on size and depth
    - utils: Utility functions and helpers

Author: ProjectDev Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "ProjectDev Team"
__email__ = "projectdev@example.com"

# Module-level constants

def _torch_usable():
    """Return True only if torch is installed AND its native libs load.

    The probe runs in a subprocess because a broken torch install can raise
    beyond ImportError/OSError (segfault / access violation), which would
    otherwise crash this interpreter during a plain `import torch as _torch`.
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
             "import torch as _torch; "
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
    import torch as _torch

    DEFAULT_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
else:
    DEFAULT_DEVICE = "cpu"

from . import preprocessing
from . import detector
from . import classical_detector
from . import depth_estimator
from . import size_estimator
from . import severity
from . import utils