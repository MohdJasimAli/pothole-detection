"""
Severity Classification Module
==============================
Combines size and depth measurements to classify pothole
severity and assign maintenance priority levels.

Classes:
    - SeverityClassifier: Classifies pothole severity
    - SeverityResult: Container for classification output

Usage:
    from src.severity import SeverityClassifier

    classifier = SeverityClassifier()
    result = classifier.classify(size_result, depth_result)
    print(result.severity, result.priority)
"""

from typing import Tuple, Optional
from dataclasses import dataclass

import config
from .utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class SeverityResult:
    """Container for severity classification results."""
    severity: str                # Critical, High, Medium, Low
    size_category: str           # Small, Medium, Large
    depth_category: str          # Shallow, Moderate, Deep
    priority: int                # 1=Critical, 4=Low (lower = more urgent)
    color_bgr: Tuple[int, int, int]  # BGR color for visualization
    response_time_hours: int     # Recommended response deadline
    recommendation: str          # Maintenance recommendation text
    score: float = 0.0           # Numerical severity score (0-100)


class SeverityClassifier:
    """
    Classify pothole severity based on size and depth.

    Uses a rule-based system combined with configurable thresholds
    from the config module.
    """

    def __init__(self):
        """Initialize the severity classifier with config thresholds."""
        self.size_thresholds = config.SIZE_CATEGORIES
        self.depth_thresholds = config.DEPTH_CATEGORIES
        self.severity_levels = config.SEVERITY_LEVELS
        logger.info("SeverityClassifier initialized")

    def classify(self, size_result=None, depth_cm: float = None,
                 depth_category: str = None, area_cm2: float = None,
                 size_category: str = None) -> SeverityResult:
        """
        Classify pothole severity.

        Can accept either SizeResult/DepthResult objects or raw values.

        Args:
            size_result: SizeResult object (optional)
            depth_cm: Depth in centimeters
            depth_category: Depth category string
            area_cm2: Pothole area in cm²
            size_category: Size category string

        Returns:
            SeverityResult with full classification
        """
        # Extract size info
        if size_result is not None:
            area_cm2 = size_result.area_cm2
            size_category = size_result.size_category
        elif area_cm2 is None or size_category is None:
            area_cm2 = 0
            size_category = "Unknown"

        # Extract depth info
        if depth_cm is not None and depth_category is None:
            depth_category = self._category_from_depth(depth_cm)

        if depth_cm is None:
            depth_cm = 0
        if depth_category is None:
            depth_category = "Shallow"

        # Calculate severity score (0-100)
        score = self._calculate_score(area_cm2, depth_cm, size_category, depth_category)

        # Determine severity level
        severity, priority, color, response_hours = self._determine_severity(score)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            severity, size_category, depth_category
        )

        logger.info(
            f"Severity: {severity} (Priority {priority}), "
            f"Score: {score:.1f}, Size: {size_category}, Depth: {depth_category}"
        )

        return SeverityResult(
            severity=severity,
            size_category=size_category,
            depth_category=depth_category,
            priority=priority,
            color_bgr=color,
            response_time_hours=response_hours,
            recommendation=recommendation,
            score=round(score, 2),
        )

    def _calculate_score(self, area_cm2: float, depth_cm: float,
                         size_category: str, depth_category: str) -> float:
        """
        Calculate severity score (0-100).

        Combines size and depth with weighted scoring.
        """
        # Size score (0-50)
        if size_category == "Small":
            size_score = min(area_cm2 / config.SIZE_SMALL_MAX_CM2 * 15, 15)
        elif size_category == "Medium":
            ratio = (area_cm2 - config.SIZE_SMALL_MAX_CM2) / \
                    (config.SIZE_MEDIUM_MAX_CM2 - config.SIZE_SMALL_MAX_CM2)
            size_score = 15 + ratio * 25  # 15-40
        else:
            ratio = min((area_cm2 - config.SIZE_MEDIUM_MAX_CM2) / 2000, 1.0)
            size_score = 40 + ratio * 10  # 40-50

        # Depth score (0-50)
        if depth_category == "Shallow":
            depth_score = min(depth_cm / config.DEPTH_SHALLOW_MAX_CM * 10, 10)
        elif depth_category == "Moderate":
            ratio = (depth_cm - config.DEPTH_SHALLOW_MAX_CM) / \
                    (config.DEPTH_MODERATE_MAX_CM - config.DEPTH_SHALLOW_MAX_CM)
            depth_score = 10 + ratio * 30  # 10-40
        else:
            ratio = min((depth_cm - config.DEPTH_MODERATE_MAX_CM) / 5.0, 1.0)
            depth_score = 40 + ratio * 10  # 40-50

        return size_score + depth_score

    def _determine_severity(self, score: float) -> Tuple[str, int, Tuple[int, int, int], int]:
        """Map severity score to severity level, priority, and color."""
        if score >= 80:
            return ("Critical", 1, (0, 0, 255), 24)
        elif score >= 60:
            return ("High", 2, (0, 165, 255), 168)
        elif score >= 35:
            return ("Medium", 3, (0, 255, 255), 720)
        else:
            return ("Low", 4, (0, 255, 0), 2160)

    def _category_from_depth(self, depth_cm: float) -> str:
        """Determine depth category from raw depth value."""
        if depth_cm < config.DEPTH_SHALLOW_MAX_CM:
            return "Shallow"
        elif depth_cm < config.DEPTH_MODERATE_MAX_CM:
            return "Moderate"
        else:
            return "Deep"

    def _generate_recommendation(self, severity: str,
                                 size_category: str,
                                 depth_category: str) -> str:
        """Generate a maintenance recommendation string."""
        if severity == "Critical":
            return (f"CRITICAL: Immediate repair required. "
                    f"Large ({size_category}) and/or deep ({depth_category}) pothole "
                    f"posing severe safety risk. Dispatch crew within 24h.")
        elif severity == "High":
            return (f"HIGH PRIORITY: Schedule repair within 1 week. "
                    f"Size: {size_category}, Depth: {depth_category}. "
                    f"Monitor for rapid deterioration.")
        elif severity == "Medium":
            return (f"MEDIUM PRIORITY: Plan repair within 1 month. "
                    f"Size: {size_category}, Depth: {depth_category}. "
                    f"Include in quarterly maintenance schedule.")
        else:
            return (f"LOW PRIORITY: Routine monitoring. "
                    f"Size: {size_category}, Depth: {depth_category}. "
                    f"Include in annual maintenance planning.")

    def classify_matrix(self, size_category: str, depth_category: str) -> SeverityResult:
        """
        Classify using a pre-defined severity matrix.

        This provides an alternative rule-based approach
        for quick severity assignment.
        """
        matrix = {
            ("Small", "Shallow"): ("Low", 4, 2160),
            ("Small", "Moderate"): ("Medium", 3, 720),
            ("Small", "Deep"): ("High", 2, 168),
            ("Medium", "Shallow"): ("Medium", 3, 720),
            ("Medium", "Moderate"): ("High", 2, 168),
            ("Medium", "Deep"): ("Critical", 1, 24),
            ("Large", "Shallow"): ("High", 2, 168),
            ("Large", "Moderate"): ("Critical", 1, 24),
            ("Large", "Deep"): ("Critical", 1, 24),
        }

        if (size_category, depth_category) in matrix:
            sev, pri, hours = matrix[(size_category, depth_category)]
        else:
            sev, pri, hours = ("Low", 4, 2160)

        colors = {
            "Critical": (0, 0, 255),
            "High": (0, 165, 255),
            "Medium": (0, 255, 255),
            "Low": (0, 255, 0),
        }

        recommendation = self._generate_recommendation(sev, size_category, depth_category)

        return SeverityResult(
            severity=sev,
            size_category=size_category,
            depth_category=depth_category,
            priority=pri,
            color_bgr=colors[sev],
            response_time_hours=hours,
            recommendation=recommendation,
            score=0.0,
        )
