"""
Unit Tests for Road Pothole Detection System
=============================================

Run with: pytest tests/ -v

Tests cover:
- Configuration validation
- Preprocessing functions
- Size estimation
- Severity classification
- Utility functions (bbox extraction, IoU, etc.)
"""

import os
import sys
import json
import numpy as np
import cv2
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import calculate_iou, generate_report_json
from src.size_estimator import SizeEstimator
from src.severity import SeverityClassifier
from src.preprocessing import enhance_contrast, reduce_noise, normalize_image


class TestConfig:
    """Test configuration constants."""

    def test_config_values_exist(self):
        assert hasattr(config, "CONFIDENCE_THRESHOLD")
        assert hasattr(config, "IOU_THRESHOLD")
        assert hasattr(config, "CLASS_NAMES")
        assert hasattr(config, "SIZE_SMALL_MAX_CM2")
        assert hasattr(config, "SIZE_MEDIUM_MAX_CM2")
        assert hasattr(config, "DEPTH_SHALLOW_MAX_CM")
        assert hasattr(config, "DEPTH_MODERATE_MAX_CM")

    def test_confidence_threshold_range(self):
        assert 0.0 < config.CONFIDENCE_THRESHOLD < 1.0

    def test_size_thresholds_ordering(self):
        assert config.SIZE_SMALL_MAX_CM2 < config.SIZE_MEDIUM_MAX_CM2

    def test_depth_thresholds_ordering(self):
        assert config.DEPTH_SHALLOW_MAX_CM < config.DEPTH_MODERATE_MAX_CM

    def test_severity_levels_exist(self):
        for level in ["Critical", "High", "Medium", "Low"]:
            assert level in config.SEVERITY_LEVELS


class TestPreprocessing:
    """Test image preprocessing functions."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return img

    def test_enhance_contrast(self, sample_image):
        enhanced = enhance_contrast(sample_image)
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == sample_image.dtype

    def test_reduce_noise(self, sample_image):
        denoised = reduce_noise(sample_image)
        assert denoised.shape == sample_image.shape
        assert denoised.dtype == sample_image.dtype

    def test_normalize_image(self, sample_image):
        normalized = normalize_image(sample_image)
        assert normalized.shape == sample_image.shape
        assert normalized.dtype == np.float32

    def test_preprocess_preserves_shape(self, sample_image):
        from src.preprocessing import preprocess_image
        result = preprocess_image(sample_image, normalize=False)
        assert result.shape == sample_image.shape


class TestSizeEstimator:
    """Test size estimation logic."""

    @pytest.fixture
    def estimator(self):
        return SizeEstimator()

    def test_size_classification_small(self, estimator):
        result = estimator.estimate_size((0, 0, 20, 20), 640)
        assert result.size_category in ["Small", "Unknown"]

    def test_size_classification_medium(self, estimator):
        # Create a bbox that should produce medium size area
        result = estimator.estimate_size((0, 0, 200, 200), 640)
        assert result.size_category in ["Small", "Medium", "Large"]

    def test_size_calculation_positive(self, estimator):
        result = estimator.estimate_size((0, 0, 100, 100), 640)
        assert result.width_cm > 0
        assert result.height_cm > 0
        assert result.area_cm2 > 0

    def test_size_result_attributes(self, estimator):
        result = estimator.estimate_size((10, 10, 110, 110), 640)
        assert hasattr(result, "width_cm")
        assert hasattr(result, "height_cm")
        assert hasattr(result, "area_cm2")
        assert hasattr(result, "size_category")
        assert hasattr(result, "pixel_density_cm_per_px")


class TestSeverityClassifier:
    """Test severity classification logic."""

    @pytest.fixture
    def classifier(self):
        return SeverityClassifier()

    def test_low_severity_small_shallow(self, classifier):
        result = classifier.classify(
            area_cm2=100, size_category="Small",
            depth_cm=1.0, depth_category="Shallow"
        )
        assert result.severity == "Low"
        assert result.priority == 4

    def test_critical_severity_large_deep(self, classifier):
        result = classifier.classify(
            area_cm2=5000, size_category="Large",
            depth_cm=8.0, depth_category="Deep"
        )
        assert result.severity == "Critical"
        assert result.priority == 1

    def test_medium_severity_medium_moderate(self, classifier):
        result = classifier.classify(
            area_cm2=800, size_category="Medium",
            depth_cm=3.5, depth_category="Moderate"
        )
        assert result.severity in ["Medium", "High"]

    def test_severity_result_has_all_fields(self, classifier):
        result = classifier.classify(
            area_cm2=500, size_category="Medium",
            depth_cm=2.5, depth_category="Moderate"
        )
        assert hasattr(result, "severity")
        assert hasattr(result, "priority")
        assert hasattr(result, "color_bgr")
        assert hasattr(result, "response_time_hours")
        assert hasattr(result, "recommendation")

    def test_matrix_classification(self, classifier):
        result = classifier.classify_matrix("Large", "Deep")
        assert result.severity == "Critical"

    def test_recommendation_generated(self, classifier):
        result = classifier.classify(
            area_cm2=5000, size_category="Large",
            depth_cm=10.0, depth_category="Deep"
        )
        assert len(result.recommendation) > 0
        assert "Immediate" in result.recommendation or "repair" in result.recommendation.lower()


class TestUtils:
    """Test utility functions."""

    def test_calculate_iou_perfect_overlap(self):
        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]
        assert calculate_iou(box1, box2) == 1.0

    def test_calculate_iou_no_overlap(self):
        box1 = [0, 0, 10, 10]
        box2 = [100, 100, 200, 200]
        assert calculate_iou(box1, box2) == 0.0

    def test_calculate_iou_partial_overlap(self):
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = calculate_iou(box1, box2)
        assert 0.0 < iou < 1.0

    def test_generate_report_structure(self):
        detections = [
            {"bbox": [10, 20, 50, 60], "confidence": 0.95,
             "severity": "High", "size_cm2": 800, "size_category": "Medium"},
            {"bbox": [100, 200, 150, 250], "confidence": 0.85,
             "severity": "Low", "size_cm2": 200, "size_category": "Small"},
        ]
        report = generate_report_json(detections, "test.jpg", (37.7749, -122.4194))
        assert "timestamp" in report
        assert "gps_coordinates" in report
        assert "total_potholes_detected" in report
        assert report["total_potholes_detected"] == 2
        assert report["summary"]["high_count"] == 1
        assert report["summary"]["low_count"] == 1


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    def test_detection_to_dict(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from src.detector import DetectionResult

        det = DetectionResult(
            bbox=[10, 20, 100, 200],
            confidence=0.95,
            class_id=0,
            class_name="Pothole",
            width_px=90,
            height_px=180,
            center=(55, 110),
            size_cm2=750.0,
            size_category="Medium",
            width_cm=15.5,
            height_cm=31.0,
            depth_cm=3.5,
            depth_category="Moderate",
            severity="High",
            priority=2,
            response_time_hours=168,
        )

        d = det.to_dict()
        assert d["class_name"] == "Pothole"
        assert d["confidence"] == 0.95
        assert d["severity"] == "High"
        assert d["priority"] == 2


class TestIntegration:
    """Integration test for the full pipeline (requires torch)."""

    @pytest.fixture
    def real_image(self):
        """Load a sample road image if available."""
        sample_paths = [
            os.path.join(config.SAMPLES_DIR, "road1.jpg"),
            os.path.join(config.SAMPLES_DIR, "road2.jpg"),
            os.path.join(config.SAMPLES_DIR, "sample.jpg"),
        ]
        for path in sample_paths:
            if os.path.exists(path):
                return cv2.imread(path)
        # Create a dummy image if no sample exists
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.mark.skipif(not os.path.exists(config.YOLO_MODEL_PATH),
                        reason="YOLO model not available")
    def test_full_pipeline(self, real_image):
        """Test the complete detection pipeline."""
        from main import PotholeAnalysisPipeline

        pipeline = PotholeAnalysisPipeline()
        result = pipeline.analyze_image(real_image)

        assert "detections" in result
        assert "report" in result
        assert "processing_time_seconds" in result
        assert "annotated_image" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
