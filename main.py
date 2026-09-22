"""
Main Pipeline Orchestrator
==========================
Ties together detection, depth estimation, size estimation,
and severity classification into a single pipeline.

Usage:
    python main.py --image road.jpg
    python main.py --batch data/samples/
    python main.py --web
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from glob import glob

import config
from src.detector import PotholeDetector, DetectionResult
from src.depth_estimator import DepthEstimator, DepthResult
from src.size_estimator import SizeEstimator, SizeResult
from src.severity import SeverityClassifier, SeverityResult
from src.utils import setup_logger, save_json, load_image, draw_bounding_boxes

logger = setup_logger("main", config.LOG_FILE)


class PotholeAnalysisPipeline:
    """
    Complete pipeline: detect -> estimate size -> estimate depth -> classify severity.

    This orchestrator combines all sub-modules into a single end-to-end system.
    """

    def __init__(self,
                 detector: PotholeDetector = None,
                 depth_estimator: DepthEstimator = None,
                 size_estimator: SizeEstimator = None,
                 severity_classifier: SeverityClassifier = None,
                 device: str = None):
        """
        Initialize the analysis pipeline.

        Args:
            detector: PotholeDetector instance (creates default if None)
            depth_estimator: DepthEstimator instance
            size_estimator: SizeEstimator instance
            severity_classifier: SeverityClassifier instance
            device: Device for inference
        """
        self.device = device or config.DEFAULT_DEVICE
        self.size_estimator = size_estimator or SizeEstimator()
        self.severity_classifier = severity_classifier or SeverityClassifier()

        # Detector: prefer YOLOv8, but fall back to the classical detector
        # when PyTorch/Ultralytics are unavailable (see config.ALLOW_CLASSICAL_FALLBACK).
        self.detection_backend = "yolo"
        if detector is not None:
            self.detector = detector
        else:
            try:
                self.detector = PotholeDetector(device=self.device)
            except Exception as e:
                if not getattr(config, "ALLOW_CLASSICAL_FALLBACK", True):
                    raise
                from src.classical_detector import ClassicalPotholeDetector
                logger.warning(
                    f"YOLOv8 unavailable ({e}); "
                    f"using ClassicalPotholeDetector fallback."
                )
                self.detector = ClassicalPotholeDetector()
                self.detection_backend = "classical"

        # Depth: prefer MiDaS, but synthesized brightness-based depth is used
        # inside DepthEstimator when its torch model cannot load.
        if depth_estimator is not None:
            self.depth_estimator = depth_estimator
        else:
            try:
                self.depth_estimator = DepthEstimator(device=self.device)
            except Exception as e:
                logger.warning(
                    f"MiDaS unavailable ({e}); depth will use the "
                    f"synthetic fallback inside DepthEstimator."
                )
                self.depth_estimator = DepthEstimator.__new__(DepthEstimator)
                self.depth_estimator.model = None
                self.depth_estimator.model_type = "synthetic-fallback"
                self.depth_estimator.device = self.device

        self.depth_backend = getattr(self.depth_estimator, "model_type", "unknown")

        logger.info(
            f"PotholeAnalysisPipeline initialized "
            f"(detector={self.detection_backend}, depth={self.depth_backend})"
        )

    def analyze_image(self, image: np.ndarray,
                      gps_coords: Tuple[float, float] = None,
                      reference_bbox: Tuple[int, int, int, int] = None,
                      visualize: bool = True) -> Dict:
        """
        Run the complete analysis pipeline on a single image.

        Args:
            image: Input road image (BGR format)
            gps_coords: Optional GPS coordinates (lat, lng)
            reference_bbox: Optional reference object bbox for calibration
            visualize: Whether to draw results on the image

        Returns:
            Dictionary containing:
                - detections: List of enriched detection results
                - annotated_image: Image with visualizations (if visualize=True)
                - report: Complete analysis report
        """
        start_time = datetime.now()
        img_h, img_w = image.shape[:2]

        # Step 1: Detect potholes
        logger.info("Step 1: Running pothole detection...")
        results = self.detector.detect(image)
        raw_detections = self.detector.extract_detections(results)
        logger.info(f"  Found {len(raw_detections)} pothole(s)")

        # Step 2: Estimate size for each detection
        logger.info("Step 2: Estimating pothole sizes...")
        for det in raw_detections:
            size_result = self.size_estimator.estimate_size(
                det.bbox, img_w, reference_bbox
            )
            det.width_cm = size_result.width_cm
            det.height_cm = size_result.height_cm
            det.size_cm2 = size_result.area_cm2
            det.size_category = size_result.size_category

        # Step 3: Estimate depth for each detection
        logger.info("Step 3: Estimating pothole depths...")
        depth_map = self.depth_estimator.estimate_depth(image)

        for det in raw_detections:
            depth_cm, depth_cat = self.depth_estimator.get_pothole_depth(
                depth_map, det.bbox
            )
            det.depth_cm = depth_cm
            det.depth_category = depth_cat

        # Step 4: Classify severity
        logger.info("Step 4: Classifying severity...")
        for det in raw_detections:
            severity_result = self.severity_classifier.classify(
                area_cm2=det.size_cm2,
                size_category=det.size_category,
                depth_cm=det.depth_cm,
                depth_category=det.depth_category,
            )
            det.severity = severity_result.severity
            det.priority = severity_result.priority
            det.response_time_hours = severity_result.response_time_hours

        # Step 5: Visualize
        annotated_image = None
        if visualize:
            annotated_image = self._annotate_image(image, raw_detections, depth_map)

        elapsed = (datetime.now() - start_time).total_seconds()

        # Generate report
        report = self._generate_report(
            raw_detections, image, gps_coords, elapsed
        )

        logger.info(f"Pipeline complete in {elapsed:.2f}s")

        return {
            "detections": [d.to_dict() for d in raw_detections],
            "annotated_image": annotated_image,
            "report": report,
            "processing_time_seconds": elapsed,
        }

    def _annotate_image(self, image: np.ndarray,
                        detections: List[DetectionResult],
                        depth_map: np.ndarray) -> np.ndarray:
        """Create annotated output image with boxes, labels, and severity colors."""
        from src.utils import draw_bounding_boxes as original_draw
        # Use custom drawing with severity colors
        img = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = det.get_color() if hasattr(det, 'get_color') else (0, 255, 0)

            # Use severity color
            from src.severity import SeverityClassifier
            temp_result = self.severity_classifier.classify(
                area_cm2=det.size_cm2,
                size_category=det.size_category,
                depth_cm=det.depth_cm,
                depth_category=det.depth_category,
            )
            color = temp_result.color_bgr

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"Pothole: {det.confidence:.2f} | "
            label += f"{det.size_cm2:.0f}cm2 ({det.size_category}) | "
            label += f"D:{det.depth_cm:.1f}cm ({det.depth_category}) | "
            label += f"{det.severity}"

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return img

    def _generate_report(self, detections: List[DetectionResult],
                         image: np.ndarray,
                         gps_coords: Tuple[float, float],
                         elapsed: float) -> Dict:
        """Generate a complete analysis report."""
        detection_dicts = [d.to_dict() for d in detections]

        critical = sum(1 for d in detections if d.severity == "Critical")
        high = sum(1 for d in detections if d.severity == "High")
        medium = sum(1 for d in detections if d.severity == "Medium")
        low = sum(1 for d in detections if d.severity == "Low")

        report = {
            "timestamp": datetime.now().isoformat(),
            "gps_coordinates": {"lat": gps_coords[0], "lng": gps_coords[1]} if gps_coords else None,
            "image_shape": {"height": image.shape[0], "width": image.shape[1]},
            "total_potholes": len(detections),
            "severity_summary": {
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low,
            },
            "potholes": detection_dicts,
            "processing_time_seconds": round(elapsed, 3),
            "model_info": {
                "detector": getattr(self, "detection_backend", "yolo"),
                "depth_model": getattr(self, "depth_backend",
                                       "MiDaS DPT_Hybrid"),
                "device": self.device,
            },
        }
        return report

    def analyze_from_path(self, image_path: str,
                          gps_coords: Tuple[float, float] = None,
                          output_dir: str = None) -> Dict:
        """Analyze an image from file path and optionally save results."""
        image = load_image(image_path)
        result = self.analyze_image(image, gps_coords=gps_coords)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(image_path))[0]

            # Save annotated image
            annotated_path = os.path.join(output_dir, f"{filename}_annotated.jpg")
            cv2.imwrite(annotated_path, result["annotated_image"])

            # Save report
            report_path = os.path.join(output_dir, f"{filename}_report.json")
            save_json(result["report"], report_path)

            result["annotated_image_path"] = annotated_path
            result["report_path"] = report_path

        return result

    def process_folder(self, folder_path: str,
                       gps_coords: Tuple[float, float] = None,
                       output_dir: str = None,
                       pattern: str = "*.*") -> List[Dict]:
        """Process all images in a folder."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
        files = glob(os.path.join(folder_path, "**", pattern), recursive=True)
        image_files = [f for f in files
                       if os.path.splitext(f)[1].lower() in image_extensions]

        results = []
        for image_path in image_files:
            logger.info(f"Processing: {image_path}")
            try:
                result = self.analyze_from_path(image_path, gps_coords, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")

        return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Road Pothole Detection and Severity Analysis"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image")
    parser.add_argument("--batch", type=str, default=None,
                        help="Path to folder for batch processing")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory for results")
    parser.add_argument("--gps-lat", type=float, default=None,
                        help="GPS latitude")
    parser.add_argument("--gps-lng", type=float, default=None,
                        help="GPS longitude")
    parser.add_argument("--web", action="store_true",
                        help="Start web API server")

    args = parser.parse_args()

    if args.web:
        # Start Flask web server
        from api.app import app
        app.run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG)
        return

    # Initialize pipeline
    logger.info("Initializing Pothole Analysis Pipeline...")
    pipeline = PotholeAnalysisPipeline()

    gps = None
    if args.gps_lat and args.gps_lng:
        gps = (args.gps_lat, args.gps_lng)

    if args.image:
        # Single image processing
        result = pipeline.analyze_from_path(args.image, gps, args.output)
        print(f"\n{'='*60}")
        print(f"Results for: {args.image}")
        print(f"{'='*60}")
        for det in result["detections"]:
            print(f"  Pothole: {det['confidence']:.2f} conf")
            print(f"  Size: {det['size_cm2']} cm2 ({det['size_category']})")
            print(f"  Depth: {det['depth_cm']} cm ({det['depth_category']})")
            print(f"  Severity: {det['severity']} (Priority {det['priority']})")
            print(f"  Response: {det['response_time_hours']}h")
        print(f"\nProcessing time: {result['processing_time_seconds']:.2f}s")
        print(f"Annotated image: {result.get('annotated_image_path', 'N/A')}")
        print(f"Report: {result.get('report_path', 'N/A')}")

    elif args.batch:
        results = pipeline.process_folder(args.batch, gps, args.output)
        print(f"\nProcessed {len(results)} images successfully.")

    else:
        print("No input specified. Use --image or --batch.")
        parser.print_help()


if __name__ == "__main__":
    main()
