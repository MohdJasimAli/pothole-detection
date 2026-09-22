"""
Flask API Application for Pothole Detection
==========================================
REST API endpoints for automated pothole detection and severity analysis.

Endpoints:
    GET  /          - API info
    GET  /dashboard - Web dashboard
    POST /api/detect - Detect potholes in uploaded image
    POST /api/analyze - Full analysis (detection + size + depth + severity)
    GET  /api/health - Health check
    GET  /api/reports - List reports
    GET  /api/report/<id> - Get specific report

Usage:
    python api/app.py
    or
    python main.py --web
"""

import os
import io
import json
import sys
import uuid

# Ensure the project root is on sys.path so that `python api/app.py` works.
# Running a script inside a subpackage only puts that subpackage's directory
# on sys.path by default, which would break `import config` and `import src`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
from glob import glob
from threading import Thread

from flask import Flask, request, jsonify, render_template, \
    send_from_directory, Response, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

import config
from src.detector import PotholeDetector, DetectionResult
from src.depth_estimator import DepthEstimator
from src.size_estimator import SizeEstimator
from src.severity import SeverityClassifier
from main import PotholeAnalysisPipeline
from src.utils import setup_logger, load_image_from_bytes, save_image, \
    image_to_base64, save_json, allowed_file

logger = setup_logger("api", config.LOG_FILE)

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "static"),
    static_url_path="/static",
)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH_MB * 1024 * 1024
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOADS_DIR
CORS(app)

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Lazy-load pipeline (loaded on first request to handle model downloads)
_pipeline = None


def get_pipeline() -> PotholeAnalysisPipeline:
    """Lazy-initialize the analysis pipeline."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing analysis pipeline...")
        _pipeline = PotholeAnalysisPipeline()
        logger.info("Pipeline ready")
    return _pipeline


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    """API root endpoint."""
    return jsonify({
        "service": "Pothole Detection & Severity Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/api/health": "GET - Health check",
            "/api/detect": "POST - Upload image, get detection results",
            "/api/analyze": "POST - Full analysis (detect + size + depth + severity)",
            "/dashboard": "GET - Web dashboard",
            "/api/reports": "GET - List all reports",
        },
    })


@app.route("/dashboard")
def dashboard():
    """Serve the web dashboard."""
    return render_template("index.html")


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": config.DEFAULT_DEVICE,
    })


@app.route("/api/detect", methods=["POST"])
def detect_potholes():
    """
    Detect potholes in an uploaded image.

    Expects: multipart/form-data with 'image' file
    Optional: 'gps_lat', 'gps_lng' as form fields

    Returns:
        JSON with detection results and annotated image (base64)
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    # Load image
    image_bytes = file.read()
    image = load_image_from_bytes(image_bytes)

    if image is None:
        return jsonify({"error": "Failed to load image"}), 400

    # Get GPS coords if provided
    gps_lat = request.form.get("gps_lat", type=float)
    gps_lng = request.form.get("gps_lng", type=float)
    gps = (gps_lat, gps_lng) if gps_lat and gps_lng else None

    try:
        pipeline = get_pipeline()
        result = pipeline.analyze_image(image, gps_coords=gps, visualize=True)

        # Save annotated image
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        annotated_filename = f"{os.path.splitext(filename)[0]}_{unique_id}_annotated.jpg"
        annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], annotated_filename)
        save_image(result["annotated_image"], annotated_path)

        # Save report
        report_filename = f"{os.path.splitext(filename)[0]}_{unique_id}_report.json"
        report_path = os.path.join(app.config["UPLOAD_FOLDER"], report_filename)
        save_json(result["report"], report_path)

        # Add paths to response
        result["report"]["annotated_image_path"] = annotated_path
        result["report"]["report_path"] = report_path
        result["report"]["report_id"] = unique_id

        return jsonify(result["report"])

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_pothole():
    """
    Full analysis: detection + size + depth + severity.

    Expects: multipart/form-data with 'image' file
    Optional: 'gps_lat', 'gps_lng' as form fields

    Returns:
        JSON with complete analysis report and base64 annotated image
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    image = load_image_from_bytes(image_bytes)

    if image is None:
        return jsonify({"error": "Failed to load image"}), 400

    gps_lat = request.form.get("gps_lat", type=float)
    gps_lng = request.form.get("gps_lng", type=float)
    gps = (gps_lat, gps_lng) if gps_lat and gps_lng else None

    try:
        pipeline = get_pipeline()
        result = pipeline.analyze_image(image, gps_coords=gps, visualize=True)

        # Convert annotated image to base64
        annotated_b64 = image_to_base64(result["annotated_image"])

        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "height": image.shape[0],
                "width": image.shape[1],
            },
            "detections": result["detections"],
            "annotated_image_base64": annotated_b64,
            "report": result["report"],
            "processing_time_seconds": result["processing_time_seconds"],
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/reports", methods=["GET"])
def list_reports():
    """List all saved reports."""
    upload_dir = app.config["UPLOAD_FOLDER"]
    reports = []

    for f in glob(os.path.join(upload_dir, "*_report.json")):
        try:
            with open(f, "r") as fh:
                report = json.load(fh)
            report["report_file"] = os.path.basename(f)
            report["annotated_image_file"] = os.path.basename(
                f).replace("_report.json", "_annotated.jpg")
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to read report {f}: {e}")

    return jsonify({"reports": reports, "count": len(reports)})


@app.route("/api/report/<report_id>", methods=["GET"])
def get_report(report_id):
    """Get a specific report by ID."""
    upload_dir = app.config["UPLOAD_FOLDER"]

    for f in glob(os.path.join(upload_dir, f"*_{report_id}_report.json")):
        with open(f, "r") as fh:
            report = json.load(fh)
        return jsonify(report)

    return jsonify({"error": "Report not found"}), 404


@app.route("/api/visualize", methods=["POST"])
def visualize_detections_endpoint():
    """Endpoint to just get the annotated image without full analysis."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    image = load_image_from_bytes(image_bytes)

    if image is None:
        return jsonify({"error": "Failed to load image"}), 400

    try:
        pipeline = get_pipeline()
        result = pipeline.analyze_image(image, gps_coords=None, visualize=True)
        annotated_b64 = image_to_base64(result["annotated_image"])

        return jsonify({
            "status": "success",
            "annotated_image_base64": annotated_b64,
            "detections": result["detections"],
        })

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large (max 50MB)"}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.API_DEBUG,
    )
