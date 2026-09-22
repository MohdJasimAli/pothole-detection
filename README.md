# Road Pothole Detection & Severity Analysis

A smart road maintenance system that uses computer vision and deep learning to automatically detect potholes from road images and estimate their size, depth, and severity for proactive road maintenance.

## 🚀 Features

- **Real-time Pothole Detection**: YOLOv8-based object detection for identifying potholes in road images
- **Size Estimation**: Converts pixel measurements to real-world dimensions (cm) using camera calibration
- **Depth Estimation**: MiDaS monocular depth estimation for measuring pothole depth
- **Severity Classification**: Rule-based classification into Critical, High, Medium, and Low priority
- **Web Dashboard**: Interactive map-based interface for visualization and reporting
- **REST API**: Full API for integration with existing municipal systems
- **Batch Processing**: Process entire folders of road images at once

## Documentation

Comprehensive documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| **docs/RESEARCH_PAPER.md** | Full academic research report - literature review, methodology, experiments, results, references |
| **docs/PRESENTATION_CONTENT.md** | 24 ready-to-use presentation slides with speaker notes and timing guide |
| **docs/API_DOCUMENTATION.md** | Complete REST API reference with examples and data models |
| **docs/README.md** | Documentation index and quick navigation |

## 📦 Project Structure

```
ProjectDev/
├── config.py                    # Configuration settings and thresholds
├── main.py                      # Pipeline orchestrator (CLI entry point)
├── run.py                       # Simple runner script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── src/
│   ├── __init__.py              # Package initialization
│   ├── preprocessing.py         # Image preprocessing (CLAHE, denoising, etc.)
│   ├── detector.py              # YOLOv8 pothole detection
│   ├── depth_estimator.py       # MiDaS depth estimation
│   ├── size_estimator.py        # Pixel-to-cm size conversion
│   ├── severity.py              # Severity classification (score + matrix)
│   └── utils.py                 # Utility functions (logging, IO, visualization)
├── api/
│   ├── __init__.py              # API package init
│   └── app.py                   # Flask REST API with web dashboard
├── frontend/
│   ├── templates/
│   │   └── index.html           # Web dashboard HTML
│   ├── static/
│   │   ├── css/style.css        # Dashboard styling
│   │   ├── js/app.js            # Frontend JavaScript
│   │   ├── images/              # Static images
│   │   └── models/              # Client-side models (if applicable)
│   └── uploads/                 # Uploaded image storage
├── data/
│   ├── models/                  # Trained model weights (.pt files)
│   └── samples/                 # Sample road images for testing
├── tests/
│   └── test_detector.py         # Unit and integration tests
└── notebooks/
    └── analysis_template.ipynb  # Jupyter notebook for analysis
```

## 🔧 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended for real-time processing)
- pip package manager

### Setup

```bash
# Clone or navigate to the project directory
cd E:\Project\CODE\ProjectDev

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Ultralytics YOLOv5 dependencies
pip install ultralytics

# Download pre-trained models
# YOLOv8: Will auto-download on first run (e.g., yolov8s.pt)
# MiDaS: Will auto-download via torch.hub

# Run tests
pytest tests/ -v
```

## 🚀 Quick Start

### Command Line Usage

```bash
# Analyze a single image
python main.py --image data/samples/road1.jpg

# Process a batch of images
python main.py --batch data/samples/

# Start the web dashboard
python main.py --web

# Analyze with GPS coordinates
python main.py --image road.jpg --gps-lat 37.7749 --gps-lng -122.4194 --output outputs/
```

### Python API Usage

```python
from main import PotholeAnalysisPipeline

# Initialize the pipeline
pipeline = PotholeAnalysisPipeline()

# Analyze a single image
image = cv2.imread("road.jpg")
result = pipeline.analyze_image(image, gps_coords=(37.7749, -122.4194))

# Access results
for detection in result["detections"]:
    print(f"Size: {detection['size_cm2']} cm2 ({detection['size_category']})")
    print(f"Depth: {detection['depth_cm']} cm ({detection['depth_category']})")
    print(f"Severity: {detection['severity']} (Priority {detection['priority']})")
```

### REST API Usage

```bash
# Start the API server
python api/app.py

# Or via main
python main.py --web
```

```python
# Upload an image for analysis
import requests

with open("road.jpg", "rb") as f:
    files = {"image": f}
    data = {"gps_lat": 37.7749, "gps_lng": -122.4194}
    response = requests.post("http://localhost:5000/api/analyze", files=files, data=data)
    
result = response.json()
print(result["report"])
```

## ⚙️ Configuration

All configuration is managed in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.4 | Minimum detection confidence |
| `IOU_THRESHOLD` | 0.45 | NMS IoU threshold |
| `IMAGE_SIZE` | 640 | YOLO input image size |
| `SIZE_SMALL_MAX_CM2` | 500 | Max area for "Small" classification |
| `SIZE_MEDIUM_MAX_CM2` | 2000 | Max area for "Medium" classification |
| `DEPTH_SHALLOW_MAX_CM` | 2.0 | Max depth for "Shallow" |
| `DEPTH_MODERATE_MAX_CM` | 5.0 | Max depth for "Moderate" |
| `CAMERA_HEIGHT_CM` | 80 | Camera height from ground |
| `FOCAL_LENGTH_PIXELS` | 800 | Camera focal length |

## 📊 Severity Classification

The system classifies potholes into four severity levels:

| Severity | Size | Depth | Priority Response |
|----------|------|-------|-------------------|
| 🔴 Critical | Large | Deep (Moderate+) | 24 hours |
| 🟠 High | Medium+ | Moderate+ | 1 week |
| 🟡 Medium | Any | Shallow+ | 1 month |
| 🟢 Low | Small | Shallow | 3 months |

### Severity Matrix

| Size \ Depth | Shallow | Moderate | Deep |
|-------------|---------|----------|------|
| Small | Low | Low-Medium | Medium |
| Medium | Medium | High | High |
| Large | Medium-High | High | **Critical** |

## 🧠 Technical Architecture

### Detection Pipeline
```
[Input Image] → [Preprocessing] → [YOLOv8 Detection] → [Bounding Boxes]
     → [Size Estimation] → [Depth Estimation (MiDaS)] → [Severity Classification]
     → [Report + Annotated Image]
```

### Models Used
1. **YOLOv8**: State-of-the-art real-time object detector
   - Pre-trained on COCO, fine-tuned on pothole datasets
   - Input: 640x640 RGB images
   - Output: Bounding boxes with confidence scores

2. **MiDaS DPT_Hybrid**: Monocular depth estimation
   - Predicts relative depth from single RGB image
   - Used to estimate pothole depth relative to road surface

### Key Algorithms
- **CLAHE**: Contrast enhancement for better pothole visibility
- **Morphological Operations**: Shadow and noise reduction
- **Pixel-to-CM Conversion**: Camera geometry-based size estimation
- **Severity Scoring**: Weighted combination of size and depth metrics

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_detector.py::TestSeverityClassifier -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📸 Sample Output

The system produces:
1. **Annotated Image**: Original image with bounding boxes and severity labels
2. **JSON Report**: Complete analysis with all metrics
3. **Severity Summary**: Count by priority level
4. **CSV Export**: Tabular data for spreadsheet analysis

## 🛠️ Training Your Own Model

If you need to train a custom YOLOv8 model for your specific road conditions:

```bash
# 1. Prepare dataset in YOLO format
#   images/train/*.jpg, images/val/*.jpg
#   labels/train/*.txt, labels/val/*.txt

# 2. Create dataset YAML
echo 'path: ../datasets/pothole_data
train: images/train
val: images/val
names:
  0: Pothole' > pothole.yaml

# 3. Train
yolo task=detect mode=train data=pothole.yaml model=yolov8n.pt epochs=100 imgsz=640

# 4. Copy the best model to data/models/best.pt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- **Ultralytics YOLOv8** - Object detection framework
- **MiDaS** - Monocular depth estimation by Intel Labs
- **OpenCV** - Computer vision library
- **KOMATSU Road Damage Dataset** - Research dataset
- **PyTorch** - Deep learning framework

## 📞 Contact

- **Author**: ProjectDev Team
- **Email**: projectdev@example.com
- **GitHub**: github.com/projectdev/pothole-detection

---

> **Note**: This system requires a trained YOLOv8 model for pothole detection. You can download pre-trained models from [Ultralytics GitHub](https://github.com/ultralytics/ultralytics) or train your own using road images with pothole annotations.
