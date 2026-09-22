# Documentation Index

Complete documentation for the **Road Pothole Detection & Severity Analysis** project.

---

## Documents

| Document | Description | Use Case |
|----------|-------------|----------|
| [RESEARCH_PAPER.md](RESEARCH_PAPER.md) | Full academic research report with literature review, methodology, experiments, and results | Research submission, report writing, thesis chapter |
| [PRESENTATION_CONTENT.md](PRESENTATION_CONTENT.md) | 24 ready-to-use slides with speaker notes, timing guide, and anticipated Q&A | PowerPoint / Google Slides presentation |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API reference with endpoints, data models, and integration examples | Backend integration, developer reference |
| [../README.md](../README.md) | Project overview, installation, and quick start guide | Getting started, setup |

---

## Quick Navigation

### For Academic / Research Use
1. Start with **RESEARCH_PAPER.md** — contains abstract, literature review, methodology, results
2. Use **PRESENTATION_CONTENT.md** for your defense or conference presentation
3. Cite the references from Section 9 of the research paper

### For Development / Deployment
1. Read **../README.md** for installation and setup
2. Reference **API_DOCUMENTATION.md** for integration
3. See `train.py` docstrings for model training

### For Presentation Preparation
1. Copy slide content from **PRESENTATION_CONTENT.md** into PowerPoint
2. Add your own screenshots from the dashboard (`/dashboard`)
3. Use the architecture diagram from Slide 8
4. Follow the timing guide for a 20-minute presentation

---

## Project Summary

**Goal:** Automatically detect potholes from road images and classify their severity for smart road maintenance.

**Pipeline:**
```
Image -> Preprocessing -> YOLOv8 Detection -> Size Estimation
      -> MiDaS Depth Estimation -> Severity Classification -> Report
```

**Key Results:**
- Detection accuracy (mAP@0.5): **92%**
- Size estimation error: **12%**
- Depth classification accuracy: **87%**
- Inference speed: **45 FPS (GPU)** / **18 FPS (CPU)**

**Severity Levels:** Critical (24h) / High (1 week) / Medium (1 month) / Low (3 months)

---

## Code Structure

```
ProjectDev/
├── config.py              # All configuration constants
├── main.py                # Pipeline orchestrator (CLI)
├── train.py               # YOLOv8 training script
├── run.py                 # Simple runner
├── src/
│   ├── preprocessing.py   # CLAHE, denoising, shadow removal
│   ├── detector.py        # YOLOv8 pothole detection
│   ├── depth_estimator.py # MiDaS depth estimation
│   ├── size_estimator.py  # Pixel-to-cm conversion
│   ├── severity.py        # Severity scoring & classification
│   └── utils.py           # Helper functions
├── api/
│   └── app.py             # Flask REST API
├── frontend/              # Web dashboard
├── tests/                 # Unit tests (25 tests)
├── notebooks/             # Jupyter analysis notebook
└── docs/                  # This documentation
```

---

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Analyze a single image
python main.py --image data/samples/road1.jpg

# Process a folder
python main.py --batch data/samples/ --output outputs/

# Start web dashboard
python main.py --web
# Then open http://localhost:5000/dashboard

# Train a custom model
python train.py --create-config
python train.py --data dataset.yaml --epochs 100 --model yolov8s.pt

# Validate trained model
python train.py --validate runs/train/pothole_model/weights/best.pt

# Export for deployment
python train.py --export runs/train/pothole_model/weights/best.pt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | `pip install torch ultralytics` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| Model not found | Run `train.py` or place `best.pt` in `data/models/` |
| Slow inference | Use GPU: set `DEFAULT_DEVICE = "cuda"` in config.py |
| Import errors | Run commands from the project root directory |

---

**Version:** 1.0.0
**Last Updated:** 2024
**Authors:** ProjectDev Team