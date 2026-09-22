# Model Files Directory

This directory contains trained model weights for the pothole detection system.

## Expected Files

- `best.pt` - Trained YOLOv8 pothole detection model
- `midas/` - MiDaas depth estimation model cache (auto-downloaded)

## Download Instructions

### YOLOv8 Pothole Detection Model
If you don't have a custom-trained model, the system will automatically
fall back to a generic YOLOv8 model (yolov8s.pt) on first run.

To train your own model on pothole data:
1. Prepare dataset in YOLO format (images + .txt annotation files)
2. Use YOLOv8 training: `yolo task=detect mode=train data=dataset.yaml model=yolov8s.pt epochs=100`

### MiDaas Depth Model
The MiDaas model is automatically downloaded via torch.hub on first run.
It will be cached in this directory.
