"""
YOLOv8 Training Script for Pothole Detection
=============================================
Trains a custom YOLOv8 model on pothole datasets.

Usage:
    # 1. Create a dataset config template
    python train.py --create-config

    # 2. Prepare dataset in YOLO format:
    #    dataset/
    #      images/train/*.jpg
    #      images/val/*.jpg
    #      labels/train/*.txt
    #      labels/val/*.txt
    #      dataset.yaml

    # 3. Train:
    python train.py --data dataset.yaml --epochs 100 --model yolov8s.pt

    # 4. Validate:
    python train.py --validate runs/train/pothole_model/weights/best.pt

    # 5. Export to ONNX:
    python train.py --export runs/train/pothole_model/weights/best.pt
"""

import os
import sys
import argparse
import shutil
import yaml

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

import config
from src.utils import setup_logger

logger = setup_logger("train", config.LOG_FILE)


# Default dataset configuration template
DATASET_TEMPLATE = {
    "path": "datasets/pothole_data",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "Pothole"},
}


def create_dataset_yaml(output_path: str, dataset_root: str = None,
                        class_names: dict = None) -> str:
    """
    Create a YOLO dataset configuration YAML file.

    Args:
        output_path: Path where to write the YAML
        dataset_root: Root directory of the dataset
        class_names: Dictionary of {class_id: class_name}

    Returns:
        Path to created YAML file
    """
    cfg = DATASET_TEMPLATE.copy()
    if dataset_root:
        cfg["path"] = dataset_root
    if class_names:
        cfg["names"] = class_names

    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger.info(f"Dataset config created: {output_path}")
    return output_path


def train_model(data_yaml, model_name="yolov8s.pt", epochs=100,
                batch_size=16, img_size=640, device=None,
                project="runs/train", name="pothole_model",
                resume=False, patience=20):
    """
    Train a YOLOv8 model for pothole detection.

    Args:
        data_yaml: Path to dataset configuration YAML
        model_name: Base model to fine-tune (yolov8n/s/m/l/x.pt)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device ('cuda', 'cpu', or None for auto)
        project: Output project directory
        name: Run name
        resume: Resume from last checkpoint
        patience: Early stopping patience

    Returns:
        Training results
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "Ultralytics is required for training. "
            "Install with: pip install ultralytics"
        )

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    if device is None:
        device = config.DEFAULT_DEVICE

    logger.info("=" * 60)
    logger.info("STARTING YOLOv8 TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Base model:   {model_name}")
    logger.info(f"  Dataset:      {data_yaml}")
    logger.info(f"  Epochs:       {epochs}")
    logger.info(f"  Batch size:   {batch_size}")
    logger.info(f"  Image size:   {img_size}")
    logger.info(f"  Device:       {device}")
    logger.info(f"  Output:       {project}/{name}")

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        resume=resume,
        patience=patience,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        verbose=True,
    )

    best_model = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best_model):
        dest = config.YOLO_MODEL_PATH
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(best_model, dest)
        logger.info(f"Best model copied to: {dest}")

    logger.info("Training complete!")
    return results


def validate_model(model_path, data_yaml, device=None):
    """Validate a trained model on the validation set."""
    if device is None:
        device = config.DEFAULT_DEVICE

    logger.info(f"Validating model: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device)

    logger.info(f"  mAP@0.5:      {metrics.box.map50:.4f}")
    logger.info(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    logger.info(f"  Precision:    {metrics.box.mp:.4f}")
    logger.info(f"  Recall:       {metrics.box.mr:.4f}")

    return metrics


def export_model(model_path, export_format="onnx"):
    """
    Export a trained model to a deployment format.

    Args:
        model_path: Path to trained .pt model
        export_format: onnx, tflite, torchscript, coreml, etc.
    """
    logger.info(f"Exporting {model_path} to {export_format}...")
    model = YOLO(model_path)
    model.export(format=export_format)
    logger.info(f"Export complete: {export_format}")


def main():
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 model for pothole detection"
    )
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Base model (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu, or 0,1 for multi-GPU")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="pothole_model",
                        help="Run name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--validate", type=str, default=None,
                        help="Validate a model at this path (skip training)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export model to format (onnx, tflite, etc.)")
    parser.add_argument("--create-config", action="store_true",
                        help="Create a template dataset.yaml file")

    args = parser.parse_args()

    if args.create_config:
        path = create_dataset_yaml("dataset.yaml")
        print(f"\nDataset config template created: {path}")
        print("Edit it to point to your dataset, then run training.")
        return

    if args.export:
        export_model(args.export, "onnx")
        return

    if args.validate:
        validate_model(args.validate, args.data, args.device)
        return

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()