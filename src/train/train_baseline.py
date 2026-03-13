"""Train a baseline YOLOv8s model on the Football2025 dataset.

Usage:
    python src/train/train_baseline.py --data data/yolo_dataset/data.yaml
"""

import argparse
import time

from ultralytics import YOLO


def train_baseline(
    data_yaml: str,
    imgsz: int = 640,
    epochs: int = 100,
    batch: int = 16,
    patience: int = 20,
    name: str = "baseline_yolov8s",
) -> None:
    """Train a baseline YOLOv8s model."""
    model = YOLO("yolov8s.pt")

    start_time = time.time()

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,

        # Output
        project="outputs/models",
        name=name,

        # Augmentation (sensible defaults for football)
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,

        # Optimization
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,

        # Training strategy
        close_mosaic=10,
        patience=patience,

        # Logging
        save=True,
        save_period=10,
        plots=True,
    )

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\nTraining complete in {elapsed_hours:.2f} GPU-hours")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline YOLOv8s on Football2025")
    parser.add_argument(
        "--data",
        type=str,
        default="data/yolo_dataset/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0=disabled)")
    parser.add_argument("--name", type=str, default="baseline_yolov8s", help="Experiment name")
    args = parser.parse_args()

    train_baseline(args.data, args.imgsz, args.epochs, args.batch, args.patience, args.name)


if __name__ == "__main__":
    main()
