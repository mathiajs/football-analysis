"""Train YOLOv8 with configurable parameters from a YAML config file.

Usage:
    python src/train/train_improved.py --config configs/phase3_improved.yaml --name phase3_improved
"""

import argparse
import time

import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def train(config: dict, name: str) -> None:
    """Run training with the given configuration."""
    model_name = config.get("model", "yolov8m.pt")
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    # Extract training parameters from config
    data = config["data"]
    epochs = config.get("epochs", 100)
    imgsz = config.get("imgsz", 1280)
    batch = config.get("batch", 4)

    # Optimization
    optimizer = config.get("optimizer", "AdamW")
    lr0 = config.get("lr0", 0.001)
    lrf = config.get("lrf", 0.01)
    warmup_epochs = config.get("warmup_epochs", 3)
    weight_decay = config.get("weight_decay", 0.0005)

    # Augmentation
    mosaic = config.get("mosaic", 1.0)
    mixup = config.get("mixup", 0.0)
    scale = config.get("scale", 0.5)
    hsv_h = config.get("hsv_h", 0.015)
    hsv_s = config.get("hsv_s", 0.7)
    hsv_v = config.get("hsv_v", 0.4)
    degrees = config.get("degrees", 0.0)
    translate = config.get("translate", 0.1)
    fliplr = config.get("fliplr", 0.5)
    flipud = config.get("flipud", 0.0)

    # Training strategy
    close_mosaic = config.get("close_mosaic", 10)
    patience = config.get("patience", 20)

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Data: {data}")
    print(f"  Model: {model_name}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Epochs: {epochs}")
    print(f"  Optimizer: {optimizer}, LR: {lr0} -> {lr0 * lrf}")
    print(f"  Mosaic: {mosaic}, Scale: {scale}")
    print(f"  Patience: {patience}")
    print(f"{'='*60}\n")

    start_time = time.time()

    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        # Output
        project="outputs/models",
        name=name,
        # Augmentation
        mosaic=mosaic,
        mixup=mixup,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        fliplr=fliplr,
        flipud=flipud,
        # Optimization
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        # Training strategy
        close_mosaic=close_mosaic,
        patience=patience,
        # Resources
        workers=4,
        # Logging
        save=True,
        save_period=10,
        plots=True,
    )

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\nTraining complete: {name}")
    print(f"  GPU-hours: {elapsed_hours:.2f}")
    print(f"  Weights: outputs/models/{name}/weights/best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Experiment name (used for output directory)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override data.yaml path (useful for cluster training)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data:
        config["data"] = args.data
    train(config, args.name)


if __name__ == "__main__":
    main()
