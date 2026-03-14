"""Two-step YOLOv8 training: ball-only pretraining, then fine-tune on both classes.

Step 1: Train on ball class only — forces model to learn tiny ball detection.
Step 2: Fine-tune on both player + ball — adds player detection without forgetting ball.

Key insight: explicit AdamW optimizer + lower LR in step 2 prevents overwriting ball weights.

Usage:
    python src/train/train_twostep.py \
        --ball-config configs/phase3_twostep_ball.yaml \
        --finetune-config configs/phase3_twostep_finetune.yaml \
        --name phase3_twostep
"""

import argparse
import time

import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_step(model: YOLO, config: dict, name: str, project: str) -> str:
    """Run one training step. Returns path to best weights."""
    epochs = config.get("epochs", 100)
    imgsz = config.get("imgsz", 1280)
    batch = config.get("batch", 4)

    # Build kwargs from config, skipping non-training keys
    skip_keys = {"model", "data", "epochs", "imgsz", "batch"}
    train_kwargs = {k: v for k, v in config.items() if k not in skip_keys}

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Data: {config['data']}")
    print(f"  Image size: {imgsz}, Batch: {batch}, Epochs: {epochs}")
    print(f"  Optimizer: {config.get('optimizer', 'auto')}, LR: {config.get('lr0', '?')}")
    if "classes" in config:
        print(f"  Classes: {config['classes']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    model.train(
        data=config["data"],
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=project,
        name=name,
        save=True,
        save_period=10,
        plots=True,
        workers=4,
        **train_kwargs,
    )

    elapsed_hours = (time.time() - start_time) / 3600
    best_path = f"{project}/{name}/weights/best.pt"
    print(f"\nStep complete: {name}")
    print(f"  GPU-hours: {elapsed_hours:.2f}")
    print(f"  Best weights: {best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-step YOLOv8 training")
    parser.add_argument("--ball-config", type=str, required=True, help="Config for ball-only step")
    parser.add_argument("--finetune-config", type=str, required=True, help="Config for fine-tune step")
    parser.add_argument("--name", type=str, default="phase3_twostep", help="Base experiment name")
    parser.add_argument("--data", type=str, default=None, help="Override data.yaml path")
    args = parser.parse_args()

    project = "outputs/models"

    # --- Step 1: Ball-only training ---
    ball_config = load_config(args.ball_config)
    if args.data:
        ball_config["data"] = args.data

    model_name = ball_config.get("model", "yolov8x.pt")
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    step1_name = f"{args.name}_step1_ball"
    best_ball_weights = train_step(model, ball_config, step1_name, project)

    # --- Step 2: Fine-tune on both classes ---
    finetune_config = load_config(args.finetune_config)
    if args.data:
        finetune_config["data"] = args.data

    print(f"\nLoading ball-only weights: {best_ball_weights}")
    model = YOLO(best_ball_weights)

    step2_name = f"{args.name}_step2_full"
    train_step(model, finetune_config, step2_name, project)

    print(f"\n{'='*60}")
    print("Two-step training complete!")
    print(f"  Final weights: {project}/{step2_name}/weights/best.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
