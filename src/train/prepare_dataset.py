"""Prepare the Football2025 dataset for Ultralytics YOLO training.

Collects images and labels from multiple match directories into a single
Ultralytics-compatible structure with train/val splits by match.
Filters out the event_labels class (class 2), keeping only player (0) and ball (1).
"""

import argparse
import shutil
from pathlib import Path


# Match directories relative to data root, with their image/label subdirs
MATCH_SOURCES = {
    "RBK-AALESUND": {
        "images": "RBK-AALESUND/data/images/train",
        "labels": "RBK-AALESUND/labels/train",
    },
    "RBK-BODO-P1": {
        "images": "RBK-BODO/part1/RBK_BODO_PART1/data/images/train",
        "labels": "RBK-BODO/part1/RBK_BODO_PART1/labels/train",
    },
    "RBK-BODO-P2": {
        "images": "RBK-BODO/part2/RBK_BODO_PART2/data/images/train",
        "labels": "RBK-BODO/part2/RBK_BODO_PART2/labels/train",
    },
    "RBK-BODO-P3": {
        "images": "RBK-BODO/part3/RBK_BODO_PART3/data/images/train",
        "labels": "RBK-BODO/part3/RBK_BODO_PART3/labels/train",
    },
    "RBK-FREDRIKSTAD": {
        "images": "RBK-FREDRIKSTAD/data/images/train",
        "labels": "RBK-FREDRIKSTAD/labels/train",
    },
    "RBK-HamKam": {
        "images": "RBK-HamKam/data/images/train",
        "labels": "RBK-HamKam/labels/train",
    },
    "RBK-VIKING": {
        "images": "RBK-VIKING/data/images/train",
        "labels": "RBK-VIKING/labels/train",
    },
}

# Which matches go to validation (rest go to train)
VAL_MATCHES = ["RBK-VIKING"]

# Classes to keep (filter out class 2 = event_labels)
KEEP_CLASSES = {0, 1}


def filter_label_file(src: Path, dst: Path) -> None:
    """Copy a label file, removing lines with unwanted classes."""
    lines = src.read_text().strip().split("\n")
    filtered = []
    for line in lines:
        if not line.strip():
            continue
        class_id = int(line.split()[0])
        if class_id in KEEP_CLASSES:
            filtered.append(line)
    dst.write_text("\n".join(filtered) + "\n" if filtered else "")


def prepare_dataset(data_root: Path, output_root: Path) -> None:
    """Build the Ultralytics dataset structure."""
    for split in ["train", "val"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "filtered_events": 0}

    for match_name, paths in MATCH_SOURCES.items():
        split = "val" if match_name in VAL_MATCHES else "train"
        img_dir = data_root / paths["images"]
        lbl_dir = data_root / paths["labels"]

        if not img_dir.exists():
            print(f"WARNING: {img_dir} not found, skipping")
            continue

        images = sorted(img_dir.glob("*.png"))
        print(f"  {match_name}: {len(images)} images -> {split}")

        for img_path in images:
            # Prefix filename with match name to avoid collisions
            new_name = f"{match_name}_{img_path.name}"
            lbl_name = f"{match_name}_{img_path.stem}.txt"

            # Copy image (symlink to save disk space)
            dst_img = output_root / "images" / split / new_name
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            # Filter and copy label
            src_lbl = lbl_dir / f"{img_path.stem}.txt"
            dst_lbl = output_root / "labels" / split / lbl_name
            if src_lbl.exists():
                filter_label_file(src_lbl, dst_lbl)
            else:
                # Empty label file if no annotations
                dst_lbl.write_text("")

            stats[split] += 1

    # Write data.yaml
    yaml_content = (
        f"path: {output_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: player\n"
        f"  1: ball\n"
    )
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(yaml_content)

    print(f"\nDataset prepared at: {output_root}")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  data.yaml: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Football2025 dataset for YOLO training")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing match folders (default: data/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/yolo_dataset"),
        help="Output directory for prepared dataset (default: data/yolo_dataset/)",
    )
    args = parser.parse_args()

    print("Preparing Football2025 dataset for YOLO training...")
    print(f"  Data root: {args.data_root}")
    print(f"  Output: {args.output}")
    print(f"  Val matches: {VAL_MATCHES}")
    print()

    prepare_dataset(args.data_root, args.output)


if __name__ == "__main__":
    main()
