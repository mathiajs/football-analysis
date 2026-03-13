"""
Football Dataset EDA — TDT4265 Mini-Project

Performs comprehensive exploratory data analysis on the Football2025 dataset.
Generates plots and statistics saved to outputs/eda/.

Usage:
    python src/eda/analyze_dataset.py --data-root data
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "player", 1: "ball", 2: "event"}
CLASS_COLORS_BGR = {
    0: (0, 255, 0),     # player — green
    1: (0, 0, 255),     # ball — red
    2: (255, 255, 0),   # event — cyan
}
CLASS_COLORS_RGB = {k: (b / 255, g / 255, r / 255) for k, (b, g, r) in CLASS_COLORS_BGR.items()}

OUTPUT_DIR = "outputs/eda"

# Consistent plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Annotation:
    """A single YOLO bounding-box annotation."""

    __slots__ = ("class_id", "xc", "yc", "w", "h")

    def __init__(self, class_id: int, xc: float, yc: float, w: float, h: float):
        self.class_id = class_id
        self.xc = xc
        self.yc = yc
        self.w = w
        self.h = h

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) pixel coordinates."""
        x1 = int((self.xc - self.w / 2) * img_w)
        y1 = int((self.yc - self.h / 2) * img_h)
        x2 = int((self.xc + self.w / 2) * img_w)
        y2 = int((self.yc + self.h / 2) * img_h)
        return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_matches(data_root: Path) -> list[dict]:
    """Discover all match directories and their image/label paths.

    Returns a list of dicts with keys: name, images_dir, labels_dir.
    """
    matches: list[dict] = []

    for match_dir in sorted(data_root.iterdir()):
        if not match_dir.is_dir():
            continue

        # Handle RBK-BODO's multi-part structure
        if match_dir.name == "RBK-BODO":
            for part_dir in sorted(match_dir.iterdir()):
                if not part_dir.is_dir():
                    continue
                for sub_dir in sorted(part_dir.iterdir()):
                    if not sub_dir.is_dir():
                        continue
                    images_dir = sub_dir / "data" / "images" / "train"
                    labels_dir = sub_dir / "labels" / "train"
                    if images_dir.exists() and labels_dir.exists():
                        matches.append({
                            "name": f"RBK-BODO/{part_dir.name}",
                            "images_dir": images_dir,
                            "labels_dir": labels_dir,
                        })
        else:
            images_dir = match_dir / "data" / "images" / "train"
            labels_dir = match_dir / "labels" / "train"
            if images_dir.exists() and labels_dir.exists():
                matches.append({
                    "name": match_dir.name,
                    "images_dir": images_dir,
                    "labels_dir": labels_dir,
                })

    return matches


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_label_file(label_path: Path) -> list[Annotation]:
    """Parse a YOLO-format label file into a list of Annotations."""
    annotations: list[Annotation] = []
    text = label_path.read_text().strip()
    if not text:
        return annotations

    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        # Clamp to [0, 1]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        annotations.append(Annotation(class_id, xc, yc, w, h))

    return annotations


def load_all_annotations(matches: list[dict]) -> dict:
    """Load all annotations from all matches.

    Returns a dict with:
        - 'per_match': {match_name: {image_stem: [Annotation, ...]}}
        - 'all': [Annotation, ...]  (flat list)
        - 'per_image_counts': {match_name: [int, ...]}
        - 'class_counts': {class_id: int}
        - 'per_match_class_counts': {match_name: {class_id: int}}
    """
    all_anns: list[Annotation] = []
    per_match: dict[str, dict[str, list[Annotation]]] = {}
    per_image_counts: dict[str, list[int]] = {}
    class_counts: dict[int, int] = defaultdict(int)
    per_match_class_counts: dict[str, dict[int, int]] = {}

    for match in matches:
        match_name = match["name"]
        per_match[match_name] = {}
        per_image_counts[match_name] = []
        per_match_class_counts[match_name] = defaultdict(int)

        label_files = sorted(match["labels_dir"].glob("*.txt"))
        for lf in label_files:
            anns = parse_label_file(lf)
            per_match[match_name][lf.stem] = anns
            per_image_counts[match_name].append(len(anns))
            all_anns.extend(anns)
            for a in anns:
                class_counts[a.class_id] += 1
                per_match_class_counts[match_name][a.class_id] += 1

    return {
        "per_match": per_match,
        "all": all_anns,
        "per_image_counts": per_image_counts,
        "class_counts": dict(class_counts),
        "per_match_class_counts": {k: dict(v) for k, v in per_match_class_counts.items()},
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_plot(fig: plt.Figure, name: str) -> None:
    """Save a figure with consistent settings."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png")


# ---------------------------------------------------------------------------
# Analysis 1: Basic statistics
# ---------------------------------------------------------------------------

def print_basic_stats(matches: list[dict], data: dict) -> dict:
    """Print and return basic dataset statistics."""
    stats: dict = {"matches": {}, "totals": {}}
    total_images = 0
    total_labels = 0
    total_anns = 0

    print("\n=== Basic Statistics ===")
    for match in matches:
        name = match["name"]
        n_images = len(list(match["images_dir"].glob("*.png")))
        n_labels = len(list(match["labels_dir"].glob("*.txt")))
        n_anns = sum(data["per_image_counts"][name])
        empty = sum(1 for c in data["per_image_counts"][name] if c == 0)
        total_images += n_images
        total_labels += n_labels
        total_anns += n_anns

        stats["matches"][name] = {
            "images": n_images,
            "labels": n_labels,
            "annotations": n_anns,
            "empty_labels": empty,
        }
        print(f"  {name:30s}  images={n_images:5d}  labels={n_labels:5d}  "
              f"annotations={n_anns:6d}  empty={empty:4d}")

    stats["totals"] = {
        "images": total_images,
        "labels": total_labels,
        "annotations": total_anns,
    }
    print(f"  {'TOTAL':30s}  images={total_images:5d}  labels={total_labels:5d}  "
          f"annotations={total_anns:6d}")

    return stats


# ---------------------------------------------------------------------------
# Analysis 2: Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(data: dict) -> None:
    """Bar chart of class distribution (overall and per match)."""
    # Overall
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = sorted(data["class_counts"].keys())
    counts = [data["class_counts"].get(c, 0) for c in classes]
    names = [CLASS_NAMES.get(c, f"class_{c}") for c in classes]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    bars = axes[0].bar(names, counts, color=[CLASS_COLORS_RGB.get(c, (0.5, 0.5, 0.5)) for c in classes],
                       edgecolor="black", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Overall Class Distribution")
    axes[0].set_ylabel("Count")

    # Per match
    match_names = sorted(data["per_match_class_counts"].keys())
    x = np.arange(len(match_names))
    width = 0.25
    for i, c in enumerate(classes):
        vals = [data["per_match_class_counts"][m].get(c, 0) for m in match_names]
        axes[1].bar(x + i * width, vals, width, label=CLASS_NAMES.get(c, f"class_{c}"),
                    color=CLASS_COLORS_RGB.get(c, (0.5, 0.5, 0.5)), edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x + width)
    short_names = [m.replace("RBK-BODO/", "BODO/") for m in match_names]
    axes[1].set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    axes[1].set_title("Class Distribution per Match")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "class_distribution")


# ---------------------------------------------------------------------------
# Analysis 3: Bounding box size analysis
# ---------------------------------------------------------------------------

def plot_bbox_sizes(data: dict) -> None:
    """Box/violin plots of bounding box dimensions per class."""
    IMG_W, IMG_H = 1920, 1080

    class_widths: dict[str, list[float]] = defaultdict(list)
    class_heights: dict[str, list[float]] = defaultdict(list)
    class_areas: dict[str, list[float]] = defaultdict(list)
    class_aspects: dict[str, list[float]] = defaultdict(list)

    for ann in data["all"]:
        name = CLASS_NAMES.get(ann.class_id, f"class_{ann.class_id}")
        w_px = ann.w * IMG_W
        h_px = ann.h * IMG_H
        class_widths[name].append(w_px)
        class_heights[name].append(h_px)
        class_areas[name].append(w_px * h_px)
        if h_px > 0:
            class_aspects[name].append(w_px / h_px)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Width
    plot_data = [(name, vals) for name, vals in sorted(class_widths.items())]
    axes[0, 0].boxplot([v for _, v in plot_data], tick_labels=[n for n, _ in plot_data], showfliers=False)
    axes[0, 0].set_title("Bounding Box Width (px)")
    axes[0, 0].set_ylabel("Pixels")

    # Height
    plot_data = [(name, vals) for name, vals in sorted(class_heights.items())]
    axes[0, 1].boxplot([v for _, v in plot_data], tick_labels=[n for n, _ in plot_data], showfliers=False)
    axes[0, 1].set_title("Bounding Box Height (px)")
    axes[0, 1].set_ylabel("Pixels")

    # Area (log scale)
    plot_data = [(name, vals) for name, vals in sorted(class_areas.items())]
    axes[1, 0].boxplot([v for _, v in plot_data], tick_labels=[n for n, _ in plot_data], showfliers=False)
    axes[1, 0].set_title("Bounding Box Area (px²)")
    axes[1, 0].set_ylabel("Pixels²")
    axes[1, 0].set_yscale("log")

    # Aspect ratio
    plot_data = [(name, vals) for name, vals in sorted(class_aspects.items())]
    axes[1, 1].boxplot([v for _, v in plot_data], tick_labels=[n for n, _ in plot_data], showfliers=False)
    axes[1, 1].set_title("Aspect Ratio (w/h)")
    axes[1, 1].set_ylabel("Ratio")
    axes[1, 1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Bounding Box Size Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "bbox_sizes")

    # Detailed histogram for ball vs player widths
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if "ball" in class_widths and "player" in class_widths:
        axes[0].hist(class_widths["ball"], bins=50, alpha=0.7, color=CLASS_COLORS_RGB[1], label="ball", edgecolor="black", linewidth=0.3)
        axes[0].set_title("Ball Width Distribution (px)")
        axes[0].set_xlabel("Width (px)")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        axes[1].hist(class_widths["player"], bins=50, alpha=0.7, color=CLASS_COLORS_RGB[0], label="player", edgecolor="black", linewidth=0.3)
        axes[1].set_title("Player Width Distribution (px)")
        axes[1].set_xlabel("Width (px)")
        axes[1].set_ylabel("Count")
        axes[1].legend()

    fig.suptitle("Width Comparison: Ball vs Player", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "bbox_width_comparison")


# ---------------------------------------------------------------------------
# Analysis 4: Objects per image
# ---------------------------------------------------------------------------

def plot_objects_per_image(data: dict) -> None:
    """Histogram of objects per image."""
    all_counts = []
    for match_counts in data["per_image_counts"].values():
        all_counts.extend(match_counts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(all_counts, bins=range(0, max(all_counts) + 2), alpha=0.7,
                 color="steelblue", edgecolor="black", linewidth=0.3)
    axes[0].set_title("Total Objects per Image")
    axes[0].set_xlabel("Number of objects")
    axes[0].set_ylabel("Number of images")
    axes[0].axvline(np.mean(all_counts), color="red", linestyle="--",
                    label=f"Mean: {np.mean(all_counts):.1f}")
    axes[0].axvline(np.median(all_counts), color="orange", linestyle="--",
                    label=f"Median: {np.median(all_counts):.0f}")
    axes[0].legend()

    # Per match box plot
    match_names = sorted(data["per_image_counts"].keys())
    box_data = [data["per_image_counts"][m] for m in match_names]
    short_names = [m.replace("RBK-BODO/", "BODO/") for m in match_names]
    axes[1].boxplot(box_data, tick_labels=short_names, showfliers=False)
    axes[1].set_title("Objects per Image by Match")
    axes[1].set_ylabel("Number of objects")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("Objects per Image", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "objects_per_image")


# ---------------------------------------------------------------------------
# Analysis 5: Spatial distribution heatmaps
# ---------------------------------------------------------------------------

def plot_spatial_heatmaps(data: dict) -> None:
    """2D heatmaps of bounding box center positions per class."""
    centers_by_class: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for ann in data["all"]:
        centers_by_class[ann.class_id].append((ann.xc, ann.yc))

    classes_to_plot = sorted(centers_by_class.keys())
    n = len(classes_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, cls_id in zip(axes, classes_to_plot):
        centers = centers_by_class[cls_id]
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        heatmap, xedges, yedges = np.histogram2d(
            ys, xs, bins=50, range=[[0, 1], [0, 1]]
        )
        ax.imshow(heatmap, extent=[0, 1, 1, 0], aspect="auto", cmap="hot", interpolation="gaussian")
        ax.set_title(f"{CLASS_NAMES.get(cls_id, f'class_{cls_id}')} (n={len(centers):,})")
        ax.set_xlabel("X (normalized)")
        ax.set_ylabel("Y (normalized)")

    fig.suptitle("Spatial Distribution of Object Centers", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "spatial_heatmaps")


# ---------------------------------------------------------------------------
# Analysis 6: Ball visibility
# ---------------------------------------------------------------------------

def plot_ball_visibility(data: dict) -> None:
    """Analyze how often the ball is annotated per match."""
    fig, ax = plt.subplots(figsize=(10, 5))

    match_names = sorted(data["per_match"].keys())
    ball_present = []
    ball_absent = []
    short_names = []

    for match_name in match_names:
        images = data["per_match"][match_name]
        total = len(images)
        has_ball = sum(1 for anns in images.values() if any(a.class_id == 1 for a in anns))
        ball_present.append(has_ball)
        ball_absent.append(total - has_ball)
        short_names.append(match_name.replace("RBK-BODO/", "BODO/"))

    x = np.arange(len(match_names))
    ax.bar(x, ball_present, label="Ball visible", color=CLASS_COLORS_RGB[1], edgecolor="black", linewidth=0.3)
    ax.bar(x, ball_absent, bottom=ball_present, label="Ball absent", color="lightgray", edgecolor="black", linewidth=0.3)

    for i, (bp, ba) in enumerate(zip(ball_present, ball_absent)):
        total = bp + ba
        pct = bp / total * 100 if total > 0 else 0
        ax.text(i, total + 5, f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Ball Visibility per Match")
    ax.legend()

    fig.tight_layout()
    save_plot(fig, "ball_visibility")


# ---------------------------------------------------------------------------
# Analysis 7: Temporal / sequence analysis
# ---------------------------------------------------------------------------

def plot_temporal_analysis(data: dict) -> None:
    """Analyze object counts over frame sequences."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Pick first non-BODO match for a clean sequence view
    match_names = sorted(data["per_match"].keys())
    main_match = match_names[0]
    images = data["per_match"][main_match]

    sorted_stems = sorted(images.keys())
    frame_nums = list(range(len(sorted_stems)))
    total_counts = [len(images[s]) for s in sorted_stems]
    ball_present = [1 if any(a.class_id == 1 for a in images[s]) else 0 for s in sorted_stems]

    axes[0].plot(frame_nums, total_counts, linewidth=0.5, alpha=0.8, color="steelblue")
    axes[0].set_title(f"Object Count over Frames — {main_match}")
    axes[0].set_ylabel("Objects per frame")
    axes[0].set_xlabel("Frame index")

    # Ball presence as a binary strip
    axes[1].fill_between(frame_nums, ball_present, step="mid", alpha=0.7, color=CLASS_COLORS_RGB[1])
    axes[1].set_title(f"Ball Presence over Frames — {main_match}")
    axes[1].set_ylabel("Ball present")
    axes[1].set_xlabel("Frame index")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["No", "Yes"])

    fig.suptitle("Temporal Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "temporal_analysis")


# ---------------------------------------------------------------------------
# Analysis 8: Sample visualizations
# ---------------------------------------------------------------------------

def plot_sample_images(matches: list[dict], data: dict, n_samples: int = 8) -> None:
    """Draw bounding boxes on sample images and save as a grid."""
    IMG_W, IMG_H = 1920, 1080

    # Collect image paths from the first match that has enough images
    sample_images: list[tuple[Path, list[Annotation]]] = []
    for match in matches:
        name = match["name"]
        if name not in data["per_match"]:
            continue
        images_dir = match["images_dir"]
        stems = sorted(data["per_match"][name].keys())
        # Pick evenly spaced samples
        step = max(1, len(stems) // n_samples)
        for stem in stems[::step][:n_samples]:
            img_path = images_dir / f"{stem}.png"
            if img_path.exists():
                sample_images.append((img_path, data["per_match"][name][stem]))
            if len(sample_images) >= n_samples:
                break
        if len(sample_images) >= n_samples:
            break

    if not sample_images:
        print("  WARNING: No sample images found.")
        return

    cols = 4
    rows = (len(sample_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, (img_path, anns) in enumerate(sample_images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for ann in anns:
            x1, y1, x2, y2 = ann.to_xyxy(IMG_W, IMG_H)
            color_bgr = CLASS_COLORS_BGR.get(ann.class_id, (128, 128, 128))
            # Convert BGR tuple to RGB for cv2 (which expects BGR, but we already converted)
            # Actually draw on the RGB image using RGB colors
            color_rgb_255 = (color_bgr[2], color_bgr[1], color_bgr[0])
            thickness = 1 if ann.class_id == 1 else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb_255, thickness)
            label = CLASS_NAMES.get(ann.class_id, str(ann.class_id))
            cv2.putText(img, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_rgb_255, 1)

        axes[i].imshow(img)
        axes[i].set_title(img_path.stem, fontsize=8)
        axes[i].axis("off")

    # Hide unused axes
    for j in range(len(sample_images), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Sample Images with Annotations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "sample_images")


# ---------------------------------------------------------------------------
# Summary stats JSON
# ---------------------------------------------------------------------------

def save_summary_stats(matches: list[dict], data: dict, basic_stats: dict) -> None:
    """Save a JSON summary of key dataset statistics."""
    IMG_W, IMG_H = 1920, 1080

    # Bbox size stats per class
    bbox_stats: dict = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        widths = [a.w * IMG_W for a in data["all"] if a.class_id == cls_id]
        heights = [a.h * IMG_H for a in data["all"] if a.class_id == cls_id]
        if widths:
            bbox_stats[cls_name] = {
                "count": len(widths),
                "width_mean": round(float(np.mean(widths)), 1),
                "width_median": round(float(np.median(widths)), 1),
                "width_std": round(float(np.std(widths)), 1),
                "height_mean": round(float(np.mean(heights)), 1),
                "height_median": round(float(np.median(heights)), 1),
                "height_std": round(float(np.std(heights)), 1),
            }

    # Objects per image stats
    all_counts = []
    for match_counts in data["per_image_counts"].values():
        all_counts.extend(match_counts)

    # Ball visibility
    total_images = 0
    images_with_ball = 0
    for match_name, images in data["per_match"].items():
        for stem, anns in images.items():
            total_images += 1
            if any(a.class_id == 1 for a in anns):
                images_with_ball += 1

    summary = {
        "dataset": {
            "num_matches": len(matches),
            "match_names": [m["name"] for m in matches],
            "image_resolution": f"{IMG_W}x{IMG_H}",
            **basic_stats["totals"],
        },
        "class_distribution": {
            CLASS_NAMES.get(k, f"class_{k}"): v
            for k, v in sorted(data["class_counts"].items())
        },
        "class_percentages": {
            CLASS_NAMES.get(k, f"class_{k}"): round(v / sum(data["class_counts"].values()) * 100, 2)
            for k, v in sorted(data["class_counts"].items())
        },
        "bbox_sizes": bbox_stats,
        "objects_per_image": {
            "mean": round(float(np.mean(all_counts)), 1),
            "median": float(np.median(all_counts)),
            "std": round(float(np.std(all_counts)), 1),
            "min": int(np.min(all_counts)),
            "max": int(np.max(all_counts)),
        },
        "ball_visibility": {
            "total_images": total_images,
            "images_with_ball": images_with_ball,
            "ball_visibility_pct": round(images_with_ball / total_images * 100, 1) if total_images > 0 else 0,
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved stats.json")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Football dataset EDA")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Path to dataset root directory")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: Data root '{data_root}' does not exist.")
        return

    print("Discovering matches...")
    matches = discover_matches(data_root)
    print(f"  Found {len(matches)} match(es): {[m['name'] for m in matches]}")

    print("Loading annotations...")
    data = load_all_annotations(matches)
    print(f"  Loaded {len(data['all']):,} annotations across "
          f"{sum(len(v) for v in data['per_match'].values()):,} images")

    basic_stats = print_basic_stats(matches, data)

    print("\nGenerating plots...")
    plot_class_distribution(data)
    plot_bbox_sizes(data)
    plot_objects_per_image(data)
    plot_spatial_heatmaps(data)
    plot_ball_visibility(data)
    plot_temporal_analysis(data)
    plot_sample_images(matches, data)

    print("\nSaving summary statistics...")
    summary = save_summary_stats(matches, data, basic_stats)

    print("\n=== Key Findings ===")
    print(f"  Total images: {summary['dataset']['images']:,}")
    print(f"  Total annotations: {summary['dataset']['annotations']:,}")
    print(f"  Class distribution: {summary['class_percentages']}")
    print(f"  Ball visibility: {summary['ball_visibility']['ball_visibility_pct']}% of images")
    if "ball" in summary["bbox_sizes"]:
        print(f"  Ball median size: {summary['bbox_sizes']['ball']['width_median']}x"
              f"{summary['bbox_sizes']['ball']['height_median']} px")
    if "player" in summary["bbox_sizes"]:
        print(f"  Player median size: {summary['bbox_sizes']['player']['width_median']}x"
              f"{summary['bbox_sizes']['player']['height_median']} px")
    print(f"  Objects per image: mean={summary['objects_per_image']['mean']}, "
          f"range=[{summary['objects_per_image']['min']}, {summary['objects_per_image']['max']}]")
    print("\nAll outputs saved to outputs/eda/")


if __name__ == "__main__":
    main()
