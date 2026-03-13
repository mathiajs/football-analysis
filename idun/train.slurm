#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi              # Change to your actual account
#SBATCH --time=0-08:00:00             # 8 hours (YOLOv8x at 1280 takes ~6-7h for 100 epochs)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40g           # Request A100 40GB (or use gpu80g for 80GB)
#SBATCH --job-name=yolov8x-football
#SBATCH --output=idun/slurm-%j.out

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

module purge
module load Anaconda3/2023.09-0
conda activate /cluster/home/$USER/envs/yolo

# Step 1: Prepare dataset (copies images + filters labels into YOLO format)
# Uses /cluster/work for fast I/O
WORK_DATA="/cluster/work/$USER/football_data"
DATASET_ROOT="/cluster/projects/vc/courses/TDT17/other/Football2025"

echo "Preparing dataset..."
python src/train/prepare_dataset.py \
    --data-root "$DATASET_ROOT" \
    --output "$WORK_DATA/yolo_dataset"

# Step 2: Train
echo "Starting training..."
python src/train/train_improved.py \
    --config configs/phase3_idun.yaml \
    --name phase3_yolov8x \
    --data "$WORK_DATA/yolo_dataset/data.yaml"

echo "========================================="
echo "Job finished: $(date)"
echo "========================================="
