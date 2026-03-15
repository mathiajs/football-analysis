#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-16:00:00            # 16 hours (1920px with batch=4 will be slower)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40g
#SBATCH --job-name=yolov8x-1920
#SBATCH --output=idun/slurm-%j.out

echo "========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="

module purge
module load Anaconda3/2023.09-0
conda activate /cluster/home/$USER/envs/yolo

# Step 1: Prepare dataset
WORK_DATA="/cluster/work/$USER/football_data"
DATASET_ROOT="/cluster/projects/vc/courses/TDT17/other/Football2025"

echo "Preparing dataset..."
python src/train/prepare_dataset.py \
    --data-root "$DATASET_ROOT" \
    --output "$WORK_DATA/yolo_dataset"

# Step 2: Train at 1920px
echo "Starting training at 1920px..."
python src/train/train_improved.py \
    --config configs/phase3_1920.yaml \
    --name phase3_yolov8x_1920 \
    --data "$WORK_DATA/yolo_dataset/data.yaml"

echo "========================================="
echo "Job finished: $(date)"
echo "========================================="
