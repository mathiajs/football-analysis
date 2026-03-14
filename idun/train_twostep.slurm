#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00             # 8 hours (20+20 epochs)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40g
#SBATCH --job-name=yolov8x-twostep
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

# Step 2: Two-step training (ball-only -> fine-tune both)
echo "Starting two-step training..."
python src/train/train_twostep.py \
    --ball-config configs/phase3_twostep_ball.yaml \
    --finetune-config configs/phase3_twostep_finetune.yaml \
    --name phase3_twostep \
    --data "$WORK_DATA/yolo_dataset/data.yaml"

echo "========================================="
echo "Job finished: $(date)"
echo "========================================="
