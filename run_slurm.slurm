#!/bin/bash
#SBATCH --job-name=lidar-arch
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Load modules (Adjust based on DGX environment)
# module load anaconda/2023.09
# module load cuda/12.1

# Source environment
source /home/nc225mj/lidar-archaeology-segmentation/.venv/bin/activate

export DATA_ROOT="/home/nc225mj/lidar-archaeology-segmentation/data/processed"
export OUTPUT_DIR="./outputs"
export LOG_DIR="./logs"

echo "Using DATA_ROOT: $DATA_ROOT"

# Run normal training with default configs
python src/train.py dataset=dem model.backbone=resnet50 ++training.epochs=20
#python src/evaluation.py +checkpoint_path=./best_model.pth
#python src/inference.py +checkpoint_path=./best_model.pth
