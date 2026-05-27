#!/bin/bash
#SBATCH --job-name=lidar-arch
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclusive

# Load modules (Adjust based on DGX environment)
# module load anaconda/2023.09
# module load cuda/12.1

# Source environment
source /home/nc225mj/lidar-archaeology-segmentation/.venv/bin/activate

echo "JOB_ID: $SLURM_JOB_ID"
echo "NODE: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export DATA_ROOT="/home/nc225mj/lidar-archaeology-segmentation/data/processed"
export OUTPUT_DIR="./outputs"
export LOG_DIR="./logs"

echo "Using DATA_ROOT: $DATA_ROOT"

# Run normal training with default configs
srun python -u src/train.py
#srun python -u src/evaluation.py +checkpoint_path=/home/nc225mj/lidar-archaeology-segmentation/outputs/2026-04-29/14-45-43/best_model.pth
#HYDRA_FULL_ERROR=1 srun python -u src/inference.py +checkpoint_path=/home/nc225mj/lidar-archaeology-segmentation/outputs/2026-04-29/14-45-43/best_model.pth
