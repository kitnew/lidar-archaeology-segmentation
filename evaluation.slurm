#!/bin/bash
#SBATCH --job-name=deeplabv3
#SBATCH --output=output.log
#SBATCH --error=errors.log
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:10
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00

# Source from virtual environment
source /home/nc225mj/lidar-archaeology-segmentation/.venv/bin/activate

# Run script
python -m src.evaluation --model-path ./2025-10-07_16-29-09_model.pth --mode slurm --tile-size 64 --stride 16
