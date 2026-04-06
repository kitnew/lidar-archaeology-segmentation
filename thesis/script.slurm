#!/bin/bash
#SBATCH --job-name=script
#SBATCH --output=script.log
#SBATCH --error=script.log
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:10
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00

# Source from virtual environment
source /home/nc225mj/lidar-archaeology-segmentation/.venv/bin/activate

# Run script
python -m notebooks.scripts.from_tiff_to_npz
