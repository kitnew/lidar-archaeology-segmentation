#!/bin/bash
#SBATCH --job-name=deeplabv3_training
#SBATCH --output=training.log
#SBATCH --error=training_errors.log
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=shard:10
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00

# Source from virtual environment
source /home/nc225mj/lidar-archaeology-segmentation/.venv/bin/activate

# Run script
python -m src.training -bone resnet101 -t 128 -s 32 -b 16 -w 32 -e 400 -l 1e-4 -pw 10.62 -bw 0.3 -dw 0.7 --reduction mean
