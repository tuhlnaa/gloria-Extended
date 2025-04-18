#!/bin/bash
#SBATCH --job-name=Segmentation_training    # Job name
#SBATCH --output=logs/%x_%A_%a.out  # Standard output log (%A: job ID, %a: array index)
#SBATCH --error=logs/%x_%A_%a.err   # Standard error log
#SBATCH --time=99:00:00                # Time limit
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks per node
#SBATCH --cpus-per-task=40             # Number of CPU cores per task
#SBATCH --mem=64G                      # Memory per node
#SBATCH --array=0                      # Array job indices (modify based on number of input dirs)
#SBATCH --partition=cputest
#SBATCH --gres=gpu:1

# Initialize conda and activate your environment
source /home/tuhln930074/miniconda3/etc/profile.d/conda.sh
conda activate py311

echo "Running..."

python ./gloria-Extended/train.py --config /home/tuhln930074/python_project/gloria-Extended/configs/default_segmentation_optimization.yaml

echo "Completed."