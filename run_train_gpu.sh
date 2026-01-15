#!/bin/bash
#SBATCH --job-name=mlp-train
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

cd ~/projects/MLOps_2026

# Activeer jouw venv met CUDA-enabled torch
source ./venv/bin/activate

echo "=== GPU CHECK ==="
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda avail:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda)"
echo "================="

python src/ml_core/train.py
