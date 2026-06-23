#!/bin/bash
#SBATCH --job-name=judge
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc33
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

cd "$SLURM_SUBMIT_DIR"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
hostname
nvidia-smi
python3 -u -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python3 -u judge.py