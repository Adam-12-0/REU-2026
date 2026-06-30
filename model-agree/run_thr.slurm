#!/bin/bash
#SBATCH --job-name=agree-threshold
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=threshold-%j.out
#SBATCH --error=threshold-%j.err

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

cd "$SLURM_SUBMIT_DIR"

hostname
python3 -u -c "import pandas, matplotlib, sklearn; print('threshold deps ok')"

python3 -u threshold.py