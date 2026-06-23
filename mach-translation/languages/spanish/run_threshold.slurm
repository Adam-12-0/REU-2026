#!/bin/bash
#SBATCH --job-name=d3_thr
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu

source activate
module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

cd "$SLURM_SUBMIT_DIR"

python3 -u threshold.py \
  --input-csv detect3_outputs/detect3_all_word_scores.csv \
  --output-dir detect3_outputs/threshold_analysis 