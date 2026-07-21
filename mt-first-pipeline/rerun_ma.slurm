#!/bin/bash
#SBATCH --job-name=ma_rerun
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclude=evc26,evc44
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=ma-rerun-%j.out
#SBATCH --error=ma-rerun-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DATASET="sdu_aaai22_ae_vietnamese_normalized.csv"
NAME="sdu_vietnamese"
OUTDIR="pipeline_outputs/${NAME}"

sbatch --export=ALL,DATASET="$DATASET",NAME="$NAME",OUTDIR="$OUTDIR" \
  --job-name="ma_${NAME}" \
  --output="$OUTDIR/logs/${NAME}_ma-%j.out" \
  --error="$OUTDIR/logs/${NAME}_ma-%j.err" \
  run_ma.slurm