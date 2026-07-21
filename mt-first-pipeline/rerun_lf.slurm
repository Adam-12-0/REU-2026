#!/bin/bash
#SBATCH --job-name=lf_rerun
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --output=lfr-%j.out
#SBATCH --error=lfr-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DATASET="sdu_aaai22_ae_vietnamese_normalized.csv"
NAME="sdu_vietnamese"
OUTDIR="pipeline_outputs/${NAME}"
LOGDIR="$OUTDIR/logs"

mkdir -p "$LOGDIR"

sbatch --export=ALL,DATASET="$DATASET",NAME="$NAME",OUTDIR="$OUTDIR" \
  --job-name="lf_${NAME}" \
  --output="$LOGDIR/${NAME}_lf-%j.out" \
  --error="$LOGDIR/${NAME}_lf-%j.err" \
  run_lf.slurm
