#!/bin/bash
#SBATCH --job-name=weighted_rerun
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --output=wvr-%j.out
#SBATCH --error=wvr-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DATASET="genz_normalized.csv"
NAME="genz_a_F1"
OUTDIR="pipeline_outputs/${NAME}"
LOGDIR="$OUTDIR/logs"

# Leave empty to select the weighted-score threshold by F1.
WEIGHTED_THRESHOLD=""

mkdir -p "$LOGDIR"

sbatch --export=ALL,DATASET="$DATASET",NAME="$NAME",OUTDIR="$OUTDIR",WEIGHTED_THRESHOLD="$WEIGHTED_THRESHOLD" \
  --job-name="weighted_${NAME}" \
  --output="$LOGDIR/${NAME}_weighted-%j.out" \
  --error="$LOGDIR/${NAME}_weighted-%j.err" \
  run_weighted_vote.slurm
