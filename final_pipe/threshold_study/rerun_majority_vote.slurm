#!/bin/bash
#SBATCH --job-name=vote_rerun
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --output=vr-%j.out
#SBATCH --error=vr-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Change these values for the dataset whose completed SC/LG/MA outputs
# should be compiled and fused.
DATASET="genz_normalized.csv"
NAME="genz_a_F1"
OUTDIR="pipeline_outputs/${NAME}"
LOGDIR="$OUTDIR/logs"

# Leave empty to choose the vote threshold by F1.
# Set to 2 for a fixed strict majority.
VOTE_THRESHOLD=""

mkdir -p "$LOGDIR"

sbatch --export=ALL,DATASET="$DATASET",NAME="$NAME",OUTDIR="$OUTDIR",VOTE_THRESHOLD="$VOTE_THRESHOLD" \
  --job-name="vote_${NAME}" \
  --output="$LOGDIR/${NAME}_vote-%j.out" \
  --error="$LOGDIR/${NAME}_vote-%j.err" \
  run_majority_vote.slurm
