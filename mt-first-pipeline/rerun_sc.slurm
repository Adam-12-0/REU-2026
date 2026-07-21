#!/bin/bash
#SBATCH --job-name=sc_rerun
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=sc-rerun-%j.out
#SBATCH --error=sc-rerun-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DATASET="sdu_aaai22_ae_persian_normalized.csv"
NAME="sdu_persian"
SOURCE_IS_ENGLISH="0"
OUTDIR="pipeline_outputs/${NAME}"
LOGDIR="$OUTDIR/logs"

mkdir -p "$LOGDIR"

EXPORTS=ALL,DATASET="$DATASET",NAME="$NAME",OUTDIR="$OUTDIR",SOURCE_IS_ENGLISH="$SOURCE_IS_ENGLISH"

sc_jid=$(
  sbatch --parsable \
    --export="$EXPORTS" \
    --job-name="sc_${NAME}" \
    --output="$LOGDIR/${NAME}_sc-rerun-%j.out" \
    --error="$LOGDIR/${NAME}_sc-rerun-%j.err" \
    run_sc.slurm
)

echo "Submitted SC rerun: $sc_jid"