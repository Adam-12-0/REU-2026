#!/bin/bash
#SBATCH --job-name=driver
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=d-%j.out
#SBATCH --error=d-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DATASET="silent_signals_sample.csv"
NAME="silentsig_spacy"
SOURCE_LANGUAGE="en"
OUTDIR="pipeline_outputs/${NAME}"
LOGDIR="$OUTDIR/logs"

mkdir -p "$OUTDIR" "$LOGDIR"

EXPORTS=ALL,DATASET="$DATASET",NAME="$NAME",OUTDIR="$OUTDIR",SOURCE_LANGUAGE="$SOURCE_LANGUAGE"

mt_jid=$(sbatch --parsable \
  --export="$EXPORTS" \
  --job-name="mt_${NAME}" \
  --output="$LOGDIR/${NAME}_mt-%j.out" \
  --error="$LOGDIR/${NAME}_mt-%j.err" \
  run_mt.slurm)
echo "Submitted MT: $mt_jid"

