#!/bin/bash
#SBATCH --job-name=ma_driver
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=d-%j.out
#SBATCH --error=d-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Edit these experiment settings. MODEL_COMBO must contain three distinct names
# chosen from llama, qwen, mistral, and gemma. Colons are used because Slurm
# reserves commas as separators inside --export.
DATASET="medslang_input.csv"
KEY="medslang_key.csv"
NAME="medslang"
MODEL_COMBO="llama:gemma"
OUTDIR="pipeline_outputs/lg/${NAME}"
START_SENTENCE="0"
LIMIT_SENTENCES=""
LOGDIR="$OUTDIR/logs"

mkdir -p "$OUTDIR" "$LOGDIR"

EXPORTS="ALL,DATASET=$DATASET,KEY=$KEY,NAME=$NAME,OUTDIR=$OUTDIR,MODEL_COMBO=$MODEL_COMBO,START_SENTENCE=$START_SENTENCE,LIMIT_SENTENCES=$LIMIT_SENTENCES"

ma_jid=$(sbatch --parsable \
  --export="$EXPORTS" \
  --job-name="ma_${NAME}" \
  --output="$LOGDIR/${NAME}_ma-%j.out" \
  --error="$LOGDIR/${NAME}_ma-%j.err" \
  run_ma.slurm)

echo "Submitted Model Agreement: $ma_jid"
echo "Models: $MODEL_COMBO"
