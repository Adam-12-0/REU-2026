#!/bin/bash
#SBATCH --job-name=mt_driver
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=d-%j.out
#SBATCH --error=d-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

DATASET="slaying_input.csv"
KEY="slaying_key.csv"
NAME="slaying"

# MODEL1 performs translation.
# MODEL2 evaluates candidate preservation.
MODEL1="llama"
MODEL2="gemma"

SOURCE_LANGUAGE="en"

# Allowed aliases: spanish, arabic, chinese, japanese
# Separate multiple aliases with colons.
TRANSLATION_COMBO="spanish:arabic:chinese:japanese"
TRANSLATION_TAG="${TRANSLATION_COMBO//:/-}"

# Include both models in the output path so different experiments do not
# overwrite one another.
MODEL_TAG="${MODEL1}-translate_${MODEL2}-judge"
OUTDIR="$MODEL_TAG/$NAME/v3"
LOGDIR="$OUTDIR/logs/$TRANSLATION_TAG"

mkdir -p "$OUTDIR" "$LOGDIR"

EXPORTS="ALL,DATASET=$DATASET,KEY=$KEY,NAME=$NAME"
EXPORTS="$EXPORTS,OUTDIR=$OUTDIR,SOURCE_LANGUAGE=$SOURCE_LANGUAGE"
EXPORTS="$EXPORTS,TRANSLATION_COMBO=$TRANSLATION_COMBO"
EXPORTS="$EXPORTS,MODEL1=$MODEL1,MODEL2=$MODEL2"

mt_jid=$(sbatch --parsable \
  --export="$EXPORTS" \
  --job-name="mt_${NAME}_${MODEL1}_${MODEL2}" \
  --output="$LOGDIR/${NAME}_${MODEL1}_${MODEL2}_mt-%j.out" \
  --error="$LOGDIR/${NAME}_${MODEL1}_${MODEL2}_mt-%j.err" \
  run_mt.slurm)

echo "Submitted MT job: $mt_jid"
echo "Input dataset: $DATASET"
echo "Key dataset: $KEY"
echo "Dataset name: $NAME"
echo "Translation model: $MODEL1"
echo "Judge model: $MODEL2"
echo "Source language: $SOURCE_LANGUAGE"
echo "Translations: $TRANSLATION_COMBO"
echo "Output root: $OUTDIR"
echo "Logs: $LOGDIR"