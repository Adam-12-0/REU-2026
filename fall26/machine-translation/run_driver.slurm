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
#
# DATASET:
#   sentence_id, sentence
#
# KEY:
#   sentence_id, sentence, special_term, term_type,
#   term_start_id, term_end_id
# ---------------------------------------------------------------------------

DATASET="slangq_input.csv"
KEY="slangq_key.csv"
NAME="slangq"

MODEL="gemma"
SOURCE_LANGUAGE="en"

# Allowed aliases: spanish, arabic, chinese, japanese
# Separate multiple aliases with colons.
TRANSLATION_COMBO="chinese"
TRANSLATION_TAG="${TRANSLATION_COMBO//:/-}"


OUTDIR="$MODEL/$NAME"
LOGDIR="$OUTDIR/logs/$TRANSLATION_TAG"

mkdir -p "$OUTDIR" "$LOGDIR"

EXPORTS="ALL,DATASET=$DATASET,KEY=$KEY,NAME=$NAME"
EXPORTS="$EXPORTS,OUTDIR=$OUTDIR,SOURCE_LANGUAGE=$SOURCE_LANGUAGE"
EXPORTS="$EXPORTS,TRANSLATION_COMBO=$TRANSLATION_COMBO"
EXPORTS="$EXPORTS,MODEL=$MODEL"

mt_jid=$(sbatch --parsable \
  --export="$EXPORTS" \
  --job-name="mt_${NAME}_${MODEL}" \
  --output="$LOGDIR/${NAME}_${MODEL}_mt-%j.out" \
  --error="$LOGDIR/${NAME}_${MODEL}_mt-%j.err" \
  run_mt.slurm)

echo "Submitted MT job: $mt_jid"
echo "Input dataset: $DATASET"
echo "Key dataset: $KEY"
echo "Dataset name: $NAME"
echo "Model: $MODEL"
echo "Source language: $SOURCE_LANGUAGE"
echo "Translations: $TRANSLATION_COMBO"
echo "Output root: $OUTDIR"
echo "Logs: $LOGDIR"