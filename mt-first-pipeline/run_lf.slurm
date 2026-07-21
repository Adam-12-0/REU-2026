#!/bin/bash
#SBATCH --job-name=lf
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=lf-%j.out
#SBATCH --error=lf-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"

: "${DATASET:?DATASET was not set by driver}"
: "${NAME:?NAME was not set by driver}"
: "${OUTDIR:?OUTDIR was not set by driver}"

METHOD_OUTDIR="$OUTDIR/lf_outputs"
SC_INPUT="$OUTDIR/sc_outputs/sc_preds_all.csv"
LG_INPUT="$OUTDIR/lg_outputs/lg_preds_all.csv"
MA_INPUT="$OUTDIR/ma_outputs/ma_preds_all.csv"
COMPILED="$METHOD_OUTDIR/preds_compiled.csv"

mkdir -p "$METHOD_OUTDIR"

echo "Started at $(date)"
echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "LF output directory: $METHOD_OUTDIR"
echo "SC input: $SC_INPUT"
echo "LG input: $LG_INPUT"
echo "MA input: $MA_INPUT"

python3 -u compile.py \
  --dataset "$DATASET" \
  --sc-input "$SC_INPUT" \
  --lg-input "$LG_INPUT" \
  --ma-input "$MA_INPUT" \
  --output "$COMPILED"

python3 -u lf.py \
  --input "$COMPILED" \
  --all-output "$METHOD_OUTDIR/lf_preds_all.csv" \
  --single-output "$METHOD_OUTDIR/lf_preds_sw.csv" \
  --multi-output "$METHOD_OUTDIR/lf_preds_mw.csv" \
  --thresholds-output "$METHOD_OUTDIR/lf_thr_metrics.csv" \
  --weights-output "$METHOD_OUTDIR/lf_weights.csv" \
  --metrics-output "$METHOD_OUTDIR/lf_metrics_all.csv"

echo "Finished at $(date)"
ls -lh "$METHOD_OUTDIR"
