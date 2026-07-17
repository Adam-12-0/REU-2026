#!/bin/bash
#SBATCH --job-name=lg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=lg-%j.out
#SBATCH --error=lg-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"

: "${DATASET:?DATASET was not set by driver}"
: "${NAME:?NAME was not set by driver}"
: "${OUTDIR:?OUTDIR was not set by driver}"

METHOD_OUTDIR="$OUTDIR/lg_outputs"
MT_INPUT="$OUTDIR/mt_outputs/mt_preds_all.csv"

mkdir -p "$METHOD_OUTDIR"

echo "Started at $(date)"
echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "LG output directory: $METHOD_OUTDIR"
echo "MT input: $MT_INPUT"

python3 -u lg.py \
  --dataset "$DATASET" \
  --output-dir "$METHOD_OUTDIR" \
  --mt-input "$MT_INPUT" \
  --features-output "$METHOD_OUTDIR/lg_scoring_breakdown.csv" \
  --thresholds-output "$METHOD_OUTDIR/lg_thr_metrics.csv"

echo "Finished lg.py at $(date)"

python3 -u lg_pred.py \
  --features "$METHOD_OUTDIR/lg_scoring_breakdown.csv" \
  --thresholds "$METHOD_OUTDIR/lg_thr_metrics.csv" \
  --single-output "$METHOD_OUTDIR/lg_preds_sw.csv" \
  --multi-output "$METHOD_OUTDIR/lg_preds_mw.csv" \
  --all-output "$METHOD_OUTDIR/lg_preds_all.csv" \
  --metrics-output "$METHOD_OUTDIR/lg_metrics_all.csv"

echo "Finished at $(date)"
ls -lh "$METHOD_OUTDIR"
