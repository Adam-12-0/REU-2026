#!/bin/bash
#SBATCH --job-name=sc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=sc-%j.out
#SBATCH --error=sc-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"

: "${DATASET:?DATASET was not set by driver}"
: "${NAME:?NAME was not set by driver}"
: "${OUTDIR:?OUTDIR was not set by driver}"

METHOD_OUTDIR="$OUTDIR/sc_outputs"
MT_INPUT="$OUTDIR/mt_outputs/mt_preds_all.csv"

mkdir -p "$METHOD_OUTDIR"
echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "SC output directory: $METHOD_OUTDIR"
echo "MT input: $MT_INPUT"

python3 -u sc.py \
  --dataset "$DATASET" \
  --output-dir "$METHOD_OUTDIR" \
  --mt-input "$MT_INPUT" \
  --scoring-output "$METHOD_OUTDIR/sc_scoring_breakdown.csv" \
  --thresholds-output "$METHOD_OUTDIR/sc_thr_metrics.csv" \
  --single-output "$METHOD_OUTDIR/sc_preds_sw.csv" \
  --multi-output "$METHOD_OUTDIR/sc_preds_mw.csv" \
  --all-output "$METHOD_OUTDIR/sc_preds_all.csv" \
  --metrics-output "$METHOD_OUTDIR/sc_metrics_all.csv"