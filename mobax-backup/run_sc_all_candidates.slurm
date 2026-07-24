#!/bin/bash
#SBATCH --job-name=sc_genz_all
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=sc-genz-all-%j.out
#SBATCH --error=sc-genz-all-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"

DATASET="genz_normalized.csv"
NAME="genz_all_candidates"
OUTDIR="pipeline_outputs/${NAME}"
METHOD_OUTDIR="$OUTDIR/sc_outputs"

mkdir -p "$METHOD_OUTDIR"

echo "Started at $(date)"
echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "SC output directory: $METHOD_OUTDIR"
echo "Candidate source: all words and consecutive phrases"

python3 -u sc_all_candidates.py \
  --dataset "$DATASET" \
  --output-dir "$METHOD_OUTDIR" \
  --scoring-output "$METHOD_OUTDIR/sc_scoring_breakdown.csv" \
  --thresholds-output "$METHOD_OUTDIR/sc_thr_metrics.csv" \
  --single-output "$METHOD_OUTDIR/sc_preds_sw.csv" \
  --multi-output "$METHOD_OUTDIR/sc_preds_mw.csv" \
  --all-output "$METHOD_OUTDIR/sc_preds_all.csv" \
  --metrics-output "$METHOD_OUTDIR/sc_metrics_all.csv"

echo "Finished at $(date)"
ls -lh "$METHOD_OUTDIR"
