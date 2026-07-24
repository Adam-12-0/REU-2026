#!/bin/bash
#SBATCH --job-name=lg_genz_all
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=lg-genz-all-%j.out
#SBATCH --error=lg-genz-all-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"

DATASET="genz_normalized.csv"
NAME="genz_all_candidates"
OUTDIR="pipeline_outputs/${NAME}"
METHOD_OUTDIR="$OUTDIR/lg_outputs"

mkdir -p "$METHOD_OUTDIR"

echo "Started at $(date)"
echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "LG output directory: $METHOD_OUTDIR"
echo "Candidate source: all words and consecutive phrases"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
hostname
nvidia-smi

python3 -u lg_all_candidates.py \
  --dataset "$DATASET" \
  --output-dir "$METHOD_OUTDIR" \
  --features-output "$METHOD_OUTDIR/lg_scoring_breakdown.csv" \
  --thresholds-output "$METHOD_OUTDIR/lg_thr_metrics.csv"

echo "Finished LG scoring at $(date)"

python3 -u lg_pred.py \
  --features "$METHOD_OUTDIR/lg_scoring_breakdown.csv" \
  --thresholds "$METHOD_OUTDIR/lg_thr_metrics.csv" \
  --single-output "$METHOD_OUTDIR/lg_preds_sw.csv" \
  --multi-output "$METHOD_OUTDIR/lg_preds_mw.csv" \
  --all-output "$METHOD_OUTDIR/lg_preds_all.csv" \
  --metrics-output "$METHOD_OUTDIR/lg_metrics_all.csv"

echo "Finished at $(date)"
ls -lh "$METHOD_OUTDIR"
