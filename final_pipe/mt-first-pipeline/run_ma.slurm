#!/bin/bash
#SBATCH --job-name=ma
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclude=evc26,evc44 
#SBATCH --time=80:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=ma-%j.out
#SBATCH --error=ma-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"

: "${DATASET:?DATASET was not set by driver}"
: "${NAME:?NAME was not set by driver}"
: "${OUTDIR:?OUTDIR was not set by driver}"

METHOD_OUTDIR="$OUTDIR/ma_outputs"
MT_INPUT="$OUTDIR/mt_outputs/mt_preds_all.csv"
DEFINITIONS_TMP="$METHOD_OUTDIR/_ma_definitions_tmp.csv"
SCORES_TMP="$METHOD_OUTDIR/_ma_scores_tmp.csv"

mkdir -p "$METHOD_OUTDIR"

echo "Started at $(date)"
echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "MA output directory: $METHOD_OUTDIR"
echo "MT input: $MT_INPUT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
hostname
nvidia-smi

python3 -u -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python3 -u ma_agree.py \
  --dataset "$DATASET" \
  --output-dir "$METHOD_OUTDIR" \
  --mt-input "$MT_INPUT" \
  --definitions-output "$DEFINITIONS_TMP" \
  --batch-size 16 \
  --gpu-ids "0,0,0"

python3 -u ma_score.py \
  --definitions "$DEFINITIONS_TMP" \
  --output-dir "$METHOD_OUTDIR" \
  --scores-output "$SCORES_TMP" \
  --thresholds-output "$METHOD_OUTDIR/ma_thr_metrics.csv"

python3 -u ma_pred.py \
  --scores "$SCORES_TMP" \
  --thresholds "$METHOD_OUTDIR/ma_thr_metrics.csv" \
  --single-output "$METHOD_OUTDIR/ma_preds_sw.csv" \
  --multi-output "$METHOD_OUTDIR/ma_preds_mw.csv" \
  --all-output "$METHOD_OUTDIR/ma_preds_all.csv" \
  --metrics-output "$METHOD_OUTDIR/ma_metrics_all.csv"

rm -f "$DEFINITIONS_TMP" "$SCORES_TMP"

echo "Finished at $(date)"
ls -lh "$METHOD_OUTDIR"
