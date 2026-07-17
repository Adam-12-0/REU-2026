#!/bin/bash
#SBATCH --job-name=mt
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=evc26
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=mt-%j.out
#SBATCH --error=mt-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "$SLURM_SUBMIT_DIR"
: "${DATASET:?DATASET was not set by driver}"
: "${NAME:?NAME was not set by driver}"
: "${OUTDIR:?OUTDIR was not set by driver}"

METHOD_OUTDIR="$OUTDIR/mt_outputs"

mkdir -p "$METHOD_OUTDIR"
echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "MT output directory: $METHOD_OUTDIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

hostname
nvidia-smi

python3 -u -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python3 -u mt_judge.py \
  --dataset "$DATASET" \
  --output-dir "$METHOD_OUTDIR" \
  --batch-size 8
  
python3 -u mt_judge_predictions.py \
  --scores "$METHOD_OUTDIR/_mt_word_scores_tmp.csv" \
  --thresholds "$METHOD_OUTDIR/thr_metrics.csv" \
  --output "$METHOD_OUTDIR/mt_preds_sw.csv"
rm -f "$METHOD_OUTDIR/_mt_word_scores_tmp.csv"

python3 -u mt_consecutive.py \
  --input "$METHOD_OUTDIR/mt_preds_sw.csv" \
  --dataset "$DATASET" \
  --phrase-output "$METHOD_OUTDIR/mt_preds_mw.csv" \
  --all-output "$METHOD_OUTDIR/mt_preds_all.csv" \
  --metrics-output "$METHOD_OUTDIR/mt_metrics_all.csv"