#!/bin/bash
#SBATCH --job-name=mt
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=evc26,evc44
#SBATCH --time=95:00:00
#SBATCH --mail-type=ALL
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
: "${SOURCE_IS_ENGLISH:?SOURCE_IS_ENGLISH was not set by driver}"

METHOD_OUTDIR="$OUTDIR/mt_outputs"

mkdir -p "$METHOD_OUTDIR"

if [[ "$SOURCE_IS_ENGLISH" == "1" || "$SOURCE_IS_ENGLISH" == "true" || "$SOURCE_IS_ENGLISH" == "TRUE" ]]; then
  MT_JUDGE_SCRIPT="mt_judge_en.py"
else
  MT_JUDGE_SCRIPT="mt_judge.py"
fi

echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "SOURCE_IS_ENGLISH: $SOURCE_IS_ENGLISH"
echo "MT judge script: $MT_JUDGE_SCRIPT"
echo "MT output directory: $METHOD_OUTDIR"

python3 -u "$MT_JUDGE_SCRIPT" \
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
