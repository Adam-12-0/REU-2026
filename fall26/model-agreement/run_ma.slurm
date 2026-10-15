#!/bin/bash
#SBATCH --job-name=ma
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=80:00:00
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --mail-type=ALL
#SBATCH --output=ma-%j.out
#SBATCH --error=ma-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

: "${DATASET:?DATASET was not set by driver}"
: "${KEY:?KEY was not set by driver}"
: "${NAME:?NAME was not set by driver}"
: "${OUTDIR:?OUTDIR was not set by driver}"
: "${MODEL_COMBO:?MODEL_COMBO was not set by driver}"

IFS=':' read -r -a MODELS <<< "$MODEL_COMBO"

if [[ ${#MODELS[@]} -lt 2 || ${#MODELS[@]} -gt 3 ]]; then
    echo "MODEL_COMBO must contain two or three colon-separated aliases." >&2
    exit 2
fi

MODEL_TAG="${MODEL_COMBO//:/-}"
METHOD_OUTDIR="$OUTDIR/ma_outputs/$MODEL_TAG"
DEFINITIONS_TMP="$METHOD_OUTDIR/_ma_definitions_tmp.csv"
SCORES_TMP="$METHOD_OUTDIR/_ma_word_scores_tmp.csv"
THRESHOLDS="$METHOD_OUTDIR/ma_thr_metrics.csv"
WORD_PREDS="$METHOD_OUTDIR/ma_preds_sw.csv"

mkdir -p "$METHOD_OUTDIR"

echo "Started at $(date)"
echo "Dataset: $DATASET"
echo "Models: ${MODELS[*]}"
echo "Output directory: $METHOD_OUTDIR"
hostname
nvidia-smi

python3 -u ma_define.py \
  --dataset "$DATASET" \
  --output "$DEFINITIONS_TMP" \
  --models "${MODELS[@]}" \
  --batch-size 16 

python3 -u ma_score.py \
  --input "$DEFINITIONS_TMP" \
  --output-scores "$SCORES_TMP" 

python3 -u ma_thresholds.py \
  --input "$SCORES_TMP" \
  --key "$KEY" \
  --output-thresholds "$THRESHOLDS" \
  --num-thresholds 200

python3 -u ma_pred.py \
  --scores "$SCORES_TMP" \
  --thresholds "$THRESHOLDS" \
  --output "$WORD_PREDS"

python3 -u ma_consecutive.py \
  --word-preds "$WORD_PREDS" \
  --key "$KEY" \
  --phrases "$METHOD_OUTDIR/ma_preds_mw.csv" \
  --combined "$METHOD_OUTDIR/ma_preds_all.csv" \
  --metrics "$METHOD_OUTDIR/ma_metrics.csv"

rm -f "$DEFINITIONS_TMP" "$SCORES_TMP"

echo "Finished at $(date)"
ls -lh "$METHOD_OUTDIR"
