#!/bin/bash
#SBATCH --job-name=mt
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=evc11
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

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# Values supplied by the driver.
: "${DATASET:?DATASET was not set by driver}"
: "${KEY:?KEY was not set by driver}"
: "${NAME:?NAME was not set by driver}"
: "${OUTDIR:?OUTDIR was not set by driver}"
: "${SOURCE_LANGUAGE:?SOURCE_LANGUAGE was not set by driver}"
: "${TRANSLATION_COMBO:?TRANSLATION_COMBO was not set by driver}"
: "${MODEL1:?MODEL1 was not set by driver}"
: "${MODEL2:?MODEL2 was not set by driver}"

# Convert spanish:arabic:chinese:japanese into a Bash array.
IFS=':' read -r -a TRANSLATIONS <<< "$TRANSLATION_COMBO"

if [[ ${#TRANSLATIONS[@]} -lt 1 || ${#TRANSLATIONS[@]} -gt 4 ]]; then
    echo "TRANSLATION_COMBO must contain between 1 and 4 languages." >&2
    exit 2
fi

# Normalized vote scores have one more possible value than the number
# of languages. Four languages produce 0, .25, .50, .75, and 1.
NUM_THRESHOLDS=$((${#TRANSLATIONS[@]} + 1))

TRANSLATION_TAG="${TRANSLATION_COMBO//:/-}"
METHOD_OUTDIR="$OUTDIR/$TRANSLATION_TAG/mt_outputs/"

TRANSLATION_OUTPUT="$METHOD_OUTDIR/mt_translations.csv"
CANDIDATE_SCORES="$METHOD_OUTDIR/_mt_candidates_tmp.csv"
THRESHOLDS="$METHOD_OUTDIR/mt_thr_metrics.csv"
WORD_PREDS="$METHOD_OUTDIR/mt_preds_sw.csv"
PHRASE_PREDS="$METHOD_OUTDIR/mt_preds_mw.csv"
COMBINED_PREDS="$METHOD_OUTDIR/mt_preds_all.csv"
METRICS="$METHOD_OUTDIR/mt_metrics.csv"

mkdir -p "$METHOD_OUTDIR"

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Dataset: $DATASET"
echo "Key: $KEY"
echo "Name: $NAME"
echo "Source language: $SOURCE_LANGUAGE"
echo "Translation languages: ${TRANSLATIONS[*]}"
echo "Model1: $MODEL1"
echo "Model2: $MODEL2"
echo "Number of vote thresholds: $NUM_THRESHOLDS"
echo "Output directory: $METHOD_OUTDIR"

nvidia-smi

# ---------------------------------------------------------------------------
# 1. Perform forward and backward translations.
# ---------------------------------------------------------------------------
python3 -u mt_translate.py \
  --dataset "$DATASET" \
  --input-lang "$SOURCE_LANGUAGE" \
  --output "$TRANSLATION_OUTPUT" \
  --model "$MODEL1" \
  --translations "${TRANSLATIONS[@]}" \
  --batch-size 8

# ---------------------------------------------------------------------------
# 2A. Tokenize original sentences and calculate per-language fuzzy scores,
#    binary votes, vote totals, and normalized vote scores.
# ---------------------------------------------------------------------------
#python3 -u mt_score_cands.py \
#  --translations "$TRANSLATION_OUTPUT" \
#  --output "$CANDIDATE_SCORES" \
#  --trans-langs "${TRANSLATIONS[@]}" \
#  --fuzzy-threshold 93 \
#  --batch-size 256
  
  
# ---------------------------------------------------------------------------
# 2B. Tokenize original sentences and calculate per-language
#    binary votes, vote totals, and normalized vote scores.
# ---------------------------------------------------------------------------
#python3 -u mt_score_cands_2.py \
#  --translations "$TRANSLATION_OUTPUT" \
#  --output "$CANDIDATE_SCORES" \
#  --trans-langs "${TRANSLATIONS[@]}" \
#  --model "$MODEL2" \
#  --batch-size 4
  
# ---------------------------------------------------------------------------
# 2C. Per sentence, prompt an LLM to identify lost or mistranslated terms.
# ---------------------------------------------------------------------------
python3 -u mt_score_cands_3.py \
  --translations "$TRANSLATION_OUTPUT" \
  --output "$CANDIDATE_SCORES" \
  --trans-langs "${TRANSLATIONS[@]}" \
  --model "$MODEL2" \
  --batch-size 4

# ---------------------------------------------------------------------------
# 3. Evaluate normalized vote-score thresholds and select the best threshold.
# ---------------------------------------------------------------------------
python3 -u mt_thresholds.py \
  --input "$CANDIDATE_SCORES" \
  --key "$KEY" \
  --output-thresholds "$THRESHOLDS" \
  --num-thresholds "$NUM_THRESHOLDS"

# ---------------------------------------------------------------------------
# 4. Apply the selected threshold to every single-word occurrence.
# ---------------------------------------------------------------------------
python3 -u mt_sw.py \
  --scores "$CANDIDATE_SCORES" \
  --thresholds "$THRESHOLDS" \
  --output "$WORD_PREDS"

# ---------------------------------------------------------------------------
# 5. Construct multiword candidates from consecutive positive words and
#    calculate single-word, multiword, and combined metrics.
# ---------------------------------------------------------------------------
python3 -u mt_consecutive.py \
  --word-preds "$WORD_PREDS" \
  --key "$KEY" \
  --phrases "$PHRASE_PREDS" \
  --combined "$COMBINED_PREDS" \
  --metrics "$METRICS"

# The candidate scores also appear in mt_preds_sw.csv, so the temporary
# candidate file is no longer required.
rm -f "$CANDIDATE_SCORES"

echo "Finished at: $(date)"
ls -lh "$METHOD_OUTDIR"