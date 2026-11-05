#!/bin/bash
#SBATCH --job-name=vp
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=95:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=vp-%j.out
#SBATCH --error=vp-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate gptoss

# Limit CPU helper libraries so they do not exceed the cluster thread quota.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# ---------------------------------------------------------------------------
# Experiment settings: edit these values before submitting this file.
# MODEL_NAME is the short directory name; MODEL_ID is the Hugging Face ID.
# ---------------------------------------------------------------------------
INPUT_DATASET="slaying_input.csv"
KEY_DATASET="slaying_key.csv"
DATASET_NAME="slaying"

#MODEL_NAME="llama"
#MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Alternative models:
#MODEL_NAME="qwen"
#MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
#
#MODEL_NAME="mistral"
#MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
#
#MODEL_NAME="gemma"
#MODEL_ID="google/gemma-2-9b-it"

MODEL_NAME="gpt-oss"
MODEL_ID="openai/gpt-oss-120b"

OUTPUT_DIR="llm-outputs/$MODEL_NAME/$DATASET_NAME"
PREDICTIONS="$OUTPUT_DIR/prompt_output.csv"
METRICS="$OUTPUT_DIR/prompt_metrics.csv"

mkdir -p "$OUTPUT_DIR"

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Input dataset: $INPUT_DATASET"
echo "Key dataset: $KEY_DATASET"
echo "Model: $MODEL_ID"
echo "Output directory: $OUTPUT_DIR"
nvidia-smi

# Run inference using only sentence_id and sentence. This program never reads
# the key and always writes the required prompt_output.csv schema.
python3 -u vanilla-prompting.py \
  --dataset "$INPUT_DATASET" \
  --output "$PREDICTIONS" \
  --model-id "$MODEL_ID" \
  --batch-size 2 \

# Evaluate the completed predictions only after inference has finished.
python3 -u check_prompt_output.py \
  --predictions "$PREDICTIONS" \
  --key "$KEY_DATASET" \
  --output "$METRICS"

echo "Finished at: $(date)"
ls -lh "$OUTPUT_DIR"
