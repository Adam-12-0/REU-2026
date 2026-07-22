#!/bin/bash
#SBATCH --job-name=weighted_vote
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --output=wv-%j.out
#SBATCH --error=wv-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"

: "${DATASET:?DATASET was not set by rerun_weighted_vote.slurm}"
: "${NAME:?NAME was not set by rerun_weighted_vote.slurm}"
: "${OUTDIR:?OUTDIR was not set by rerun_weighted_vote.slurm}"

METHOD_OUTDIR="$OUTDIR/weighted_vote_outputs"
SC_INPUT="$OUTDIR/sc_outputs/sc_preds_all.csv"
LG_INPUT="$OUTDIR/lg_outputs/lg_preds_all.csv"
MA_INPUT="$OUTDIR/ma_outputs/ma_preds_all.csv"
COMPILED="$METHOD_OUTDIR/preds_compiled.csv"

mkdir -p "$METHOD_OUTDIR"

for input_path in "$DATASET" "$SC_INPUT" "$LG_INPUT" "$MA_INPUT"; do
  if [[ ! -f "$input_path" ]]; then
    echo "Required input does not exist: $input_path" >&2
    exit 1
  fi
done

echo "Started at $(date)"
echo "Dataset: $DATASET"
echo "Name: $NAME"
echo "Weighted-vote output directory: $METHOD_OUTDIR"

python3 -u compile.py \
  --dataset "$DATASET" \
  --sc-input "$SC_INPUT" \
  --lg-input "$LG_INPUT" \
  --ma-input "$MA_INPUT" \
  --output "$COMPILED"

weighted_args=(
  --input "$COMPILED"
  --all-output "$METHOD_OUTDIR/weighted_preds_all.csv"
  --single-output "$METHOD_OUTDIR/weighted_preds_sw.csv"
  --multi-output "$METHOD_OUTDIR/weighted_preds_mw.csv"
  --thresholds-output "$METHOD_OUTDIR/weighted_thr_metrics.csv"
  --weights-output "$METHOD_OUTDIR/weighted_weights.csv"
  --metrics-output "$METHOD_OUTDIR/weighted_metrics_all.csv"
)

# Leave WEIGHTED_THRESHOLD unset to select the threshold by F1.
if [[ -n "${WEIGHTED_THRESHOLD:-}" ]]; then
  weighted_args+=(--threshold "$WEIGHTED_THRESHOLD")
fi

python3 -u weighted_vote.py "${weighted_args[@]}"

echo "Finished at $(date)"
ls -lh "$METHOD_OUTDIR"
