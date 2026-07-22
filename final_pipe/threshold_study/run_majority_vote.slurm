#!/bin/bash
#SBATCH --job-name=vote
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --output=vote-%j.out
#SBATCH --error=vote-%j.err

set -euo pipefail

module load anaconda/anaconda-2024.10
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate dlcv

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"

: "${DATASET:?DATASET was not set by rerun_majority_vote.slurm}"
: "${NAME:?NAME was not set by rerun_majority_vote.slurm}"
: "${OUTDIR:?OUTDIR was not set by rerun_majority_vote.slurm}"

METHOD_OUTDIR="$OUTDIR/vote_outputs"
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
echo "Vote output directory: $METHOD_OUTDIR"
echo "SC input: $SC_INPUT"
echo "LG input: $LG_INPUT"
echo "MA input: $MA_INPUT"

python3 -u compile.py \
  --dataset "$DATASET" \
  --sc-input "$SC_INPUT" \
  --lg-input "$LG_INPUT" \
  --ma-input "$MA_INPUT" \
  --output "$COMPILED"

vote_args=(
  --input "$COMPILED"
  --all-output "$METHOD_OUTDIR/vote_preds_all.csv"
  --single-output "$METHOD_OUTDIR/vote_preds_sw.csv"
  --multi-output "$METHOD_OUTDIR/vote_preds_mw.csv"
  --thresholds-output "$METHOD_OUTDIR/vote_thr_metrics.csv"
  --metrics-output "$METHOD_OUTDIR/vote_metrics_all.csv"
)

# Leave VOTE_THRESHOLD unset to select 1, 2, or 3 by F1.
# Export VOTE_THRESHOLD=2 to require a strict majority of the three methods.
if [[ -n "${VOTE_THRESHOLD:-}" ]]; then
  vote_args+=(--vote-threshold "$VOTE_THRESHOLD")
fi

python3 -u majority_vote.py "${vote_args[@]}"

echo "Finished at $(date)"
ls -lh "$METHOD_OUTDIR"
