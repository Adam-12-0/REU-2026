#!/bin/bash
#SBATCH --job-name=driver
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=d-%j.out
#SBATCH --error=d-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

DATASET="genz_b_normalized.csv"
NAME="genz_b"
SOURCE_IS_ENGLISH="1"
OUTDIR="pipeline_outputs/${NAME}"
LOGDIR="$OUTDIR/logs"

mkdir -p "$OUTDIR" "$LOGDIR"

EXPORTS=ALL,DATASET="$DATASET",NAME="$NAME",OUTDIR="$OUTDIR",SOURCE_IS_ENGLISH="$SOURCE_IS_ENGLISH"

mt_jid=$(sbatch --parsable \
  --export="$EXPORTS" \
  --job-name="mt_${NAME}" \
  --output="$LOGDIR/${NAME}_mt-%j.out" \
  --error="$LOGDIR/${NAME}_mt-%j.err" \
  run_mt.slurm)
echo "Submitted MT: $mt_jid"


sc_jid=$(
  sbatch --parsable \
    --dependency=afterok:$mt_jid \
    --export="$EXPORTS" \
    --job-name="sc_${NAME}" \
    --output="$LOGDIR/${NAME}_sc-%j.out" \
    --error="$LOGDIR/${NAME}_sc-%j.err" \
    run_sc.slurm
)
echo "Submitted SC after MT: $sc_jid"

lg_jid=$(
  sbatch --parsable \
    --dependency=afterok:$mt_jid \
    --export="$EXPORTS" \
    --job-name="lg_${NAME}" \
    --output="$LOGDIR/${NAME}_lg-%j.out" \
    --error="$LOGDIR/${NAME}_lg-%j.err" \
    run_lg.slurm
)
echo "Submitted LG after MT: $lg_jid"

ma_jid=$(
  sbatch --parsable \
    --dependency=afterok:$mt_jid \
    --export="$EXPORTS" \
    --job-name="ma_${NAME}" \
    --output="$LOGDIR/${NAME}_ma-%j.out" \
    --error="$LOGDIR/${NAME}_ma-%j.err" \
    run_ma.slurm
)
echo "Submitted MA after MT: $ma_jid"


lf_jid=$(
  sbatch --parsable \
    --dependency=afterok:$sc_jid:$lg_jid:$ma_jid \
    --export="$EXPORTS" \
    --job-name="lf_${NAME}" \
    --output="$LOGDIR/${NAME}_lf-%j.out" \
    --error="$LOGDIR/${NAME}_lf-%j.err" \
    run_lf.slurm
)
echo "Submitted LF after SC/LG/MA: $lf_jid"
