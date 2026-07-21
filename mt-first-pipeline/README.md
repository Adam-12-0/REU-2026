# MT-First Pipeline

This project runs a multi-stage specialized language detection pipeline on a Slurm cluster.


## Pipeline

The driver submits the following dependency graph:

1. MT runs first.
2. SC, LG, and MA run in parallel after MT succeeds.
3. LF runs after SC, LG, and MA all succeed.
4. Slurm's afterok dependency ensures that a downstream stage starts only when its required upstream jobs finish successfully.


## Requirements

- A Slurm-based computing cluster
- Bash
- Python and the dependencies required by each pipeline stage
  
These job scripts in the submission directory:
- run_mt.slurm
- run_sc.slurm
- run_lg.slurm
- run_ma.slurm
- run_lf.slurm
  
The input dataset must be available from the submission directory or at the configured path. 


## Configuration

Edit the following variables near the top of the driver script:

- DATASET="genz_normalized.csv"
- NAME="genz"
- SOURCE_IS_ENGLISH="1"
- OUTDIR="pipeline_outputs/${NAME}"

Variable Descriptions:

- DATASET	Input CSV file or path
- NAME	Unique name used for jobs and output directories
- SOURCE_IS_ENGLISH	Set to 1 when the source data is English; otherwise set to 0
- OUTDIR	Directory where pipeline results are written

Each submitted stage receives these variables through Slurm's --export option.


## Running the Pipeline

From the directory containing the driver and stage scripts, submit the driver:

sbatch run_driver.slurm

Do not run it with srun. The driver is a short Slurm job that submits the remaining pipeline jobs.

The command returns a driver job ID:

Submitted batch job 123456


The driver's output file reports the job ID assigned to each stage:

Submitted MT: 123457

Submitted SC after MT: 123458

Submitted LG after MT: 123459

Submitted MA after MT: 123460

Submitted LF after SC/LG/MA: 123461


## Output Structure

Results and logs are written under:

pipeline_outputs/${NAME}/logs/${NAME}-mt-{job-id}.out   ${NAME}-mt-{job-id}.err

The driver itself writes:

d-{job-id}.out

d-{job-id}.err


## Failure Behavior

The driver uses:
set -euo pipefail

It exits if a command fails, an undefined variable is referenced, or a pipeline command fails.
The stage dependencies use afterok. 

Therefore:

If MT fails, SC, LG, and MA will not run.

If SC, LG, or MA fails, LF will not run.


## Failure Cases

If MT fails:

1. Cancel the remaining jobs in the pipeline: SC, LG, MA, LF.
2. Submit driver again.

If SC fails:

1. Cancel LF only. 
2. Allow LG and MA to continue running.
3. Submit rerun_sc.slurm using the same configuration scheme as the driver.
4. Allow LG, MA, and the new SC run to all finish.
5. Submit rerun_lf.slurm, using the same configuration scheme as the driver.

If MA fails:

1. Cancel LF only.
2. Allow SC and LG to continue running.
3. Submit rerun_ma.slurm using the same configuration scheme as the driver.
4. Allow SC, LG, and the new MA run to all finish.
5. Submit rerun_lf.slurm, using the same configuration scheme as the driver.


# README for Child Slurms
## MT
Required Files:

The following files must be available in the Slurm submission directory:

- run_mt.slurm
- mt_judge.py : Runs translation cycle for 4 languages
- mt_judge_en.py : Runs translation cycle for 4 languages, assuming input is English
- mt_judge_predictions.py : Applies judge to 4 results
- mt_consecutive.py : Builds phrases out of consecutive candidates

Configuration:
The job inherits configured variables from the driver slurm.

English-script selection:
The job uses mt_judge_en.py when SOURCE_IS_ENGLISH is 1. Else, it uses mt_judge.py

Outputs:
- thr_metrics.csv	: Threshold evaluation results produced by the judge
- mt_preds_sw.csv	: Single-word MT predictions
- mt_preds_mw.csv	: Multiword or consecutive MT predictions
- mt_preds_all.csv : Combined single-word and multiword predictions
- mt_metrics_all.csv	: Metrics for the combined predictions

  The intermediate file is removed after successful use: _mt_word_scores_tmp.csv

## SC
Required Files:

- run_sc.slurm
- sc.py
- <OUTDIR>/mt_outputs/mt_preds_all.csv
  
Configuration:
The job inherits configured variables from the driver slurm.

Output Files:
- sc_scoring_breakdown.csv	: Detailed SC scores and scoring components
- sc_thr_metrics.csv	: Metrics evaluated across SC thresholds
- sc_preds_sw.csv	: Single-word SC predictions
- sc_preds_mw.csv	: Multiword SC predictions
- sc_preds_all.csv	: Combined single-word and multiword predictions
- sc_metrics_all.csv	: Evaluation metrics for the combined predictions

## LG
Required Files:
- run_lg.slurm
- lg.py : Generate likelihood gap scores
- lg_pred.py : Applies scores to generate single and multi word predictions, applies phrase picker
- <OUTDIR>/mt_outputs/mt_preds_all.csv

Configuration:
The job inherits configured variables from the driver slurm.

Output Files:
- lg_scoring_breakdown.csv :	LG features and detailed scoring information
- lg_thr_metrics.csv	: Metrics evaluated across LG thresholds
- lg_preds_sw.csv	: Single-word LG predictions
- lg_preds_mw.csv	: Multiword LG predictions
- lg_preds_all.csv :	Combined single-word and multiword predictions
- lg_metrics_all.csv	: Evaluation metrics for the combined predictions

## MA
Required Files:
- run_ma.slurm
- ma_agree.py - Generates Llama, Qwen, Mistral definitions
- ma_score.py - Calculates similarity scores
- ma_pred.py - Produces final predictions and phrase picker
- <OUTDIR>/mt_outputs/mt_preds_all.csv

Configuration:
The job inherits configured variables from the driver slurm.

Output Files:
- ma_thr_metrics.csv	: Metrics evaluated across MA thresholds
- ma_preds_sw.csv :	Single-word MA predictions
- ma_preds_mw.csv	: Multiword MA predictions
- ma_preds_all.csv	: Combined single-word and multiword predictions
- ma_metrics_all.csv	: Evaluation metrics for the combined predictions

## LF (Logistic Fusion)
Required Files
- run_lf.slurm
- compile.py : Compiles SC, LG, MA predictions into a single file
- lf.py : Calculates final fusion score after training a logistic regression model
- <OUTDIR>/sc_outputs/sc_preds_all.csv
- <OUTDIR>/lg_outputs/lg_preds_all.csv
- <OUTDIR>/ma_outputs/ma_preds_all.csv

Configuration:
The job inherits configured variables from the driver slurm.

Output Files:
- preds_compiled.csv	: Combined SC, LG, and MA predictions
- lf_preds_sw.csv	: Final single-word LF predictions
- lf_preds_mw.csv	: Final multiword LF predictions
- lf_preds_all.csv	: Combined final LF predictions
- lf_thr_metrics.csv	: Metrics evaluated across LF thresholds
- lf_weights.csv	: Selected or evaluated ensemble weights
- lf_metrics_all.csv	: Final evaluation metrics

