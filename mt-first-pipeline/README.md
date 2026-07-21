# MT-First Pipeline

This project runs a multi-stage specialized language detection pipeline on a Slurm cluster.


## Pipeline

The driver submits the following dependency graph:


             +-- SC --+
MT finishes -+-- LG --+-- LF
             +-- MA --+
             


MT runs first.
SC, LG, and MA run in parallel after MT succeeds.
LF runs after SC, LG, and MA all succeed.
Slurm's afterok dependency ensures that a downstream stage starts only when its required upstream jobs finish successfully.


## Requirements

A Slurm-based computing cluster
Bash
Python and the dependencies required by each pipeline stage
These job scripts in the submission directory:
run_mt.slurm
run_sc.slurm
run_lg.slurm
run_ma.slurm
run_lf.slurm
The input dataset must be available from the submission directory or at the configured path.


## Configuration

Edit the following variables near the top of the driver script:

DATASET="genz_normalized.csv"
NAME="genz"
SOURCE_IS_ENGLISH="1"
OUTDIR="pipeline_outputs/${NAME}"

Variable Descriptions:

DATASET	Input CSV file or path
NAME	Unique name used for jobs and output directories
SOURCE_IS_ENGLISH	Set to 1 when the source data is English; otherwise set to 0
OUTDIR	Directory where pipeline results are written

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
pipeline_outputs/
+-- genz/
    +-- logs/
    ¦   +-- genz_mt-<job-id>.out
    ¦   +-- genz_mt-<job-id>.err
    ¦   +-- genz_sc-<job-id>.out
    ¦   +-- genz_sc-<job-id>.err
    ¦   +-- ...
    +-- stage output directories and files
    
The driver itself writes:
d-<job-id>.out
d-<job-id>.err


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