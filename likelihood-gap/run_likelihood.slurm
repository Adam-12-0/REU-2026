#!/bin/bash
#SBATCH --job-name=likelihood
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=va608804@ucf.edu
#SBATCH --output=likelihood-%j.out
#SBATCH --error=likelihood-%j.err

cd /home/va608804/project4/detection_filters/2_langsurprise
/home/va608804/.conda/envs/dlcv/bin/python -u likelihood.py 
