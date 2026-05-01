#!/bin/bash -l

#$ -P multilm
#$ -N rec_cooc
#$ -l h_rt=04:00:00
#$ -pe omp 8
#$ -l mem_per_core=8G
#$ -j y
#$ -o job_out/

# Submit:
#   qsub scripts/run_rec_cooc.sh

conda activate cs506-project
cd /projectnb/multilm/jongin/+Projects/CS506-project-final
python rec_cooc.py
