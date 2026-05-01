#!/bin/bash -l

#$ -P multilm           # SCC project
#$ -N rec_pop           # job name
#$ -l h_rt=01:00:00     # hard time limit
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -j y                 # merge stdout/stderr
#$ -o job_out/          # output directory for job logs

# Submit:
#   qsub scripts/run_rec_pop.sh

conda activate cs506-project
cd /projectnb/multilm/jongin/+Projects/CS506-project-final
python rec_pop.py
