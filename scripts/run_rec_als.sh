#!/bin/bash -l

#$ -P multilm
#$ -N rec_als
#$ -l h_rt=08:00:00
#$ -pe omp 8
#$ -l mem_per_core=12G
#$ -j y
#$ -o job_out/

# Submit:
#   qsub scripts/run_rec_als.sh
# Override hyper-params via env, e.g.:
#   qsub -v N_FACTORS=128,N_ITER=20 scripts/run_rec_als.sh

conda activate cs506-project
cd /projectnb/multilm/jongin/+Projects/CS506-project-final
python rec_als.py
