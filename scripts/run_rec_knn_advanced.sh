#!/bin/bash -l

#$ -P multilm
#$ -N rec_knn_adv
#$ -l h_rt=24:00:00     # rho sweep is 7x the basic kNN runtime
#$ -pe omp 8
#$ -l mem_per_core=8G
#$ -j y
#$ -o job_out/

# Submit:
#   qsub scripts/run_rec_knn_advanced.sh

conda activate cs506-project
cd /projectnb/multilm/jongin/+Projects/CS506-project-final
python rec_knn_advanced.py
