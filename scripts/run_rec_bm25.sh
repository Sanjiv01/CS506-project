#!/bin/bash -l

#$ -P multilm
#$ -N rec_bm25
#$ -l h_rt=06:00:00
#$ -pe omp 8
#$ -l mem_per_core=8G
#$ -j y
#$ -o job_out/

# Submit:
#   qsub scripts/run_rec_bm25.sh
# Override hyper-params via env, e.g.:
#   qsub -v BM25_K1=1.2,BM25_B=0.6 scripts/run_rec_bm25.sh

conda activate cs506-project
cd /projectnb/multilm/jongin/+Projects/CS506-project-final
python rec_bm25.py
