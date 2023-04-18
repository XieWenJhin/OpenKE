#!/bin/sh 
#SBATCH -J openke
#SBATCH -p cpu

# srun python -u ve2tri.py --vfile ./datasets/YAGO/yago_v.csv --efile ./datasets/YAGO/yago_e.csv --test_edges ./datasets/YAGO/test_edges.csv --save_dir ./datasets/YAGO
srun python -u ve2tri.py --vfile ./datasets/DBpedia/dbpedia_v.csv --efile ./datasets/DBpedia/dbpedia_e.csv --test_edges ./datasets/DBpedia/test_edges.csv --save_dir ./datasets/DBpedia