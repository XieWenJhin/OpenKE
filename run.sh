#!/bin/sh 
#SBATCH -J openke
#SBATCH -p inspur
#SBATCH -w inspur-gpu-09
##SBATCH -x inspur-gpu-12
##SBATCH --gres=gpu:1

# srun python -u train_transe_FB15K237.py > log/FB15K237/train_transe.txt 2>&1 
# srun python -u train_simple_dblp.py > log/DBLP/train_simple.txt 2>&1 
# srun python -u train_complex_dblp.py > log/DBLP/train_complex.txt 2>&1 
# srun python -u train_simple_imdb.py > log/IMDB/train_simple.txt 2>&1 
# srun python -u train_simple_imdb.py > log/IMDB/train_complex.txt 2>&1 


srun python -u eval_F1.py > log/DBLP/eval_simple.txt 2>&1
