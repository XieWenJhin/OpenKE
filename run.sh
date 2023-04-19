#!/bin/sh 
#SBATCH -J openke
#SBATCH -p inspur
##SBATCH -w inspur-gpu-04
##SBATCH -x inspur-gpu-13
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# srun python -u train_transe_FB15K237.py > log/FB15K237/train_transe.txt 2>&1 
# srun python -u train_dblp_simple.py > log/DBLP/train_simple.txt 2>&1 
# srun python -u train_dblp_complex.py > log/DBLP/train_complex.txt 2>&1 
# srun python -u train_dblp_rotate.py > log/DBLP/train_rotate.txt 2>&1 

# srun python -u train_imdb_simple.py > log/IMDB/train_simple.txt 2>&1 
# srun python -u train_imdb_complex.py > log/IMDB/train_complex.txt 2>&1 
# srun python -u train_imdb_rotate.py > log/IMDB/train_rotate.txt 2>&1 

# srun python -u train_yago_simple.py > log/YAGO/train_simple.txt 2>&1 
# srun python -u train_yago_complex.py > log/YAGO/train_complex.txt 2>&1 
# srun python -u train_yago_rotate.py > log/YAGO/train_rotate_10epoch.txt 2>&1 

# srun python -u train_dbpedia_simple.py > log/DBpedia/train_simple.txt 2>&1 
# srun python -u train_dbpedia_complex.py > log/DBpedia/train_complex.txt 2>&1
# srun python -u train_dbpedia_rotate.py > log/DBpedia/train_rotate_25.txt 2>&1

# srun python -u eval_F1_dblp.py --model simple > log/DBLP/eval_simple.txt 2>&1
# srun python -u eval_F1_dblp.py --model complex > log/DBLP/eval_complex.txt 2>&1
# srun python -u eval_F1_dblp.py --model rotate > log/DBLP/eval_rotate.txt 2>&1

# srun python -u eval_F1_imdb.py --model simple > log/IMDB/eval_simple.txt 2>&1
# srun python -u eval_F1_imdb.py --model complex > log/IMDB/eval_complex.txt 2>&1
# srun python -u eval_F1_imdb.py --model rotate > log/IMDB/eval_rotate.txt 2>&1

# srun python -u eval_F1_yago.py --model simple > log/YAGO/eval_simple.txt 2>&1
# srun python -u eval_F1_yago.py --model complex > log/YAGO/eval_complex.txt 2>&1
srun python -u eval_F1_yago.py --model rotate > log/YAGO/eval_rotate.txt 2>&1

# srun python -u eval_F1_dbpedia.py --model simple > log/DBpedia/eval_simple.txt 2>&1
# srun python -u eval_F1_dbpedia.py --model complex > log/DBpedia/eval_complex.txt 2>&1
# srun python -u eval_F1_dbpedia.py --model rotate > log/DBpedia/eval_rotate.txt 2>&1

