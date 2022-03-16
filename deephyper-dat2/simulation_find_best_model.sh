#!/bin/bash
#SBATCH -J find_bm
#SBATCH -o fbms.o%j
#SBATCH -t 6-00:00:00
#SBATCH -N 1 -n 10
#SBATCH --mem=256GB
#SBATCH --mail-user=machaudry@uh.edu
#SBATCH --mail-type=END

/project/jun/machaudr/miniconda3.9/envs/deephyper/bin/python -u find_best_model_dat2.py
