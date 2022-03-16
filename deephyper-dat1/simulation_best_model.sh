#!/bin/bash
#SBATCH -J bm
#SBATCH -o bms.o%j
#SBATCH -t 2-00:00:00
#SBATCH -N 1 -n 1
#SBATCH --mem=64GB
#SBATCH --mail-user=machaudry@uh.edu
#SBATCH --mail-type=END

/project/jun/machaudr/miniconda3.9/envs/deephyper/bin/python -u best_model_sim.py
