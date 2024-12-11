#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J prompts
#SBATCH --mail-user=psivaram@umd.edu
#SBATCH --mail-type=ALL
#SBATCH -A m2404
#SBATCH -t 00:30:00

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application: 
srun -n 1 -c 256 --cpu_bind=cores python prompts.py
