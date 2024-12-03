#!/bin/bash
#SBATCH --job-name="MPItry"
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=mpi.out

module load OpenMPI

srun ./shallow param_simple.txt

