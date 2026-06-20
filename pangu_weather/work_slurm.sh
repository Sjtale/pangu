#!/bin/bash
#SBATCH -p newlarge
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=8
#SBATCH -J Pangu_weather
#SBATCH --time=72:00:00
#SBATCH -o logs/%j.out
#SBATCH --exclusive

source ./earth_env.sh

export OMP_NUM_THREADS=1
export MASTER_ADDR=$(hostname)

echo SLURM_NNODES=$SLURM_NNODES
echo SLURM_NTASKS=$SLURM_NTASKS

srun -u --mpi=pmix\
    bash -c "
    source ../export_DDP_vars.sh
    python train.py
    "
