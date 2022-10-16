#!/bin/bash
#SBATCH -J retrieve
#SBATCH -N 1
#SBATCH -n 96
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=300000MB
#SBATCH -o logs/fastMSA-%j.log
#SBATCH -e logs/fastMSA-%j.error
#SBATCH --time=48:00:00
#SBATCH -p 3090

echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NODELIST=$SLURM_NODELIST"

date

echo "Starting job ..."


echo "Starting @ `date` "
/user/liyu/miniconda3/envs/rt/bin/python -u $1
wait
echo "Complete ..."
