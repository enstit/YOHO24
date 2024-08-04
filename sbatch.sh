#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpu-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --job-name=yoho24

# Activate the virtual environment
source $(pwd)/yoho24/bin/activate
