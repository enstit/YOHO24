#!/bin/bash
#SBATCH -p EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50GB
#SBATCH --time=02:00:00
#SBATCH --output=_%j.out
#SBATCH --job-name=yoho24

# Activate the virtual environment
source /u/dssc/$(whoami)/scratch/torchenv/bin/activate

cd /u/dssc/$(whoami)/scratch/YOHO24/

# Run the train.py script
python3 -m yoho.dataset