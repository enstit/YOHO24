#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpu-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=_%j.out
#SBATCH --job-name=yoho24

# Activate the virtual environment
source /u/dssc/$(whoami)/scratch/torchenv/bin/activate

cd /u/dssc/$(whoami)/scratch/YOHO24/

# Install the needed packages
pip install .r requirements.txt

# Run the train.py script
python3 -m yoho.train
