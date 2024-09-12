#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=8
#SBATCH --mem=50GB
#SBATCH --time=02:00:00
#SBATCH --output=_%j.out
#SBATCH --job-name=yoho24

# Activate the virtual environment
source /u/dssc/$(whoami)/scratch/torchenv/bin/activate

cd /u/dssc/$(whoami)/scratch/YOHO24/

# Run the train.py script
python3 -m yoho.train "$@"

#jupyter nbconvert --to notebook --execute notebooks/02_urbansed.ipynb

# Commit the changes to the repository
#git add models/losses.json
#git add models/UrbanSEDYOHO_checkpoint.pth.tar
#git commit -m "feat: Update YOHO24 training results"
#git push
