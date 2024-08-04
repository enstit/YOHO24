#!/bin/bash
#SBATCH --job-name=yoho
#SBATCH --account=dcapone
#SBATCH --output=my_job_%j.out

# Activate the virtual environment
source $(pwd)/yoho24/bin/activate
