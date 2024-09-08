#!/bin/bash
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --cpu-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=_%j.out
#SBATCH --job-name=yoho24

# Create a new virtual environment with Python 3.11.9 and install the needed packages

cd /u/dssc/$(whoami)/scratch/

wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
tar -xzf Python-3.11.9.tgz
rm Python-3.11.9.tgz
cd Python-3.11.9/
./configure --enable-optimizations CC="gcc -pthread" CXX="g++ -pthread" --enable-loadable-sqlite-extensions
make -j 24

cd /u/dssc/$(whoami)/scratch

python3 -m virtualenv --python="Python-3.11.9/python" torchenv3.11

# Activate the virtual environment
source /u/dssc/$(whoami)/scratch/torchenv3.11/bin/activate

cd /u/dssc/$(whoami)/scratch/YOHO24/

# Upgrade Pip to the latest version
pip install --upgrade pip

# Install the needed packages
pip install -r requirements.txt
