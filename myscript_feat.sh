#!/bin/bash 
#SBATCH --partition usatlas
#SBATCH --time=24:00:00
#SBATCH --account=tier3
#SBATCH --nodes=1
#SBATCH -o joblogs/%j.out
#SBATCH -e joblogs/%j.err
#SBATCH --qos usatlas
#SBATCH --gres=gpu:1
#SBATCH --mem=230000
bash
eval "$(/hpcgpfs01/software/anaconda3/2020-11/bin/conda shell.bash hook)"
cd /hpcgpfs01/scratch/ftsai/DSNNrBranch
conda create --prefix ./tf2-gpu tensorflow-gpu matplotlib tensorboard
conda activate /hpcgpfs01/scratch/ftsai/DSNNrBranch/tf2-gpu
srun python PreScaleInputRange.py
