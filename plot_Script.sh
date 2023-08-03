#!/bin/bash
#SBATCH --partition usatlas
#SBATCH --time=05:00:00
#SBATCH --account=tier3
#SBATCH --nodes=1
#SBATCH -o AdmaxPlot/%j.out
#SBATCH -e AdmaxPlot/%j.err 
#SBATCH --qos usatlas
bash
eval "$(/hpcgpfs01/software/anaconda3/2020-11/bin/conda shell.bash hook)"
cd /hpcgpfs01/scratch/ftsai/DSNNr4gpu
conda create --prefix ./tf2-cpu tensorflow matplotlib tensorboard
conda activate /hpcgpfs01/scratch/ftsai/DSNNr4gpu/tf2-cpu
#source myVirtualEnv/bin/activate
srun python makingSpecPlots.py
#srun ./myDSNNr_runPlots.sh
