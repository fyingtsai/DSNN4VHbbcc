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
nvidia-smi
#conda install -c conda-forge root
#conda install -c conda-forge keras=2.4.3 flatbuffers=1.12 joblib=1.0.1 pillow=8.1.0 pytz=2021.1 scikit-learn=0.24.1 scipy=1.6.0
#pip install energyflow root-numpy
#pip install --upgrade --user https://github.com/rootpy/root_numpy/zipball/master
#pip install sklearn==0.0
#pip install pandas==1.2.2
#pip install uproot==4.0.2
#pip install awkward==1.1.1
srun ./myDSNNr_run.sh
