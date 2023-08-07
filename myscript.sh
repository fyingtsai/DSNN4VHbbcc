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
conda activate /path/to/your/scratch/tf2-gpu-yml
srun ./myDSNNr_run.sh
