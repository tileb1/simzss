#!/bin/bash
#SBATCH --job-name=process_shards
#SBATCH --account=lkeeponlearning
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=72
#SBATCH --nodes=3
#SBATCH --open-mode=append
#SBATCH --partition=batch_icelake
#SBATCH --cluster=wice
#SBATCH --time=180

source ~/.bashrc
cd /data/leuven/342/vsc34272/Contextual-CLIP/src/training
export PYTHONPATH="$PYTHONPATH:$PWD/src"
conda activate simzss
echo $(which python)

srun python make_shards_spacy.py --base_raw /scratch/leuven/342/vsc34272/data/mscoco --base_processed /scratch/leuven/342/vsc34272/data/mscoco_processed
