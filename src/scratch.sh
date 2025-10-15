salloc -N1 -pdev-g -t 30:00 --account=project_465000727
module load LUMI PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-20240209
srun -N1 -n1 --gpus 8 singularity exec $SIF /runscripts/conda-python-simple \
    -c 'import torch; print("I have this many devices:", torch.cuda.device_count())'