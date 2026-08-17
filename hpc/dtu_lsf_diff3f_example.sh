#!/bin/bash
### DTU HPC uses LSF for these GPU queues; submit with:
### bsub < hpc/dtu_lsf_diff3f_example.sh

### -- select a GPU queue; change this if DTU tells you to use another one --
#BSUB -q gpua100
### -- job name --
#BSUB -J diff3f_features
### -- CPU cores on one host --
#BSUB -n 8
#BSUB -R "span[hosts=1]"
### -- memory is per core on LSF; 8GB x 8 cores = 64GB total request --
#BSUB -R "rusage[mem=8GB]"
### -- one exclusive GPU --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- walltime hh:mm; DTU GPU queues may cap this --
#BSUB -W 06:00
### -- logs --
#BSUB -oo logs/diff3f_%J.out
#BSUB -eo logs/diff3f_%J.err

set -eu

cd "$LS_SUBCWD"
mkdir -p logs output/hpc_features hf_cache torch_cache

export HF_HOME="$PWD/hf_cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf_cache/hub"
export TORCH_HOME="$PWD/torch_cache"
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-8}"

### Adjust this block to match how your DTU environment is installed.
### Examples:
###   module load miniconda3
###   source ~/miniconda3/bin/activate diff3f
### or, if conda is already initialized:
###   conda activate diff3f
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate diff3f
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate diff3f
else
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate diff3f
fi

python scripts/compute_mesh_features.py \
    --mesh meshes/cow.obj meshes/camel.obj \
    --prompt cow camel \
    --outdir output/hpc_features \
    --num-views 100 \
    --height 512 \
    --width 512 \
    --tolerance 0.004
