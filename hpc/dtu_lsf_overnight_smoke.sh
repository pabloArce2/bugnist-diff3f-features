#!/bin/bash
### Submit from a DTU HPC login node with:
### bsub < hpc/dtu_lsf_overnight_smoke.sh

#BSUB -q gpul40s
#BSUB -J diff3f_smoke
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 03:00
#BSUB -oo logs/diff3f_smoke_%J.out
#BSUB -eo logs/diff3f_smoke_%J.err

set -eu

cd "$LS_SUBCWD"
mkdir -p logs output/hpc_smoke hf_cache torch_cache

export HF_HOME="$PWD/hf_cache"
export HUGGINGFACE_HUB_CACHE="$PWD/hf_cache/hub"
export TORCH_HOME="$PWD/torch_cache"
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-8}"

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
    --mesh meshes/cow.obj \
    --prompt cow \
    --outdir output/hpc_smoke \
    --num-views 9 \
    --height 256 \
    --width 256 \
    --tolerance 0.004
