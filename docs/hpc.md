# Running on the DTU HPC cluster

Descriptor runs on large meshes, with many views or for many specimens are easier on the
cluster. DTU's GPU queues use LSF; two job scripts are in `hpc/`:

- `hpc/dtu_lsf_overnight_smoke.sh`: cow mesh, 9 views at 256 x 256. Takes minutes and shows
  whether the environment, the model downloads and the GPU work.
- `hpc/dtu_lsf_diff3f_example.sh`: cow and camel at 100 views, 512 x 512. Copy it and change
  the `python scripts/compute_mesh_features.py ...` line for real jobs.

Both request one GPU in exclusive mode, 8 cores and 64 GB RAM, and keep the Hugging Face and
torch caches in `hf_cache/` and `torch_cache/` inside the repository, so the models are
downloaded once and not per job.

## One-time setup

Install conda in your home directory (Miniforge works well), then create the environment
from a login node:

```bash
cd ~
wget -O Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false

git clone https://github.com/pabloArce2/bugnist-diff3f-features.git
cd bugnist-diff3f-features
conda env create -f environment.dtu.yaml
```

Copy data with `scp`/`rsync` or an SFTP client to `transfer.gbar.dtu.dk`. Only copy what a
job needs (the mesh files). A descriptor is 4 KB per vertex, so about 100 MB for a
25k-vertex mesh and more than 400 MB for an unsimplified CT mesh.

## Submitting

From the repository folder:

```bash
mkdir -p logs
bsub < hpc/dtu_lsf_overnight_smoke.sh
bjobs                      # queue status
bpeek <job id>             # output of a running job
```

Logs go to `logs/diff3f_<job id>.out` and `.err` (`diff3f_smoke_...` for the smoke test). The scripts activate the `diff3f`
environment themselves; edit the activation block if conda lives somewhere else.

Do not run descriptor extraction on the login nodes. `torch.cuda.is_available()` is `False`
there, which is expected; the GPU is only visible inside a job.

Change the queue (`#BSUB -q gpua100`, `gpul40s`, ...) and the wall time (`#BSUB -W`) to what
the cluster currently offers; the DTU HPC documentation lists the GPU queues and their limits.
