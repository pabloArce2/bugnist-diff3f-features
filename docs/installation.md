# Installation

Everything runs in one conda environment called `diff3f`. Descriptor extraction needs an
NVIDIA GPU; the other steps (CT segmentation, smoothing, visualisation, matching,
benchmarks) run fine on a CPU.

## Pick an environment file

| File | Use it for |
| --- | --- |
| `environment.windows.yaml` | Windows with a recent NVIDIA driver (conda-forge PyTorch built for CUDA 13) |
| `environment.dtu.yaml` | Linux and the DTU HPC cluster (PyTorch 2.1, CUDA 11.8) |
| `environment.yaml` | the original upstream file, kept for reference |
| `eval_environment.yaml` | upstream SHREC/TOSCA evaluation only |

The first two contain everything the BugNIST scripts need, including PyTorch3D, which
conda-forge provides as a prebuilt package.

```bash
conda env create -f environment.windows.yaml
conda activate diff3f
```

To update an existing environment after the file changed:

```bash
conda env update -n diff3f -f environment.windows.yaml --prune
```

Check that PyTorch sees the GPU and that PyTorch3D imports:

```bash
python -c "import torch, pytorch3d; print(torch.__version__, torch.cuda.is_available())"
```

If `pytorch3d` fails to import, follow the
[PyTorch3D install notes](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md);
the PyTorch3D version must be built for the same PyTorch and CUDA versions as the one in
the environment.

## Models

The first descriptor run downloads the models into the Hugging Face and torch caches
(about 5.5 GB):

| Model | Source |
| --- | --- |
| Stable Diffusion 1.5 (UNet, VAE, text encoder) | `runwayml/stable-diffusion-v1-5` |
| depth ControlNet | `lllyasviel/control_v11f1p_sd15_depth` |
| normal ControlNet | `lllyasviel/control_v11p_sd15_normalbae` |
| DINOv2 ViT-B/14 | `torch.hub`, `facebookresearch/dinov2` |

RunwayML took its repository down in 2024; the Hub now redirects that name to
`stable-diffusion-v1-5/stable-diffusion-v1-5`, which holds the same weights. If the redirect
ever fails, point the code at the mirror directly:

```bash
export DIFF3F_SD_MODEL=stable-diffusion-v1-5/stable-diffusion-v1-5   # PowerShell: $env:DIFF3F_SD_MODEL="..."
```

On a shared machine you can keep the caches next to the repository with `HF_HOME` and
`TORCH_HOME`, which is what the HPC job scripts do.

## GPU memory

A laptop GPU with 8 GB is enough for 16 views at 512 x 512 on a ~25k-vertex mesh, which is
what the project used: such a run took 4-7 minutes on an RTX 5060 Laptop GPU, almost all of
it in the 16 Stable Diffusion passes. If you run out of memory:

- use `--height 256 --width 256` (much faster, fewer details),
- simplify the mesh first (`scripts/smooth_simplify_mesh.py`; a raw CT mesh often has
  100k-1.4M vertices),
- or run on the cluster ([hpc.md](hpc.md)).

Filling vertices that no camera saw is done on the CPU in chunks, so large meshes do not
need a large GPU for that step.

## Blender (optional)

Blender is only needed to place landmarks by hand and to build the optional `.blend`
scenes. Any recent version works (the project used 5.2). The Blender scripts run inside
Blender's own Python, not the conda environment:

```bash
blender --background --python scripts/make_blender_landmark_scene.py -- --help
```

## Checking the install

```bash
python -m unittest discover -s tests
```

runs the unit tests (CPU only, a few seconds). The camel example in the
[README](../README.md#quick-start) then checks the GPU part end to end.

## Known issues

**A script exits immediately with no output (Windows, exit code 127).** This happens when
the environment's `python.exe` is started without `conda activate diff3f`: Windows then
does not find the MKL DLLs and NumPy's linear algebra crashes (`Windows fatal exception:
code 0xc06d007f`). Activate the environment first. The browser app sets up the same
`PATH` itself when it starts a job.

**`--num-views must be a perfect square`.** The original camera layout (`--view-sampling
grid`) needs 4, 9, 16, 25, ... views. Use `--view-sampling insect` or `fibonacci` for any
other number.

**`geometry has N vertices but the descriptor has M rows`.** A descriptor belongs to the exact
file it was computed from. Smoothing, decimating, re-exporting from Blender or resampling a
point cloud changes the vertex count or order, so compute a new descriptor for the new file.

**`enable_model_cpu_offload requires accelerate`.** Harmless: the pipeline falls back to
keeping the models on the GPU. Install `accelerate` (it is in the environment files) to
save some GPU memory.

**`xFormers is not available` / `text_encoder/model.safetensors not found`.** Warnings only.
