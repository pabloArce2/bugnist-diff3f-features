# Diff3F on BugNIST

This is a fork of [Diffusion 3D Features (Diff3F)](https://github.com/niladridutt/Diffusion-3D-Features)
(Dutt et al., CVPR 2024) that applies the method to X-ray CT scans of insects from
[BugNIST](https://arxiv.org/abs/2304.01838). It was written for a special course at DTU.

Diff3F gives every vertex of an untextured mesh a 2048-dimensional descriptor. It renders
the shape from several cameras, lets Stable Diffusion (guided by depth and normal
ControlNets) paint a plausible texture on each render, reads features from the diffusion
UNet and from DINOv2, and projects them back onto the surface. Points with similar
descriptors should be the same part on different shapes: head with head, leg with leg.

BugNIST contains voxel volumes, not meshes, so most of the work in this fork is around
the original method:

- **CT to geometry**: crop, threshold and clean a scan so that only the insect remains,
  then extract a surface (marching cubes) or a point cloud, smooth it and reduce it to
  about 50k faces.
- **Descriptors**: command-line wrappers for meshes and point clouds, a point-cloud
  version of the feature aggregation, camera layouts for elongated insects, and an option
  that saves the images of every view used for a descriptor.
- **Analysis**: shared PCA colours and shared k-means clusters across specimens,
  nearest-neighbour correspondences, label-free matching diagnostics, and a benchmark
  against landmarks placed by hand in Blender.

A browser front end for all of this lives in a separate repository,
[Diff3f_App](https://github.com/pabloArce2/Diff3f_App); clone it next to this one.

![Diff3F teaser](assets/teaser.jpg)

## Quick start

Clone the repository and install the environment (details and GPU notes in
[docs/installation.md](docs/installation.md)):

```bash
git clone https://github.com/pabloArce2/bugnist-diff3f-features.git
cd bugnist-diff3f-features
conda env create -f environment.windows.yaml   # Windows; environment.dtu.yaml on Linux / DTU HPC
conda activate diff3f
```

Check the install on the camel mesh that ships with the repository. The first run
downloads Stable Diffusion 1.5, two ControlNets and DINOv2 (about 5.5 GB).

```bash
python scripts/compute_mesh_features.py --mesh meshes/camel.obj --prompt "camel" \
    --outdir output/camel --num-views 4 --height 256 --width 256 --tosca
python scripts/visualize_mesh_features.py --mesh meshes/camel.obj \
    --features output/camel/camel_diff3f.pt --out visualizations/camel/camel_pca.ply \
    --preview visualizations/camel/camel_pca.png
```

`camel_pca.png` should show a camel with smoothly changing colours: similar colours mean
similar descriptors. (On PowerShell, end continued lines with a backtick instead of `\`.)

For an insect, the full route is: preview the scan, segment it into a mesh, smooth it,
rotate it into a common pose, compute descriptors, then compare specimens. Every step is
a script; [docs/pipeline.md](docs/pipeline.md) walks through them with the settings used
in the project.

## Documentation

| | |
| --- | --- |
| [docs/installation.md](docs/installation.md) | conda environments, PyTorch3D, model downloads, known Windows issues |
| [docs/pipeline.md](docs/pipeline.md) | the whole workflow, from a TIFF scan to a landmark benchmark, with commands |
| [docs/ct-to-mesh.md](docs/ct-to-mesh.md) | how the insect is separated from the scan and turned into a mesh or point cloud |
| [docs/descriptors.md](docs/descriptors.md) | what a `.pt` file contains and how Diff3F fills it |
| [docs/analysis.md](docs/analysis.md) | PCA colours, k-means, correspondences, metrics, landmarks, viewers |
| [docs/hpc.md](docs/hpc.md) | running descriptor jobs on the DTU cluster (LSF) |
| [docs/results.md](docs/results.md) | short summary of what the experiments showed |
| [scripts/README.md](scripts/README.md) | one line per script |

## Repository layout

```text
scripts/          command-line tools, one per pipeline step (python scripts/<name>.py --help)
bugnist_tools/    code shared by the scripts: CT segmentation, geometry I/O, PCA/k-means, matching
tests/            unit tests (python -m unittest discover -s tests)
landmarks/        hand-placed landmark CSVs for the cricket pair and the larva pair
hpc/              LSF job scripts for the DTU cluster
docs/             documentation

diff3f.py, render.py, render_point_cloud.py, diffusion.py, dino.py, ...
                  the Diff3F method itself (upstream code, with the patches listed in docs/descriptors.md)
dataloaders/, pyFM/, extract_*.py, evaluate_pipeline_*.py, test_correspondence.ipynb
                  unchanged upstream code for the SHREC'19 / TOSCA benchmarks
meshes/           example meshes from upstream
```

Scans, meshes, descriptors and renders are not in git. The scripts expect them in these
folders, created on first use:

```text
bugNIST/          input CT volumes (.tif)
meshes/<set>/     generated meshes, e.g. meshes/bugnist_crickets/bcrick_10_010/
pointclouds/      generated point clouds
output/           descriptors (.pt)
visualizations/   coloured PLYs, clusters, benchmark results, HTML viewers
previews/, debug/ PNG previews and per-view debug images
```

## Tests

```bash
python -m unittest discover -s tests
```

## The original Diff3F

Everything upstream still works: see [test_correspondence.ipynb](test_correspondence.ipynb)
for computing features and correspondences on the example meshes, and `extract_shrec.py` /
`evaluate_pipeline_shrec.py` (and the TOSCA versions) for the benchmarks in the paper. The
evaluation needs its own environment ([eval_environment.yaml](eval_environment.yaml)); the
upstream README explains the datasets:
[SHREC'19](https://nuage.lix.polytechnique.fr/index.php/s/LJFXrsTG22wYCXx/download?path=%2F&files=SHREC_r.zip),
[TOSCA and SHREC'07](http://www.vovakim.com/projects/CorrsBlended/SurfCorr2.0.benchmarks.bins.zip),
[SHREC'20](http://robertodyke.com/shrec2020/index2.html).

## Citation

If you use Diff3F, cite the original paper:

```bibtex
@InProceedings{Dutt_2024_CVPR,
    author    = {Dutt, Niladri Shekhar and Muralikrishnan, Sanjeev and Mitra, Niloy J.},
    title     = {Diffusion 3D Features (Diff3F): Decorating Untextured Shapes with Distilled Semantic Features},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {4494-4504}
}
```

and for the data, BugNIST: P. M. Jensen, V. A. Dahl, R. Engberg, C. Gundlach, H. M. Kjer,
A. B. Dahl, *BugNIST: a large volumetric dataset for object detection under domain shift*,
[arXiv:2304.01838](https://arxiv.org/abs/2304.01838).

The code is MIT licensed, like the original (see [LICENSE](LICENSE)). The example meshes in
`meshes/` come from various sources and are included as upstream distributed them.
