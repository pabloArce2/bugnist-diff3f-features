# Documentation

Start here when navigating the BugNIST Diff3F work.

The repository has two documentation tracks:

```text
documentation/
  guides/        commands and practical workflows
  information/   explanations, concepts, and interpretation notes
```

## Guides

Use these when you want to run something.

- [Command Guide](guides/COMMAND_GUIDE.md): main command reference for previews, geometry generation, descriptors, visualization, debug views, quality checks, and correspondence tests.
- [Point Cloud Diff3F Pipeline](guides/POINTCLOUD_DIFF3F_PIPELINE.md): focused guide for point-cloud descriptor extraction and point-cloud visualization.

## Information

Use these when you want to understand what the pipeline is doing.

- [Diff3F Descriptors Explained](information/DIFF3F_DESCRIPTORS_EXPLAINED.md): what `.pt` files are, what the 2048-D vectors mean, and how descriptors attach to geometry.
- [2D Debug Images Explained](information/DIFF3F_2D_DEBUG_IMAGES_EXPLAINED.md): what the render/depth/normal/AI/PCA debug images show.

## Repository Map

```text
scripts/
  BugNIST conversion, preview, descriptor, visualization, and metric scripts.

dataloaders/
  Mesh/point-cloud data helpers inherited from the original Diff3F project.

hpc/
  DTU LSF job examples.

meshes/
  Small original example meshes plus ignored local BugNIST generated meshes.

pointclouds/
  Ignored local BugNIST point-cloud data.

output/
  Ignored local descriptor outputs, especially `.pt` files.

visualizations/
  Ignored local colored PLYs, Blender scenes, correspondence exports, and previews.

debug/
  Ignored local 2D Diff3F debug image folders.
```

Generated data is intentionally ignored by Git. Commit code, scripts, environment files, and documentation; keep large `.pt`, `.tif`, `.blend`, point-cloud, preview, and output files local unless you deliberately publish them elsewhere.
