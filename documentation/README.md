# Documentation

Start here when navigating the BugNIST Diff3F work.

The repository has two documentation tracks:

```text
documentation/
  guides/        commands and practical workflows
  information/   explanations, concepts, and interpretation notes
  presentations/ slide decks for talking through the workflow
  results/       readable summaries of completed experiments
```

## Guides

Use these when you want to run something.

- [Command Guide](guides/COMMAND_GUIDE.md): main command reference for previews, geometry generation, descriptors, visualization, debug views, quality checks, and correspondence tests.
- [Point Cloud Diff3F Pipeline](guides/POINTCLOUD_DIFF3F_PIPELINE.md): focused guide for point-cloud descriptor extraction and point-cloud visualization.
- [Landmark Benchmarking](guides/LANDMARK_BENCHMARKING.md): manual Blender landmarks and quantitative Diff3F correspondence evaluation.
- [Interactive 3-D Correspondences](guides/INTERACTIVE_CORRESPONDENCES.md): browser mesh comparison with correspondence lines, landmark errors, and surface/vertex coordinate inspection.
- [Shared-PCA K-Means Clustering](guides/PCA_KMEANS_CLUSTERING.md): turn the shared-PCA RGB colors into a discrete k-way cluster map, shared across compared items.

## Information

Use these when you want to understand what the pipeline is doing.

- [Diff3F Descriptors Explained](information/DIFF3F_DESCRIPTORS_EXPLAINED.md): what `.pt` files are, what the 2048-D vectors mean, and how descriptors attach to geometry.
- [How Diff3F `.pt` Files Are Filled](information/HOW_DIFF3F_PT_FILES_ARE_FILLED.md): how each rendered view contributes to the final `N x 2048` tensor.
- [2D Debug Images Explained](information/DIFF3F_2D_DEBUG_IMAGES_EXPLAINED.md): what the render/depth/normal/AI/PCA debug images show.
- [Otsu Thresholding Explained](information/OTSU_THRESHOLDING_EXPLAINED.md): what automatic Otsu segmentation is and when it helps or fails.

## Presentations

- [How Diff3F `.pt` Files Are Filled](presentations/Diff3F_PT_Files_Filled_Guide.pptx): slide deck for explaining how the final descriptor tensor is formed.

## Results

- [Cricket Landmark Benchmark Summary](results/CRICKET_LANDMARK_BENCHMARK_SUMMARY.md): readable summary of the brown-to-black and black-to-brown cricket landmark correspondence experiments.
- [Cricket Shared-PCA K-Means Clustering Summary](results/CRICKET_PCA_CLUSTERING_SUMMARY.md): readable summary of the shared K-means clustering run on the brown/black cricket descriptors.

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
