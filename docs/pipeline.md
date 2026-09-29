# From a CT scan to a landmark benchmark

This page goes through the whole workflow with the brown cricket `bcrick_10_010` as the
example. Every command runs from the repository root with the `diff3f` environment active.
Long commands are split with `\`; in PowerShell use a backtick instead, or put everything on
one line. Every script prints its full list of options with `--help`.

```text
 1. preview the scan             scripts/preview_tif_volume.py
 2. segment -> mesh              scripts/bugnist_tif_to_mesh.py        (or a point cloud)
 3. smooth and simplify          scripts/smooth_simplify_mesh.py
 4. orient the specimen          Blender, or scripts/transform_geometry.py
 5. check the camera views       scripts/debug_2d_diffusion_view.py --render-only --all-views
 6. compute descriptors          scripts/compute_mesh_features.py      (GPU)
 7. look at the descriptors      scripts/visualize_feature_comparison.py
 8. match two specimens          scripts/compute_feature_correspondences.py, evaluate_correspondence_metrics.py
 9. benchmark with landmarks     Blender scripts + scripts/evaluate_landmark_benchmark.py
```

Steps 1-5 and 7-9 run on the CPU; only step 6 needs a GPU. The browser app,
[Diff3f_App](https://github.com/pabloArce2/Diff3f_App), has a form for each of these steps
and runs the same scripts.

## 0. Data

The scans are not in the repository. Get the individual-specimen volumes from the BugNIST
release (Jensen et al., [arXiv:2304.01838](https://arxiv.org/abs/2304.01838)) and put them
under `bugNIST/`, for example:

```text
bugNIST/crickets/bcrick_10_010.tif     brown cricket
bugNIST/crickets/sfaar_10_010.tif      black cricket
bugNIST/raw/soldat_10_002.tif          soldier fly larva
```

Each file is a 512 x 256 x 256 volume of 8-bit X-ray attenuation (about 32 MB), indexed
(Z, Y, X). Only single-specimen scans are used; the BugNIST mixtures contain several insects
and would need the centroid files (`--centroids`) to crop one out.

## 1. Preview the scan

```bash
python scripts/preview_tif_volume.py --tif bugNIST/crickets/bcrick_10_010.tif \
    --outdir previews/bugnist_crickets_raw --threshold 45
```

This writes, into `previews/bugnist_crickets_raw/bcrick_10_010/`: the three centre slices
(`_orthos.png`), maximum-intensity projections (`_mips.png`), slice sheets along each axis,
the same images with the voxels above the threshold in red (`_overlay_...`), an intensity
histogram with the threshold marked (`_hist.png`) and a text summary. Use them to choose
the threshold and, if the insect does not fill the volume, a crop box. Without `--threshold`
the overlay uses Otsu's threshold. [ct-to-mesh.md](ct-to-mesh.md) explains how to read
these images and choose the values.

## 2. Segment the insect and extract a mesh

```bash
python scripts/bugnist_tif_to_mesh.py --tif bugNIST/crickets/bcrick_10_010.tif \
    --out meshes/bugnist_crickets/bcrick_10_010/preprocessed/bcrick_10_010_thr45_fillholes_keeplargest_ds1.obj \
    --threshold 45 --auto-crop --auto-crop-padding 12 12 12 --fill-holes --keep-largest
```

The script thresholds the volume, removes components smaller than 512 voxels, fills
enclosed cavities, keeps the largest connected component (the insect) and runs marching
cubes on the mask. Vertices are in voxel units, ordered (X, Y, Z). The file name records
the settings, which helps later: `thr45` threshold, `fillholes`, `keeplargest`, `ds1` no
downsampling.

For a point cloud instead of a mesh, the same segmentation options apply:

```bash
python scripts/bugnist_tif_to_pointcloud.py --tif bugNIST/crickets/bcrick_10_010.tif \
    --out pointclouds/bugnist_crickets/bcrick_10_010/bcrick_10_010_thr45_20k.ply \
    --npy pointclouds/bugnist_crickets/bcrick_10_010/bcrick_10_010_thr45_20k.npy \
    --threshold 45 --auto-crop --fill-holes --keep-largest --num-points 20000 --center
```

`--method mesh-surface` (the default) samples points uniformly on the marching-cubes
surface; `surface-voxels` and `volume-voxels` sample the mask voxels directly.

## 3. Smooth and simplify

A raw marching-cubes mesh has voxel stair-steps and 100k-1.4M vertices. Smoothing removes
the stairs, and reducing it to about 50k faces makes descriptor runs faster without losing
the legs and antennae:

```bash
python scripts/smooth_simplify_mesh.py \
    --input meshes/bugnist_crickets/bcrick_10_010/preprocessed/bcrick_10_010_thr45_fillholes_keeplargest_ds1.obj \
    --out meshes/bugnist_crickets/bcrick_10_010/smoothed/bcrick_10_010_thr45_fillholes_smooth10_cluster50k.obj \
    --smooth-method taubin --smooth-iterations 10 \
    --target-faces 50000 --decimate-method cluster --final-smooth-iterations 3
```

For the cricket this goes from 120,040 to 24,683 vertices. `cluster` decimation needs no
extra packages; `quadric` keeps the shape a bit better but needs Open3D.

## 4. Put the specimens in a common pose

The cameras in step 6 are placed in world coordinates, so two insects lying differently in
the scanner are seen from different body angles. Rotate each mesh so that the specimens
match roughly (same body axis, same side up). The project did this by hand in Blender:

1. File > Import > Wavefront (.obj), rotate the object until it matches the others.
2. Object > Apply > Rotation (Ctrl+A) so the rotation is written into the vertices.
3. File > Export > Wavefront (.obj) with the default axes, for example as
   `meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj`.

Rotation keeps the vertex order, but it is still a new file: compute descriptors on the
exported OBJ and use that same file for everything that follows. A scripted alternative is

```bash
python scripts/transform_geometry.py --input in.obj --out out.obj --rotate 90 0 0
```

which rotates about the bounding-box centre (X, then Y, then Z; `--order` changes that).

## 5. Check what the cameras see

```bash
python scripts/debug_2d_diffusion_view.py --kind mesh \
    --input meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj --prompt insect \
    --outdir debug/brown_cricket_views --num-views 16 --view-sampling insect \
    --height 512 --width 512 --render-only --all-views
```

`contact_sheet.png` shows the plain render of every camera. The insect should fill a good
part of each image and the views should not repeat. Without `--render-only` the script
runs one view (`--view-index`) through the whole pipeline and saves the ControlNet inputs,
denoising steps, generated image and feature PCA maps; see [descriptors.md](descriptors.md).

## 6. Compute descriptors

```bash
python scripts/compute_mesh_features.py \
    --mesh meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj \
           meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj \
    --prompt "cricket insect, full body, long antennae, large hind legs, segmented insect body, detailed exoskeleton, macro photograph" \
    --outdir output/bugnist_crickets_features_16v_512 \
    --num-views 16 --view-sampling insect --height 512 --width 512 --tolerance 0.008
```

This writes one `<mesh name>_diff3f.pt` per mesh: a float16 tensor with one 2048-D row per
vertex. One prompt is used for all meshes unless you give one per mesh. Useful options:

- `--view-sampling insect` places the cameras on rings around the body with most of them
  near the side, and works with any number of views. `grid` is the original Diff3F layout
  (a square number of views, several of them duplicates); the cricket descriptors in the
  project report used `grid`, the larvae used `insect`.
- `--debug-outdir debug/descriptor_runs/crickets --debug-views all` saves the render,
  ControlNet inputs and generated image of each view of this very run, with a
  `manifest.json`.
- `--skip-existing` skips meshes whose `.pt` already exists, so a batch can be restarted.

Point clouds have their own script with the same options, plus the rendered point size:

```bash
python scripts/compute_pointcloud_features.py \
    --pointcloud pointclouds/bugnist_crickets/bcrick_10_010/bcrick_10_010_thr45_20k.ply \
    --prompt "cricket insect, full body" --outdir output/bugnist_crickets_pointcloud_features \
    --num-views 16 --view-sampling insect --point-radius 0.012
```

## 7. Look at the descriptors

To colour one shape by the first three principal components of its descriptor:

```bash
python scripts/visualize_mesh_features.py \
    --mesh meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj \
    --features output/bugnist_crickets_features_16v_512/BrownCricket_rotated_diff3f.pt \
    --out visualizations/crickets/brown_pca.ply --preview visualizations/crickets/brown_pca.png
```

To compare specimens, fit one PCA on both so that equal colours mean equal descriptors, and
optionally one shared k-means:

```bash
python scripts/visualize_feature_comparison.py \
    --item brownCricket meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj \
           output/bugnist_crickets_features_16v_512/BrownCricket_rotated_diff3f.pt \
    --item blackCricket meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj \
           output/bugnist_crickets_features_16v_512/BlackCricket_10_10_rotated_diff3f.pt \
    --outdir visualizations/crickets/shared_k6 --kmeans 6 --preview
```

The `.ply` files open in Blender, MeshLab or the browser app. `plot_pca_kmeans_scatter.py`
and `plot_cluster_grid.py` draw the same clustering as a feature-space scatter plot and as
one panel per cluster. Details in [analysis.md](analysis.md).

## 8. Match two specimens

```bash
python scripts/compute_feature_correspondences.py \
    --source-name brownCricket --source-mesh meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj \
    --source-features output/bugnist_crickets_features_16v_512/BrownCricket_rotated_diff3f.pt \
    --target-name blackCricket --target-mesh meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj \
    --target-features output/bugnist_crickets_features_16v_512/BlackCricket_10_10_rotated_diff3f.pt \
    --outdir visualizations/crickets/matches --num-source-points 80 --mutual-check
```

matches 80 well-spread source vertices to their most similar target vertices.
`evaluate_correspondence_metrics.py` takes the same arguments (with `--num-samples 300`)
and measures how consistent the matching is without any ground truth: cycle consistency,
how many queries land on the same target, and whether distances are preserved.

## 9. Benchmark against hand-placed landmarks

1. Create a Blender scene with one movable marker per landmark:

   ```bash
   blender --background --python scripts/make_blender_landmark_scene.py -- \
       --geometry meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj --label brownCricket \
       --landmark HEAD_TIP THORAX_CENTER ABDOMEN_TIP LEFT_ANTENNA_BASE RIGHT_ANTENNA_BASE \
                  LEFT_FORELEG_BASE RIGHT_FORELEG_BASE LEFT_HIND_LEG_BASE RIGHT_HIND_LEG_BASE \
       --output landmarks/scenes/brownCricket.blend
   ```

2. Open the `.blend`, move each `LM_<name>` sphere onto the anatomy (move the markers, not
   the mesh) and save. Use the same names on every specimen.
3. Export the markers:

   ```bash
   blender landmarks/scenes/brownCricket.blend --background \
       --python scripts/export_blender_landmarks.py -- \
       --output landmarks/bugnist_crickets/brownCricket_rotated_obj_landmarks.csv
   ```

4. Repeat for the other specimen, then run the benchmark once per direction:

   ```bash
   python scripts/evaluate_landmark_benchmark.py \
       --source-name brownCricket --source-geometry meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj \
       --source-features output/bugnist_crickets_features_16v_512/BrownCricket_rotated_diff3f.pt \
       --source-landmarks landmarks/bugnist_crickets/brownCricket_rotated_obj_landmarks.csv \
       --target-name blackCricket --target-geometry meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj \
       --target-features output/bugnist_crickets_features_16v_512/BlackCricket_10_10_rotated_diff3f.pt \
       --target-landmarks landmarks/bugnist_crickets/blackCricket_rotated_obj_landmarks.csv \
       --outdir visualizations/crickets/landmark_benchmark/brown_to_black
   ```

The landmark CSVs used in the project are in `landmarks/`. To inspect a benchmark in 3D,
use the correspondence viewer of the browser app, `correspondence_to_html_viewer.py` (a
single HTML file), or `make_blender_landmark_benchmark_scene.py`.

## Settings used in the project

| Specimen | Scan | Segmentation | Descriptor mesh (vertices) | Prompt |
| --- | --- | --- | --- | --- |
| brown cricket | `crickets/bcrick_10_010` | threshold 45, auto-crop, fill holes, largest | `BrownCricket_rotated.obj` (24,683) | cricket prompt above |
| black cricket | `crickets/sfaar_10_010` | threshold 45, auto-crop, fill holes, largest | `BlackCricket_10_10_rotated.obj` (24,761) | cricket prompt above |
| mealworm (MEL) | `mel_10_002` | threshold 75, auto-crop, fill holes, largest | `mel_10_002_thr75_smooth10_cluster50k_oriented.obj` (24,856) | *mealworm larva, full body, elongated segmented beetle larva, visible head and body segments* |
| soldier fly larva (SL) | `raw/soldat_10_002` | threshold 28, ROI start 72 60 73 size 358 171 123, fill holes, largest | `soldat_10_002_thr28_smooth10_cluster50k_oriented.obj` (24,275) | *soldier fly larva, full body, elongated segmented fly larva, visible head and body segments* |

All four were smoothed with Taubin (10 iterations), cluster-decimated to 50k faces with 3
final smoothing iterations, rotated in Blender, and described with 16 views at 512 x 512:
the crickets with `--view-sampling grid`, the larvae with `insect` and `--tolerance 0.008`.

Earlier experiments on other individual scans used these crops and thresholds (no hole
filling, full resolution):

| Scan | ROI start (Z Y X) | ROI size (Z Y X) | Threshold | Mesh vertices |
| --- | --- | --- | --- | --- |
| `raw/bcrick_10_001` | 185 30 29 | 180 220 200 | 29 | 107,066 |
| `raw/sfaar_10_001` | 171 37 29 | 313 192 164 | 11 | 192,909 |
| `raw/soldat_10_002` | 72 60 73 | 358 171 123 | 28 | 198,760 |
| `raw/guld_1_002` | 92 58 49 | 333 143 165 | 11 | 1,405,679 |
