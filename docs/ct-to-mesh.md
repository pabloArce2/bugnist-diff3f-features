# From a CT volume to a mesh

Diff3F works on surfaces, but a BugNIST scan is a block of voxels. This page describes how
one insect is separated from everything else in the scan and turned into a clean,
light mesh (or a point cloud). It is classical image processing, no learning involved.
The code is in `bugnist_tools/ct.py` and the scripts `preview_tif_volume.py`,
`bugnist_tif_to_mesh.py`, `bugnist_tif_to_pointcloud.py` and `smooth_simplify_mesh.py`.

```text
TIFF volume -> crop -> threshold -> clean the mask -> keep the largest piece
            -> marching cubes -> smooth + simplify -> rotate -> descriptor mesh
```

## What is in a scan

Each file is a 512 x 256 x 256 array of 8-bit values, indexed (Z, Y, X) with Z along the
long axis. Bright voxels are dense material, mostly the insect's cuticle; dark voxels are
air. In between there is soft tissue, the cotton or foam that holds the specimen in the
tube, and some reconstruction noise. The job is to keep the insect and nothing else.

## 1. Look before choosing anything

`preview_tif_volume.py` saves, per scan:

- `_orthos.png`: the centre slice along each axis,
- `_mips.png`: maximum-intensity projections. The brightest voxel along each line of
  sight, which shows the whole insect at once and where it sits in the volume,
- `_slices_z.png` (and y, x): evenly spaced slices,
- `_overlay_...png`: the same images with every voxel above the threshold in red,
- `_hist.png`: the intensity histogram (log scale) with the threshold marked,
- `_summary.txt`: shape, intensity range and the threshold used.

Run it a few times with different `--threshold` values and compare the overlays. A good
threshold covers the legs and antennae completely without painting the cotton red.

## 2. Crop

Cropping removes material that has nothing to do with the insect (the tube, packing, the
edge of the field of view) and makes every later step faster. There are three ways:

- `--roi-start Z Y X --roi-size Z Y X`: a box in voxel indices, read off the previews.
  For example `--roi-start 185 30 29 --roi-size 180 220 200` keeps Z 185-364, Y 30-249,
  X 29-228. Leave a margin of a few voxels so no leg tip is cut off.
- `--auto-crop`: threshold the volume once, take the bounding box of the result and add
  `--auto-crop-padding` voxels on each side. This works when the insect is the only large
  bright object, which was the case for the cricket scans.
- `--centroids file.csv --centroid-index N --crop-size Z Y X`: a box around one specimen of
  a BugNIST mixture, using the centroid files that come with those volumes.

If you prefer to pick the box in 3D Slicer: the Red (axial) slider moves along Z, Green
(coronal) along Y and Yellow (sagittal) along X. Slicer shows positions in millimetres,
while the scripts want voxel indices, so do not paste the millimetre values.

The crop offset is remembered, so the final mesh is still in the coordinates of the full
scan.

## 3. Threshold

Every voxel brighter than the threshold becomes insect (`--invert` for the opposite). There
are three ways to choose it (`--threshold-method`):

| Method | How | When |
| --- | --- | --- |
| `manual` | `--threshold 45` | the normal case: pick it from the histogram and overlays |
| `percentile` | `--threshold-percentile 95` | a quick first guess; the value moves whenever the crop changes |
| `otsu` | automatic | a starting point only |

`auto` (the default) uses the manual value if given, then the percentile, then Otsu.

Otsu's method picks the threshold that best splits the histogram into two groups. A scan
has more than two materials, so it can end up separating cotton from air instead of insect
from everything else. On the cropped `bcrick_10_001` it chose 73, far stricter than the 29
that segmented the cricket well. The thresholds used in the project were between 11 and 75,
depending on the scan (see the table at the end of [pipeline.md](pipeline.md)).

## 4. Clean the mask

After thresholding, the mask still has specks of noise, bits of cotton and sometimes holes.
The cleaning steps run in this order:

1. `--min-size 512`: remove connected pieces smaller than 512 voxels.
2. `--opening-radius R` (off by default): erode then dilate with a ball of radius R. This
   breaks thin bridges, for example between a leg and a cotton fibre, but it can also
   remove antennae, so use 1 at most.
3. `--closing-radius R` (off by default): dilate then erode, closing small gaps in the
   cuticle.
4. `--fill-holes`: fill cavities that are completely enclosed. Without it the air inside
   the body produces a second, inner surface, which gets rendered and described too.
5. `--keep-largest`: label the connected components and keep only the biggest one. This
   is the step that finally separates the insect from any remaining debris.

The result is a binary mask: 1 inside the insect, 0 elsewhere.

## 5. Extract the surface

`bugnist_tif_to_mesh.py` runs marching cubes (scikit-image) on the mask at level 0.5, the
boundary between 0 and 1. The vertices are shifted by the crop offset, scaled back if
`--downsample` was used, reordered from (Z, Y, X) to (X, Y, Z) and saved as OBJ or PLY.
`--center` moves the centroid to the origin.

`--downsample N` keeps every Nth voxel before all of this. It is handy for fast trials
(`--downsample 3` meshes a scan in seconds), but final meshes were made at full resolution.

For point clouds, `bugnist_tif_to_pointcloud.py` does the same crop, threshold and cleaning,
then samples `--num-points` points:

- `mesh-surface` (default): uniform samples on the marching-cubes surface,
- `surface-voxels`: centres of the voxels on the boundary of the mask, jittered by up to
  half a voxel,
- `volume-voxels`: centres of all mask voxels, including the inside (rarely useful, since
  Diff3F only sees the surface).

`--seed` makes the sampling repeatable.

## 6. Smooth and simplify

The raw surface follows the voxel grid, so it is covered in small stair-steps, and it is
large: 100k-200k vertices for most specimens and 1.4M for `guld_1_002`. The benchmark
shapes Diff3F was developed on are clean and much lighter.

`smooth_simplify_mesh.py` does, in order:

1. clean-up: drop unreferenced vertices and degenerate faces,
2. Taubin smoothing (`--smooth-iterations 10`). Each iteration moves every vertex towards
   the average of its neighbours and then slightly back, which removes the stairs without
   shrinking thin parts the way plain Laplacian smoothing does,
3. decimation to `--target-faces`:
   - `cluster` merges all vertices that fall into the same cell of a regular grid. The
     cell size is found by bisection so that the face count ends just under the target.
     No extra dependencies, but the count is approximate.
   - `quadric` uses Open3D's quadric error simplification, which keeps the shape a little
     better. `auto` tries it and falls back to `cluster`.
4. a few more Taubin iterations (`--final-smooth-iterations 3`) to even out the new
   triangles.

The project settled on about 50k faces. On `bcrick_10_001` (107,066 vertices, 215,952
faces) this gave 24,148 vertices and 49,977 faces; the head, thorax, abdomen, legs and
antennae all survive, and the descriptor looks the same at 50k, 75k and full resolution
while a single view is processed 40-55 % faster.

## 7. Orient

The last step before descriptors is rotating each specimen into a common pose, because
the camera layouts are defined in world coordinates. See [pipeline.md](pipeline.md#4-put-the-specimens-in-a-common-pose).

## A word on vertex order

Descriptors are stored per vertex, in file order. Anything that changes the vertex list
(smoothing, decimation, re-meshing, Blender import/export with merging) makes the old
descriptor useless for the new file. Rotations and translations keep the order, but the
rule of thumb is simple: compute the descriptor on the final file and use that same file
everywhere afterwards.

## Troubleshooting a segmentation

| What you see | What to try |
| --- | --- |
| cotton or packing attached to the legs | raise the threshold a little, or `--opening-radius 1` |
| legs or antennae missing or broken into pieces | lower the threshold; check that `--keep-largest` did not drop them |
| a second surface inside the body | `--fill-holes` |
| several insects or large debris left | a tighter ROI, then `--keep-largest` |
| the mesh is huge and slow | `--downsample 2` for tests; `smooth_simplify_mesh.py` for descriptors |
