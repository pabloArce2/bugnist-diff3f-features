# Interactive 3-D Correspondence Viewer

`correspondence_to_html_viewer.py` creates one self-contained HTML file with
two real triangle meshes, the landmark-benchmark predictions between them,
and interactive point inspection. The result opens in a normal browser and
does not need a web server or internet connection.

## What The Viewer Shows

The source and target meshes are placed side by side. For every row in the
supplied benchmark CSV, the viewer can show:

```text
source marker -> predicted target marker   Diff3F correspondence
manual target marker -> prediction         target error
```

Use the landmark selector to focus on one of the supplied rows and inspect its
cosine score, prediction error, normalized error, and vertex indices. The layer
controls can show all correspondence lines or isolate the selected landmark.

For a regular correspondence export, use `--matches` instead of
`--benchmark-csv`. Source-to-target lines and available scores are shown, but
manual target markers and target-error lines are unavailable because that CSV
does not contain ground truth.

The mesh placement is only a display transform. Coordinates shown in the
selection panel and copied as JSON remain in each input geometry's original
local coordinate system.

## September 4 Cricket Results

The quickest entry point for the already generated pair is:

```text
visualizations\bugnist_crickets_landmark_benchmark\updated_points_2026-09-04\index.html
```

Open it by double-clicking; it links both correspondence directions. The
commands below reproduce those two viewer files from the current meshes and
benchmark CSVs.

Activate the project environment first:

```powershell
conda activate diff3f
```

Create the black-to-brown viewer:

```powershell
python scripts\correspondence_to_html_viewer.py `
  --source-label blackCricket `
  --source-geometry meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj `
  --target-label brownCricket `
  --target-geometry meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj `
  --benchmark-csv visualizations\bugnist_crickets_landmark_benchmark\updated_points_2026-09-04\blackCricket_to_brownCricket\blackCricket_to_brownCricket_landmark_benchmark.csv `
  --out visualizations\bugnist_crickets_landmark_benchmark\updated_points_2026-09-04\blackCricket_to_brownCricket\blackCricket_to_brownCricket_interactive.html
```

Create the brown-to-black viewer:

```powershell
python scripts\correspondence_to_html_viewer.py `
  --source-label brownCricket `
  --source-geometry meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj `
  --target-label blackCricket `
  --target-geometry meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj `
  --benchmark-csv visualizations\bugnist_crickets_landmark_benchmark\updated_points_2026-09-04\brownCricket_to_blackCricket\brownCricket_to_blackCricket_landmark_benchmark.csv `
  --out visualizations\bugnist_crickets_landmark_benchmark\updated_points_2026-09-04\brownCricket_to_blackCricket\brownCricket_to_blackCricket_interactive.html
```

Open either generated `.html` file by double-clicking it. All geometry and
benchmark data are embedded in the file.

## Controls

- Drag anywhere in the 3-D scene to orbit the view.
- Hold Shift while dragging, or right-drag, to pan.
- Use the mouse wheel or trackpad scroll to zoom.
- Adjust **mesh separation** to separate or bring together the two specimens.
- Switch between **mesh** and **points** rendering when surface detail or
  individual vertices are more useful.
- Toggle correspondence lines, target-error lines, and manual ground-truth
  markers independently.
- Choose a landmark and enable **isolate selected** to untangle overlapping
  lines.
- Click a mesh surface to inspect the exact surface hit and its nearest original
  vertex. A compact readout stays visible over the 3-D scene, with full details
  and **Copy JSON** in the sidebar.
- To inspect a known row directly, choose source or target, enter its zero-based
  vertex index, and click **Go**. **Inspect a random vertex** is useful for
  quickly exploring the mesh without aiming at a small surface region.

If dragging moves the camera, it is treated as navigation rather than a point
selection.

## Understanding A Click

A mesh click reports two related positions:

- **Surface hit** is the exact point where the viewing ray intersects a
  triangle. It may lie between vertices.
- **Nearest vertex** is the closest original mesh vertex in 3-D, searched over
  the whole selected mesh. Its index is the original geometry row index used by
  the Diff3F feature tensor and benchmark CSV, and its XYZ value is copied from
  the original mesh.

This distinction is useful when checking rows such as `source_index`,
`predicted_target_index`, and `target_gt_index`. A random surface click is an
inspection tool only: it does **not** run Diff3F, search the feature tensors, or
create a new correspondence. Correspondence lines are limited to the rows in
the CSV passed with `--benchmark-csv` (or the simpler correspondence CSV passed
with `--matches`).

## Benchmark Line Semantics

For a landmark row:

- the correspondence starts at `source_x/y/z` (`source_index`),
- its prediction endpoint is `pred_target_x/y/z`
  (`predicted_target_index`), and
- the error line joins that prediction to the raw manual annotation at
  `target_landmark_x/y/z`.

The raw manual target is deliberately used for the error line. The snapped
`target_gt_x/y/z` vertex remains in the source CSV and embedded match data, but
using it as the visual endpoint would hide the landmark-to-mesh snap distance.

## Input Requirements And Troubleshooting

The source and target geometry can be OBJ or PLY triangle meshes. Point-only
PLY, NPY, XYZ, and text geometry are also accepted; use **Vertices / points**
mode when no faces are available. Keep vertex order unchanged after computing
the feature files: CSV indices refer to that exact order. The generator
validates required CSV columns and checks that every referenced index is inside
its geometry.

If generation fails, read the command-line error; the usual cause is a
mismatched or reordered geometry/CSV pair. If a point-only input looks blank,
switch from **Mesh surfaces** to **Vertices / points**. If lines look crowded,
select one landmark and enable isolation instead of reducing the underlying
data.

## Existing Single-Point-Cloud Viewer

`pointcloud_to_html_viewer.py` now also supports click inspection. It reports
the original geometry row index, original XYZ, and RGB even when `--max-points`
subsamples the display. Previously generated HTML files do not contain this new
interaction; rerun the same viewer command to refresh them, for example:

```powershell
python scripts\pointcloud_to_html_viewer.py `
  --pointcloud visualizations\my_colored_points.ply `
  --out visualizations\my_colored_points.html
```

This only rebuilds the HTML wrapper. It does not recompute Diff3F features.
