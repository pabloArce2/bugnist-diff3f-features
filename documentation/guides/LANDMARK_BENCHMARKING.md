# Landmark Benchmarking Guide

This guide explains how to benchmark Diff3F correspondences using manual
anatomical landmarks placed in Blender.

The goal is:

```text
source landmark on insect A
  |
  v
Diff3F nearest-neighbor match in insect B
  |
  v
compare predicted point to your manual landmark on insect B
```

This gives real evaluation numbers, not only unsupervised diagnostics.

## What You Need First

For each specimen you want to compare, you need:

```text
1. the exact geometry file used for descriptors
2. the matching Diff3F .pt file
3. a manual landmark CSV exported from Blender
```

The geometry and `.pt` file must match row-by-row.

```text
GOOD:
  bcrick_final.obj
  bcrick_final_diff3f.pt

BAD:
  bcrick_final.obj
  bcrick_old_geometry_diff3f.pt
```

If you smooth, decimate, rotate, resample, or re-export the geometry in a way
that changes vertex/point order, compute fresh descriptors before benchmarking.

## Recommended First Landmark Set

Start small. Use landmarks that are visible and repeatable.

Example starter set:

```text
head_tip
thorax_center
abdomen_tip
left_antenna_base
right_antenna_base
left_foreleg_base
right_foreleg_base
left_hind_leg_base
right_hind_leg_base
```

Use exactly the same label names for every insect. The evaluator only compares
labels that exist in both CSV files.

Do not start with too many subtle labels. First prove the benchmark works with
6 to 10 stable points.

## Step 1: Create A Blender Labeling Scene

Run from the repository root:

```powershell
cd C:\Users\pablo\Documents\COPENAGUE\SECOND_SEMESTER\SPETIAL\Diffusion-3D-Features
conda activate diff3f
```

Create a landmark scene for `bcrick`:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python scripts\make_blender_landmark_scene.py `
  -- `
  --geometry meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --label bcrick `
  --landmark head_tip thorax_center abdomen_tip left_antenna_base right_antenna_base left_foreleg_base right_foreleg_base left_hind_leg_base right_hind_leg_base `
  --output landmarks\bugnist_manual\bcrick_landmarks.blend
```

Create a matching scene for `sfaar`:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python scripts\make_blender_landmark_scene.py `
  -- `
  --geometry meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --label sfaar `
  --landmark head_tip thorax_center abdomen_tip left_antenna_base right_antenna_base left_foreleg_base right_foreleg_base left_hind_leg_base right_hind_leg_base `
  --output landmarks\bugnist_manual\sfaar_landmarks.blend
```

Then open each `.blend` normally in Blender.

Move every object named:

```text
LM_<landmark_name>
```

onto the anatomical location. For example:

```text
LM_head_tip
LM_thorax_center
LM_abdomen_tip
```

Save the `.blend` after moving the markers.

Important:

```text
Move the marker objects, not the mesh.
```

The export script records the marker positions relative to the geometry object,
so the coordinates match the original mesh/point-cloud file.

## Step 2: Export Blender Landmarks To CSV

Export `bcrick` landmarks:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background landmarks\bugnist_manual\bcrick_landmarks.blend `
  --python scripts\export_blender_landmarks.py `
  -- `
  --mesh-object bcrick `
  --output landmarks\bugnist_manual\bcrick_landmarks.csv
```

Export `sfaar` landmarks:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background landmarks\bugnist_manual\sfaar_landmarks.blend `
  --python scripts\export_blender_landmarks.py `
  -- `
  --mesh-object sfaar `
  --output landmarks\bugnist_manual\sfaar_landmarks.csv
```

The CSV has at least these columns:

```text
label,x,y,z
```

Those `x,y,z` coordinates are in the local coordinate system of the geometry
object, so the evaluator can map them to the nearest mesh vertex or point-cloud
point.

## Step 3: Run The Landmark Benchmark

Example for `bcrick -> sfaar`:

```powershell
python scripts\evaluate_landmark_benchmark.py `
  --source-name bcrick `
  --source-geometry meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --source-features output\bugnist_clean_features\bcrick_10_001_thr29_roi_keeplargest_ds1_diff3f.pt `
  --source-landmarks landmarks\bugnist_manual\bcrick_landmarks.csv `
  --target-name sfaar `
  --target-geometry meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --target-features output\bugnist_clean_features\sfaar_10_001_thr11_roi_keeplargest_ds1_diff3f.pt `
  --target-landmarks landmarks\bugnist_manual\sfaar_landmarks.csv `
  --outdir visualizations\bugnist_landmark_benchmark\bcrick_to_sfaar
```

Use the descriptor output folder that matches the geometry you are evaluating.
For example, if you recomputed 16-view descriptors into:

```text
output\bugnist_clean_features_hq_16v_512_insect_views
```

then use those `.pt` files instead.

## Benchmark Outputs

The evaluator writes three files:

```text
<source>_to_<target>_landmark_summary.json
<source>_to_<target>_landmark_benchmark.csv
<source>_to_<target>_predicted_matches.csv
```

The summary JSON contains global metrics.

The benchmark CSV contains one row per landmark.

The predicted matches CSV is compatible with the existing correspondence
visualization scripts.

## Opening The Benchmark CSV Without The Numbers Looking Broken

Python/pandas write the CSV using a plain `.` as the decimal point. If Excel
or Google Sheets is set to a locale that expects `,` as the decimal
separator instead (for example Danish, the default around DTU/Copenhagen),
opening the file by double-clicking it can misread a long decimal like
`0.7974902987480164` as one giant integer with thousands grouping inserted,
so a cosine score that should read "about 0.8" instead looks like it is in
the millions. The coordinate columns are not actually in millions; they are
plain local mesh coordinates in the same units you see for that mesh in
Blender.

If a benchmark CSV looks like every number exploded in size after opening
it:

```text
1. Excel: use Data -> Get Data -> From File -> From Text/CSV instead of
   double-clicking the file, so you can set the decimal/thousands
   separators explicitly on import.
2. Google Sheets: File -> Settings -> Locale, set it to "United States" (or
   any "." decimal locale), then re-import the CSV.
3. Or skip the raw CSV and read documentation/results/CRICKET_LANDMARK_BENCHMARK_SUMMARY.md
   instead, which already has this exact brown/black cricket run written out
   as plain, correctly-scaled tables.
```

`evaluate_landmark_benchmark.py` also rounds every number to 6 decimal
digits before writing the CSV/JSON, specifically so a locale mismatch
produces a short, clearly-wrong-looking number instead of a long one that
can pass for a real (if huge) value.

## Benchmark CSV Column Glossary

| Column | Meaning |
| --- | --- |
| `label` | Landmark name, matched between the source and target landmark CSVs. |
| `source_index` | Row index in the source geometry nearest to the manual source landmark. |
| `target_gt_index` | Row index in the target geometry nearest to the manual target landmark (the "ground truth" row). |
| `predicted_target_index` | Row index in the target geometry that Diff3F predicted by nearest-neighbor feature match. |
| `cosine_score` | Feature-space cosine similarity between the source landmark's descriptor and the predicted target row's descriptor. |
| `second_cosine_score` | Cosine similarity of the second-best target match; used to compute `nn_margin`. |
| `nn_margin` | `cosine_score - second_cosine_score`. Small margin means the best and second-best matches were nearly tied, i.e. an ambiguous match. |
| `gt_cosine_score` | Cosine similarity between the source descriptor and the manual target landmark's own descriptor (not the predicted one). |
| `gt_feature_rank` | Where the manual target landmark ranked among all target rows by feature similarity. `1` means it was the single best match; `100` means 99 other rows scored higher. |
| `target_error` | Raw 3-D distance between the predicted point and the manual target landmark, in local mesh units. This is the main metric (see below). |
| `target_error_bbox` | `target_error` divided by the target bounding-box diagonal (`target_bbox_diagonal` in the summary JSON). Comparable across differently-sized insects. |
| `target_error_bbox_pct` | `target_error_bbox * 100`, the same value written as a percentage. |
| `target_vertex_error` | Raw 3-D distance between the predicted point and the *snapped* ground-truth vertex (`target_gt_index`), instead of the raw manual landmark position. Differs from `target_error` only by the snap distance. |
| `target_vertex_error_bbox` | `target_vertex_error` divided by the target bounding-box diagonal. |
| `source_snap_distance` / `target_snap_distance` | Distance from the manual marker to the nearest actual geometry row, for the source/target side respectively. Large values mean the marker was floating away from the mesh surface. |
| `source_landmark_x/y/z`, `target_landmark_x/y/z` | The raw manual marker positions placed in Blender, in local mesh coordinates. |
| `source_x/y/z` | The actual source geometry row position at `source_index` (the snapped source point). |
| `target_gt_x/y/z` | The actual target geometry row position at `target_gt_index` (the snapped ground-truth point). |
| `pred_target_x/y/z` | The actual target geometry row position at `predicted_target_index` (where Diff3F guessed). |
| `pck_0.01`, `pck_0.02`, `pck_0.05`, `pck_0.1` | Whether `target_error_bbox` fell within that threshold (`True`/`False`), one column per `--thresholds` value. |

All coordinate columns share the same local mesh coordinate system as the
`.obj`/`.ply` geometry file, so they line up with what you see for that same
mesh in Blender.

## Main Metrics

### Target Error

```text
distance from predicted target point to your manual target landmark
```

This is the main metric.

### Normalized Target Error

```text
target_error_bbox = target_error / target_bbox_diagonal
```

This makes errors comparable across insects of different size.

Example:

```text
target_error_bbox = 0.05
```

means:

```text
the prediction is 5 percent of the target bounding-box diagonal away from the manual landmark
```

### PCK

PCK means percentage of correct keypoints.

The script reports:

```text
PCK@1%
PCK@2%
PCK@5%
PCK@10%
```

For example:

```text
pck_at_0.05_bbox = 0.60
```

means:

```text
60 percent of landmarks landed within 5 percent of the target bounding-box diagonal
```

### Ground-Truth Feature Rank

For each source landmark, the script also checks where your manual target
landmark ranked in feature space.

```text
gt_feature_rank = 1
```

means the manual target landmark was the best feature match.

```text
gt_feature_rank = 100
```

means 99 target points had higher feature similarity than your manual landmark.

This is useful because nearest-neighbor prediction can be spatially close but
not exactly the same vertex.

### Snap Distance

Manual landmarks are arbitrary 3D positions. The `.pt` file only has rows for
mesh vertices or point-cloud points.

So the evaluator first snaps each manual landmark to the nearest geometry row.

The CSV reports:

```text
source_snap_distance
target_snap_distance
```

Large snap distances mean the manual marker was far from the actual geometry,
or the point cloud is too sparse.

## Step 4: Visualize Manual Vs Predicted Landmarks

Create a Blender scene showing:

```text
blue marker   = source manual landmark
green marker  = target manual landmark
red marker    = Diff3F predicted target point
red line      = prediction error on the target
teal line     = source landmark to predicted target point
```

Command:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
  --background `
  --python scripts\make_blender_landmark_benchmark_scene.py `
  -- `
  --source-label bcrick `
  --source-geometry meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --target-label sfaar `
  --target-geometry meshes\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1.obj `
  --benchmark-csv visualizations\bugnist_landmark_benchmark\bcrick_to_sfaar\bcrick_to_sfaar_landmark_benchmark.csv `
  --output visualizations\bugnist_landmark_benchmark\bcrick_to_sfaar\bcrick_to_sfaar_landmark_benchmark.blend
```

Open the `.blend` to inspect the result.

The scene sorts landmarks by largest error first, so the worst failures are
easier to inspect.

## How To Interpret Results

A good result would look like:

```text
low median target_error_bbox
high PCK@5%
high PCK@10%
manual target landmarks often ranked near the top in feature space
prediction markers land near the green manual markers in Blender
```

A weak result would look like:

```text
high target_error_bbox
low PCK@5%
large gt_feature_rank values
red predicted markers consistently far from green manual markers
many predictions collapse to the same anatomical area
```

Do not rely on cosine similarity alone. A match can have a decent cosine score
and still be anatomically wrong.

## Recommended Experiment Order

Start with this sequence:

```text
1. bcrick -> sfaar
2. sfaar -> bcrick
3. bcrick -> soldat
4. sfaar -> soldat
```

Run both directions when possible. If a pair only works in one direction, the
descriptor space is probably unstable or ambiguous.

For each pair, compare:

```text
4-view descriptors vs 16-view descriptors
generic prompt vs cricket/insect-specific prompt
original mesh vs smoothed mesh
mesh descriptors vs point-cloud descriptors
```

Change only one thing at a time.

## The Benchmark In One Sentence

Manual landmarks let us ask:

```text
When Diff3F says this source point matches that target point,
how close is it to the anatomical point Pablo manually labeled?
```

That is the cleanest next metric for this project.

