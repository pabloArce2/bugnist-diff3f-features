# Working with descriptors

Everything here runs on the CPU (the matching scripts use the GPU if there is one; pass
`--device cpu` to avoid it). All scripts need the geometry file the descriptor was computed
from.

## Colouring a shape by its descriptor

`visualize_mesh_features.py` / `visualize_pointcloud_features.py` project the 2048-D rows
onto their first three principal components and use them as RGB. Nearby colours mean
similar descriptors. The output is a PLY with vertex colours plus an optional PNG preview.

Colours from two separate runs are **not** comparable: each shape gets its own PCA axes.
To compare specimens use `visualize_feature_comparison.py`, which fits a single PCA on
samples from all `--item`s (8000 rows each by default) and colours every item with it.

```bash
python scripts/visualize_feature_comparison.py \
    --item brownCricket <mesh> <features.pt> --item blackCricket <mesh> <features.pt> \
    --outdir visualizations/crickets/shared_pca --preview
```

Each item can be a mesh or a point cloud. The PCA is exact and deterministic: the same
input always gives the same colours. (Older runs used a randomised PCA without a fixed seed,
so their third colour channel could change between runs; regenerate old PLYs if you need
them to match new ones.)

In Blender, a PLY with vertex colours imports grey until the material uses them: add an
*Attribute* node named `Col` (or `Color`) and connect it to the Base Color of the Principled
BSDF, then switch the viewport to Material Preview.

## Shared k-means clusters

Add `--kmeans K` to the comparison script to fit one k-means model on samples from all
items and label every vertex with it. Cluster 3 then means the same group of descriptors on
every specimen. Per item it writes `_shared_kmeans_k<K>_clusters.ply` (flat colours),
`_labels.npy` (one integer per vertex), and with `--preview` a PNG; once per run a legend
with the cluster sizes, and `manifest.txt` with all the settings.

- `--cluster-on features` (default) clusters the full L2-normalised 2048-D descriptors.
  `--cluster-on pca` clusters the 3-D PCA coordinates instead, so the clusters match the
  visible colours, but it can only find structure that survives the compression to three
  numbers.
- There is no natural K. A cricket is one connected surface, not separate blobs, and a sweep
  of K from 3 to 16 showed no clear elbow in the k-means inertia or the silhouette score.
  K=6 gives body regions, K=10 starts separating appendages. Much larger K (100, 500)
  fragments the two specimens differently instead of finding finer anatomy.
- To run several K, run the script once per K with its own `--outdir`.

Two more views of the same clustering (same items, K and seed give the same cluster ids):

- `plot_pca_kmeans_scatter.py --item NAME FEATURES ... --kmeans K`: the samples in the
  PC1/PC2, PC1/PC3 and PC2/PC3 planes, coloured by cluster, with centroids and a dashed
  2-sigma ellipse per cluster, and one marker shape per item. If the clusters of the two
  specimens overlap in these plots, the descriptor puts their body parts in the same place.
- `plot_cluster_grid.py --item NAME GEOMETRY FEATURES ... --kmeans K`: one image per item
  with K panels, each showing one cluster in colour on the rest of the shape in grey.

`make_contact_sheet.py` tiles any of these PNGs into a labelled grid
(`--cell ROW COLUMN image.png`, repeated).

## Nearest-neighbour correspondences

`compute_feature_correspondences.py` picks `--num-source-points` vertices on the source
(farthest-point sampling by default, `--sampling random`, or your own list with
`--source-indices`) and finds, for each, the target vertex with the highest cosine
similarity. `--mutual-check` also searches back from each target vertex and flags the
matches that return to the same source vertex. Output: `<source>_to_<target>_matches.csv`
(indices, cosine score, coordinates on both shapes) and the same as `.npz`.

This rule looks at every source point on its own, so neighbouring points can map to the
same target and a high score does not guarantee the right anatomy. The next two scripts
measure how much to trust it.

## Diagnostics without ground truth

`evaluate_correspondence_metrics.py` (same arguments, `--num-samples 300`) matches source to
target and back, and writes `<pair>_metrics.json` and a per-point CSV:

| Metric | Meaning | Better |
| --- | --- | --- |
| `cosine_median` | similarity of the chosen matches | higher, but can be high for wrong matches |
| `nn_margin_median` | best score minus runner-up score | higher: the match is less ambiguous |
| `cycle_within_5pct_bbox`, `_10pct_` | fraction of points that come back within 5 / 10 % of the source bounding-box diagonal after source -> target -> source | higher |
| `mutual_exact_ratio` | fraction that come back to exactly the same vertex | higher; very strict on dense meshes |
| `unique_target_ratio`, `max_target_reuse` | how many different targets were used, and the most queries sent to one target | high ratio, low reuse |
| `pairwise_distance_spearman` | rank correlation between distances among the queries and distances among their matches | higher: the layout is preserved |

## Landmark benchmark

The real test: place the same named landmarks on two specimens by hand and check where the
descriptor sends each source landmark ([pipeline.md](pipeline.md#9-benchmark-against-hand-placed-landmarks)
shows the Blender steps). `evaluate_landmark_benchmark.py`:

1. reads both landmark CSVs (`label,x,y,z`) and keeps the labels present in both,
2. snaps each landmark to its nearest vertex (the snap distance is reported),
3. takes the source vertex's descriptor, finds the most similar target vertex,
4. measures the distance from that prediction to the manual target landmark.

Outputs, per direction:

- `<source>_to_<target>_landmark_benchmark.csv`: one row per landmark with the three vertex
  indices (source, snapped target, predicted target), all coordinates, the cosine score and
  margin, the error in mesh units and as a fraction of the target's bounding-box diagonal,
  the rank of the true target among all target vertices by similarity, and one `pck_<t>`
  column per threshold,
- `..._landmark_summary.json`: PCK at each threshold (`--thresholds 0.01 0.02 0.05 0.10`,
  as fractions of the diagonal), mean/median error, and how often the true target ranked in
  the top 5, 10, 100 and 1000,
- `..._predicted_matches.csv`: the predictions in the matches format, for the viewers.

Matching is not symmetric, so run both directions. "Thorax centre" and similar labels are
regions rather than points, so a single-vertex error there is partly a matter of where the
marker was placed.

## Looking at matches in 3D

- **Browser app** ([Diff3f_App](https://github.com/pabloArce2/Diff3f_App)): the *Correspondences* view loads benchmark
  results listed in a `correspondence_presets.json` file anywhere under `visualizations/`.
  Its documentation describes the format.
- **Stand-alone HTML**: `correspondence_to_html_viewer.py --source-geometry ... --target-geometry
  ... --benchmark-csv ...` (or `--matches` for a matches CSV) writes one file with both
  shapes and all lines embedded; it opens by double-clicking. It refuses a geometry file
  whose vertices do not match the coordinates in the CSV.
  `pointcloud_to_html_viewer.py` does the same for a single coloured point cloud.
- **Blender**: `make_blender_correspondence_scene.py` (matches CSV) and
  `make_blender_landmark_benchmark_scene.py` (benchmark CSV) build `.blend` scenes with the
  two shapes side by side, `make_blender_feature_comparison.py` places coloured PLYs next to
  each other, and `pointcloud_to_blender_splats.py` turns a coloured point cloud into small
  coloured octahedra, because Blender does not display colours on a point cloud without
  faces.

## Checking a descriptor file

```bash
python scripts/inspect_feature_quality.py --geometry mesh.obj --features mesh_diff3f.pt
```

prints NaN/zero-row counts, the spread of row norms, and the cosine similarity of
neighbouring vertices against random pairs. A useful descriptor is smooth over the surface,
so `local_minus_random_median` should be well above zero. `--compare-features other.pt`
compares two descriptors of the same geometry row by row, for example two prompts or two
view layouts.
