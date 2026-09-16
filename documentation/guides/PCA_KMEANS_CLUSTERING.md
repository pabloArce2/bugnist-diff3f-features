# Shared-PCA K-Means Clustering Guide

This guide explains the shared K-means clustering path added on top of the
existing shared-PCA feature comparison scripts.

The starting point is the existing shared-PCA visualization:

```text
2048-D Diff3F descriptor per vertex/point
        |
        v
one shared PCA basis, fit across all compared items
        |
        v
3-D color coordinate per vertex/point
        |
        v
RGB color (continuous shared_pca_features.ply)
```

The new path adds a discrete step on top of that same shared space:

```text
3-D shared PCA coordinate (or the full normalized descriptor)
        |
        v
one shared K-means model, fit across all compared items
        |
        v
cluster id 0..K-1 per vertex/point
        |
        v
flat palette color per cluster (shared_kmeans_kK_clusters.ply)
```

"Shared" matters here the same way it matters for the PCA colors: one K-means
model is fit once, using samples pooled from every `--item`, and then every
item is labeled with that same model. That makes cluster id `3` on one insect
and cluster id `3` on another insect at least come from the same region of
feature space, which is what makes the cluster maps comparable across
specimens.

## What You Need First

Exactly the same requirement as the existing shared-PCA comparison:

```text
1. the exact mesh/point cloud used to compute each .pt file
2. the matching Diff3F .pt file, same row order
3. at least two --item entries (clustering across a single item still works,
   but the point of "shared" is comparison)
```

If you are not already familiar with the shared-PCA comparison itself, read
`documentation/information/DIFF3F_DESCRIPTORS_EXPLAINED.md` first (see
"Single PCA Vs Shared PCA").

## Where This Lives

No new script was added. The clustering path is optional flags on the two
existing comparison scripts:

```text
scripts/visualize_feature_comparison.py             mesh version
scripts/visualize_pointcloud_feature_comparison.py   point-cloud version
```

Both scripts behave exactly as before when `--kmeans` is omitted. Everything
below is additive.

## New Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--kmeans K` | off | Fit one shared K-means with `K` clusters and export a discrete cluster-colored PLY per item. Omit to keep the old PCA-only behavior. |
| `--cluster-on {pca,features}` | `features` | `features` clusters the full normalized 2048-D descriptor -- the standard choice. `pca` clusters the 3-D shared PCA coordinate used for the RGB colors instead. |
| `--cluster-sample-per-item N` | `--fit-sample-per-item` | Rows sampled per item to fit the shared K-means. |
| `--preview` | off | Also render a flat PNG for the continuous PCA colors, and for the cluster colors if `--kmeans` is set. |
| `--preview-max-points`, `--preview-size`, `--elev`, `--azim` | same as the single-item visualizers | Preview camera/rendering controls. |

`--cluster-on features` (the default) clusters the full 2048-D descriptor
before any compression to 3 dimensions. This is the standard approach: PCA's
top-3 components were chosen to make a good RGB image, not to preserve
whatever structure matters for clustering, so throwing away 2045 of the 2048
dimensions before clustering is usually the wrong order of operations.
`--cluster-on pca` clusters the same 3 numbers that get mapped to
red/green/blue instead, so its cluster boundaries are a direct
discretization of the colors you already see in the continuous PLY, and it
tends to keep cluster ids more consistently aligned across specimens because
there is far less room for two individuals' shapes to disagree in a 3-D
space than in a 2048-D one (see the trade-off writeup in
`documentation/results/CRICKET_PCA_CLUSTERING_SUMMARY.md`, section
"PCA-Space Vs Feature-Space Clustering"). Both are legitimate; `features` is
the default because it is the more defensible choice, not because `pca` is
wrong.

## Example: Mesh Comparison With Clustering

```powershell
python scripts\visualize_feature_comparison.py `
  --item brownCricket meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt `
  --item blackCricket meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt `
  --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6 `
  --kmeans 6 `
  --preview
```

## Example: Point-Cloud Comparison With Clustering

```powershell
python scripts\visualize_pointcloud_feature_comparison.py `
  --item bcrick pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply output\bugnist_pointcloud_features\bcrick_10_001_thr29_roi_keeplargest_ds1_20k_diff3f.pt `
  --item sfaar pointclouds\bugnist_individual\sfaar_10_001\preprocessed\sfaar_10_001_thr11_roi_keeplargest_ds1_20k.ply output\bugnist_pointcloud_features\sfaar_10_001_thr11_roi_keeplargest_ds1_20k_diff3f.pt `
  --outdir visualizations\bugnist_pointcloud_features\shared_pca_kmeans_bcrick_sfaar `
  --kmeans 8 `
  --preview
```

## Output Files

For every `--item name ...`, in addition to the existing
`{name}_shared_pca_features.ply`:

```text
{name}_shared_kmeans_k{K}_clusters.ply    flat-colored PLY, one solid color per cluster
{name}_shared_kmeans_k{K}_labels.npy      integer cluster id per vertex/point, same row order as the geometry
```

If `--preview` is set, also:

```text
{name}_shared_pca_features.png            continuous PCA color preview
{name}_shared_kmeans_k{K}_clusters.png    cluster color preview
```

Once per run, if `--kmeans` is set:

```text
shared_kmeans_k{K}_legend.png   swatch per cluster id with the per-item point/vertex count and percentage
manifest.txt                    now also records kmeans_k, cluster_on, cluster_seed, the palette RGB values,
                                 and per-item cluster_sizes
```

The `_labels.npy` files are the part worth keeping if you plan to do anything
downstream with the clusters (for example, restricting correspondence search
to same-cluster vertices only). The PLY and PNG files are for looking at the
result.

## Choosing K

There is no single "correct" K here, and the numbers back that up. A cricket
mesh is one connected surface, not a set of separated blobs, so K-means on it
does not have a sharp natural cluster count the way it might for genuinely
separated point clusters.

A silhouette/inertia sweep on the brown/black cricket shared-PCA space (see
`documentation/results/CRICKET_PCA_CLUSTERING_SUMMARY.md` for the full table)
showed silhouette score falling off smoothly from `K=3` through `K=16`, with
no clear elbow. In practice, treat `K` as a granularity dial:

```text
K = 3-4   very coarse: roughly separates a couple of broad surface regions
K = 5-8   coarse anatomical regions (a good starting point)
K = 9-14  finer regions; can start separating individual appendages
K > 15    starts fragmenting into small, less interpretable patches
```

Start around `K=6`, look at the preview PNGs and the legend, then raise or
lower `K` depending on whether the regions look too coarse or too fragmented.

## Running Several K Values

There is no `--kmeans` sweep flag; run the command once per `K`, each with
its own `--outdir` so the runs do not overwrite each other. In PowerShell,
loop over a list of `K` values instead of retyping the command:

```powershell
$Ks = 4, 6, 8, 10, 12
foreach ($k in $Ks) {
  python scripts\visualize_feature_comparison.py `
    --item brownCricket meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt `
    --item blackCricket meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt `
    --outdir "visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k$k" `
    --kmeans $k `
    --preview
}
```

The same pattern works for the other cluster-aware scripts; just swap the
inner command:

```powershell
foreach ($k in $Ks) {
  python scripts\plot_pca_kmeans_scatter.py `
    --item brownCricket output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt `
    --item blackCricket output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt `
    --kmeans $k `
    --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black
}

foreach ($k in $Ks) {
  python scripts\plot_cluster_grid.py `
    --item brownCricket meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt `
    --item blackCricket meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt `
    --kmeans $k `
    --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\cluster_grid
}
```

Notes:

```text
$k inside a double-quoted string interpolates automatically (as in
"...\k$k"), but PowerShell needs $($k) instead of $k directly next to word
characters that would otherwise be parsed as part of the variable name.

Every one of these scripts refits its own shared PCA/K-means from scratch
(same seed=42 default), so results are fully reproducible without needing to
save or pass the fitted model between runs.

If you only want the cluster counts/percentages to compare granularities
without regenerating every PLY/PNG, read the "cluster_sizes" line each run
appends to its outdir's manifest.txt instead of rerunning with --preview
every time.
```

## Interpreting Cluster Maps

A shared cluster id being spatially consistent across two items (for example,
always landing on the legs, or always on one side of the thorax) is evidence
that the descriptor field captures some structure that generalizes across
specimens. That is a genuinely useful qualitative signal, in the same spirit
as the shared-PCA colors.

It is not the same claim as anatomical correspondence, and it is not
evaluated the same way. Two important caveats carried over from the rest of
the descriptor work:

```text
Cluster ids are unordered and arbitrary. K-means does not know that cluster 3
is "legs"; it only knows that cluster 3's points are close together in the
clustering space. Read the color maps, do not assume the numeric id means
anything on its own.

A cluster boundary is not a landmark. This tool produces a coarse region
map, not point-to-point correspondence. For point-level evaluation, use
the landmark benchmark (documentation/guides/LANDMARK_BENCHMARKING.md).
```

`--cluster-on pca` clusters are also bounded by whatever the 3-D PCA
projection kept or discarded; a K-means fit on `--cluster-on features`
instead can disagree with the visible RGB gradient because it sees
directions in the 2048-D space that PCA's top 3 components did not keep.
Treat a difference between the two as a hint about how much the 3-D
compression is losing, not as one of them being "wrong."

## Feature-Space Scatter Plot (The "Centroids" View)

Everything above colors the mesh/point cloud itself: the cluster id becomes a
region on the 3-D surface. That is one useful view, but it does not directly
show the thing K-means actually operates on: points scattered in the 3-D
shared-PCA space, grouped into blobs around centroids. That is the second
view, and it is what `scripts/plot_pca_kmeans_scatter.py` draws.

Since the shared PCA space here is 3-D (not 2-D), the script plots all three
pairwise projections as a small grid of scatter panels:

```text
PC1 vs PC2   PC1 vs PC3   PC2 vs PC3
```

If it is not obvious what "PC1", "PC2", and a "PC1 vs PC2" panel actually
mean, see `documentation/information/DIFF3F_DESCRIPTORS_EXPLAINED.md` under
"What Do PC1, PC2, PC3 Actually Mean?" for the plain-language version before
reading the rest of this section.

Each panel shows sampled points from every `--item`, colored by the same
shared cluster id and palette used for the on-mesh cluster PLYs (so a color
in this plot means the same cluster as that color on the mesh), a dashed
ellipse per cluster (mean +/- `--ellipse-std` standard deviations), and the
K-means centroids marked with a black `x`. Different items get different
marker shapes (circle, triangle, ...), so you can see whether two specimens'
points actually land in the same blobs, not just whether the blobs exist.

This script only needs the `.pt` feature files, no mesh or point cloud, since
it never touches geometry:

```powershell
python scripts\plot_pca_kmeans_scatter.py `
  --item brownCricket output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt `
  --item blackCricket output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt `
  --kmeans 6 `
  --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black
```

It refits the same shared PCA and shared K-means internally (same seed,
same `--fit-sample-per-item` default of 8000), so with default settings the
cluster ids and colors match a `visualize_feature_comparison.py` run on the
same items exactly. Output:

```text
pca_kmeans_scatter_k{K}.png            the 3-panel scatter grid
pca_kmeans_scatter_k{K}_manifest.txt   items, K, seed, sample sizes used
```

Set `--ellipse-std 0` to drop the ellipses if the plot gets too busy at
higher `K`. `--plot-sample-per-item` controls how many points get drawn (default
1500 per item); it is independent of `--fit-sample-per-item`, which controls
how many points are used to fit the PCA/K-means themselves.

## Per-Cluster Isolation Grid (One Cluster At A Time)

The on-mesh cluster PLY/PNG shows every cluster overlapping at once, which
can get visually busy at higher `K` (thin regions like legs sit right next
to several other clusters). `scripts/plot_cluster_grid.py` instead produces
one grid image per `--item`, with one panel per cluster: that cluster is
highlighted in its color, and every other vertex is greyed out.

```powershell
python scripts\plot_cluster_grid.py `
  --item brownCricket meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt `
  --item blackCricket meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt `
  --kmeans 6 `
  --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\cluster_grid
```

Same shared PCA + shared K-means fit as the other scripts (same seed and
sample size by default), so cluster ids/colors line up with everything else.
Each panel is labeled with the cluster's vertex count and share of that
item's total. Output, one grid per item:

```text
{name}_cluster_grid_k{K}.png
cluster_grid_k{K}_manifest.txt
```

This view is the one to use when a specific cluster's shape is hard to make
out in the combined coloring, or when comparing where "the same" cluster id
lands on two different specimens without the other clusters as visual noise.

## Comparing Everything In One Grid Image

Each run of the comparison scripts writes separate PNG files per item (and
per output, if `--preview` is set). To compare them at a glance instead of
opening files one at a time, tile them into one labeled grid with
`scripts/make_contact_sheet.py`. It takes any set of existing images and
places each one at a `--cell ROW COL IMAGE` position; row/column order
follows first appearance, so you control the layout just by the order you
list `--cell` entries in.

Example: brown vs. black cricket (rows) compared across the continuous
shared-PCA colors and two cluster granularities (columns), reusing the PNGs
already produced by two separate `--kmeans 6` and `--kmeans 10` runs:

```powershell
python scripts\make_contact_sheet.py `
  --cell brownCricket "Shared PCA" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\brownCricket_shared_pca_features.png `
  --cell brownCricket "K=6 clusters" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\brownCricket_shared_kmeans_k6_clusters.png `
  --cell brownCricket "K=10 clusters" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k10\brownCricket_shared_kmeans_k10_clusters.png `
  --cell blackCricket "Shared PCA" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\blackCricket_shared_pca_features.png `
  --cell blackCricket "K=6 clusters" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\blackCricket_shared_kmeans_k6_clusters.png `
  --cell blackCricket "K=10 clusters" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k10\blackCricket_shared_kmeans_k10_clusters.png `
  --title "Brown vs Black Cricket: Shared PCA and K-Means Cluster Maps" `
  --out visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\cluster_comparison_grid.png
```

This is a generic tiling tool, not specific to clustering. It works on any
PNGs this project produces (continuous PCA previews, cluster previews,
different `K` runs, different specimens, even the `debug_2d_diffusion_view.py`
outputs), since it just places whatever image path you give it at whatever
row/column label you give it.

## Optional: Viewing Cluster PLYs In Blender

The cluster PLYs are colored the same way the shared-PCA PLYs are (vertex
colors, RGBA), so the existing comparison scene builder works unmodified:

```powershell
python scripts\make_blender_feature_comparison.py `
  --item brownCricket_clusters visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\brownCricket_shared_kmeans_k6_clusters.ply `
  --item blackCricket_clusters visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\blackCricket_shared_kmeans_k6_clusters.ply `
  --output visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\clusters_compare.blend
```

Remember the Blender vertex-color material note from the main descriptor
guide: the Attribute node must point at the `Col` attribute for the colors to
show up in Material Preview/Rendered mode.
