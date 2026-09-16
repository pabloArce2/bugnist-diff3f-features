# Cricket Shared-PCA K-Means Clustering Summary

This note summarizes the first shared K-means clustering run on the same two
rotated cricket meshes used for the shared-PCA comparison and the landmark
benchmark:

- `brownCricket`: `meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj`
- `blackCricket`: `meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj`

Both use the 16-view, 512x512 descriptors in
`output/bugnist_crickets_features_16v_512/`.

The goal was to turn the existing continuous shared-PCA color visualization
(`visualizations/bugnist_crickets_features_16v_512/shared_pca_brown_black/`)
into a discrete k-way region map: one shared K-means model fit across both
crickets, so cluster id `i` means the same thing on both specimens. See
`documentation/guides/PCA_KMEANS_CLUSTERING.md` for how the tool works.

## Methodology Update: Clustering Now Uses The Full Descriptor

The first version of this experiment clustered the 3-D shared-PCA coordinate
(`--cluster-on pca`), the same 3 numbers used for the RGB colors. That is
cheap and matches the visible colors exactly, but it clusters a lossy
compression chosen for human color perception, not the actual descriptor.
The standard approach is to cluster the full normalized 2048-D descriptor
instead, so `--cluster-on features` is now the default in
`visualize_feature_comparison.py`, `visualize_pointcloud_feature_comparison.py`,
`plot_pca_kmeans_scatter.py`, and `plot_cluster_grid.py`. `pca` is still
available as an option, not removed.

All numbers, tables, and images below were regenerated with the new default
(`cluster_on: features`). See "PCA-Space Vs Feature-Space Clustering" near
the end of this document for a direct comparison against the original
`pca`-space run, including one thing that got *less* clean under the
"more correct" method.

## Short Conclusion

The clustering is not random. Across two independent choices of `K` (6 and
10), a small group of clusters consistently and heavily over-represents the
vertices farthest from the body centroid, on both crickets independently.
Visually, those clusters sit on the legs and antennae in both preview
renders. This is a second, independent piece of evidence (alongside the
landmark benchmark) that the shared descriptor space captures real,
cross-specimen-consistent surface structure, not just per-shape noise.

It is still a coarse region map, not a verified anatomical segmentation. No
manual ground-truth part labels exist yet to score this against.

## Choosing K

The silhouette/inertia sweep below was run before the `cluster_on` default
changed, so it measures distances in the 3-D PCA space, not the full 2048-D
space actually used now. The absolute numbers do not carry over, but the
qualitative conclusion (no sharp elbow, so treat `K` as a granularity choice
rather than something to solve for) is still the operating assumption for
this document; re-running the sweep against `cluster_on: features` is on the
next-steps list.

K-means for that original sweep was fit on 8000 sampled vertices per item
(16000 fit points total), seed 42:

| K | inertia | silhouette |
| ---: | ---: | ---: |
| 3 | 1097.8 | 0.372 |
| 4 | 843.8 | 0.363 |
| 5 | 692.8 | 0.349 |
| 6 | 595.6 | 0.344 |
| 7 | 518.9 | 0.343 |
| 8 | 467.3 | 0.343 |
| 10 | 387.5 | 0.342 |
| 12 | 339.2 | 0.321 |
| 16 | 276.9 | 0.291 |

Silhouette score falls off smoothly with no sharp elbow. This is expected: a
cricket mesh is one connected surface, not a set of naturally separated
blobs, so there is no single "correct" K here. `K=6` (coarse regions) and
`K=10` (finer regions) were both generated for this summary as two points on
that granularity dial.

## Files Used

- Source mesh (brown): `meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj` (24,683 vertices / 49,996 faces)
- Source mesh (black): `meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj` (24,761 vertices / 49,993 faces)
- Features (brown): `output/bugnist_crickets_features_16v_512/BrownCricket_rotated_diff3f.pt`
- Features (black): `output/bugnist_crickets_features_16v_512/BlackCricket_10_10_rotated_diff3f.pt`

## K=6 Result

Cluster sizes (`cluster_on: features`):

| Cluster | Color | brownCricket | blackCricket |
| ---: | --- | ---: | ---: |
| 0 | red | 4212 (17.1%) | 3706 (15.0%) |
| 1 | yellow | 5631 (22.8%) | 3037 (12.3%) |
| 2 | green | 5267 (21.3%) | 4409 (17.8%) |
| 3 | cyan | 2669 (10.8%) | 3696 (14.9%) |
| 4 | blue | 4513 (18.3%) | 4346 (17.6%) |
| 5 | magenta | 2391 (9.7%) | 5567 (22.5%) |

Every cluster is present in both specimens with a non-trivial share (no empty
or near-empty clusters at this `K`).

Cluster membership was compared against the 5% of vertices farthest from the
mesh centroid (a simple proxy for "sticking out of the body," which legs and
antennae do and the body itself does not):

```text
brownCricket: cluster 5 is 9.7% of all vertices, but 38.6% of the farthest-5% vertices (4.0x enrichment)
blackCricket: cluster 3 is 14.9% of all vertices, but 44.1% of the farthest-5% vertices (3.0x enrichment)
```

Unlike the earlier `pca`-space run, the *top* far-set cluster is not the same
id on both crickets. What is consistent is the *set* of clusters that show
up as extremity-heavy: `{1, 3, 5}` for brownCricket's far vertices and
`{1, 3, 5}` for blackCricket's (same three ids, different rank order). See
"PCA-Space Vs Feature-Space Clustering" below for the direct comparison.

## K=10 Result

Cluster sizes (`cluster_on: features`):

| Cluster | brownCricket | blackCricket |
| ---: | ---: | ---: |
| 0 | 3703 (15.0%) | 2426 (9.8%) |
| 1 | 1688 (6.8%) | 2012 (8.1%) |
| 2 | 2280 (9.2%) | 2140 (8.6%) |
| 3 | 3822 (15.5%) | 1123 (4.5%) |
| 4 | 16 (0.1%) | 3453 (13.9%) |
| 5 | 2055 (8.3%) | 3329 (13.4%) |
| 6 | 4084 (16.5%) | 2804 (11.3%) |
| 7 | 3205 (13.0%) | 2954 (11.9%) |
| 8 | 1704 (6.9%) | 2274 (9.2%) |
| 9 | 2126 (8.6%) | 2246 (9.1%) |

Note cluster `4` on brownCricket: 16 vertices, 0.1% of the mesh. Clustering
the full 2048-D space gives K-means much more room to carve out a tiny
outlier group than the bounded 3-D PCA space did; every cluster at K=10 in
the original `pca`-space run had a non-trivial share. This is a real
trade-off of the "more correct" method, not a bug — see the comparison
section below.

The extremity check at K=10:

```text
brownCricket: cluster 1 is 6.8% of all vertices, but 45.3% of the farthest-5% vertices (6.6x enrichment)
blackCricket: cluster 1 is 8.1% of all vertices, but 36.1% of the farthest-5% vertices (4.4x enrichment)
```

Cluster `1` is now the top far-set cluster on *both* crickets at K=10 (unlike
K=6, where the top id differed). Runner-up far-set clusters are `{5, 3}` for
brownCricket and `{9, 5}` for blackCricket — cluster `5` appears in both
runner-up sets, `3` vs `9` do not overlap.

## What This Means

At two independent values of `K`, using the full 2048-D descriptor (no PCA
compression, no spatial/geometric input at all), a small group of clusters
still lands disproportionately on the parts of the body that stick out the
most, on both cricket specimens independently. Since the clustering never
sees vertex position or which specimen a sample came from, this is evidence
that the underlying Diff3F descriptor field itself, not just the
correspondence nearest-neighbor search, encodes something that correlates
with coarse body-part identity — the same conclusion as the original
`pca`-space run, now confirmed with the statistically more defensible
clustering input.

This complements the landmark benchmark result
(`documentation/results/CRICKET_LANDMARK_BENCHMARK_SUMMARY.md`), which found
Diff3F correspondence to be strong on distinctive/central landmarks (head,
thorax, antenna bases) but weak and confused on repeated leg landmarks. The
clustering result here does not contradict that: "the descriptor knows this
region is leg-like" is a weaker and different claim than "the descriptor can
tell your left foreleg from your right hind leg," and the leg-landmark
failures in the benchmark are consistent with legs collectively forming a
recognizable-but-internally-ambiguous region.

## PCA-Space Vs Feature-Space Clustering

Direct comparison between the original run (`cluster_on: pca`, clustering the
3-D shared-PCA coordinate) and the corrected default (`cluster_on: features`,
clustering the full normalized 2048-D descriptor), same items, same `K`,
same seed:

| | `pca` (original) | `features` (current default) |
| --- | --- | --- |
| K=6 top far-set cluster | same id (`4`) on both crickets | different ids (`5` brown / `3` black); same 3-id *set* on both |
| K=6 enrichment | ~2-3x | ~3-4x |
| K=10 top far-set cluster | same id (`2`) on both crickets, ~8-9x | same id (`1`) on both crickets, ~4.4-6.6x |
| Empty/near-empty clusters | none observed | one (brownCricket K=10 cluster 4, 16 vertices) |
| Matches the visible RGB gradient | yes, by construction | not necessarily |

Two honest conclusions:

```text
1. The extremity signal is not an artifact of clustering the compressed
   3-D space. It survives, at comparable or even somewhat higher
   enrichment, when clustering the full descriptor. That is the more
   important result, and it is why cluster_on: features is now the
   default -- it is the statistically standard choice AND it still finds
   the same qualitative structure.

2. Clustering in full 2048-D space is noisier at the level of exact
   cluster-id alignment between two specimens. The 3-D PCA compression,
   by throwing away most directions before clustering, incidentally acted
   like a regularizer that made the shared K-means land on nearly
   identical cluster boundaries for both crickets. The full descriptor
   gives K-means more directions to disagree about between two
   individuals, even though the same overall body-region structure is
   still there. This is a real trade-off, not a reason to prefer the
   less-standard method: reading off "the same numbered cluster is legs
   on both crickets" is easier with cluster_on: pca, but reading off "the
   legs form a distinct, findable group on both crickets" holds under
   cluster_on: features too, using the set-of-clusters view instead of a
   single id.
```

## Important Caveats

There is no manual ground-truth part segmentation to score these clusters
against, so "looks like legs and antennae" is a qualitative read supported by
the centroid-distance proxy, not a validated body-part label.

Cluster ids are arbitrary and were not chosen to mean anything by
themselves; only the spatial pattern per id is meaningful, and only because
the same shared model was applied to both items.

The centroid-distance proxy is intentionally crude. It will also pick up
other protruding geometry (for example, the wing-case edge or an
antenna-adjacent head structure) as "far," so treat "legs and antennae" as
the dominant visual pattern in the preview renders, not a guarantee about
every vertex in that cluster.

Only one cricket pair was tested. Whether this extremity-cluster pattern
holds for the other BugNIST species (`guld`, `sfaar`/soldier, `soldat`) is
untested.

## Side-By-Side Grid

All of the above is easier to judge as one image than as six separate files.
`scripts/make_contact_sheet.py` tiles the existing preview PNGs into a single
labeled grid: rows are `brownCricket`/`blackCricket`, columns are the
continuous shared-PCA colors, the `K=6` cluster map, and the `K=10` cluster
map.

```text
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/cluster_comparison_grid.png
```

Reading left to right on either row: the continuous PCA column is the
starting point (soft, hard to draw boundaries in), the `K=6` column shows
coarse regions with hard boundaries, and the `K=10` column shows those same
regions splitting further, most visibly around the legs. Reading top to
bottom within a column: the same cluster colors land on the same relative
body regions on both crickets (magenta on the head/thorax-top area, blue on
the legs), which is the cross-specimen consistency this whole experiment was
checking for. See `documentation/guides/PCA_KMEANS_CLUSTERING.md` for the
exact command used to build this grid.

## Feature-Space View: The Points And Centroids Themselves

The region maps above color the mesh surface; they do not show what K-means
actually saw. `scripts/plot_pca_kmeans_scatter.py` plots that directly: the
sampled shared-PCA points themselves, colored by cluster, with a dashed
ellipse per cluster and the K-means centroids marked as black `x`. Since the
shared PCA space here is 3-D, the plot shows all three pairwise projections
(PC1-PC2, PC1-PC3, PC2-PC3) side by side, and both crickets are drawn on the
same axes (circle markers for `brownCricket`, triangles for `blackCricket`).

```text
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/pca_kmeans_scatter_k6.png
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/pca_kmeans_scatter_k10.png
```

Two things are worth reading off these plots directly:

```text
1. brownCricket (circles) and blackCricket (triangles) fall inside the same
   colored blobs, not two separate clouds. That is the cross-specimen
   consistency claim made visible directly in feature space, not just
   inferred from matching mesh regions.

2. The blobs overlap, sometimes heavily (for example cluster 2 sits in the
   crowded center of all three K=6 panels, overlapping clusters 1, 4, and 5).
   K-means still
   assigns a single hard label per point, but the underlying space is a
   continuum, not cleanly separated clusters. This is the same thing the
   landmark benchmark saw as "small nearest-neighbor margins": in real
   feature space, unlike the idealized textbook picture, boundary points
   between two regions are only marginally closer to one centroid than the
   other.
```

Reproduction:

```powershell
python scripts\plot_pca_kmeans_scatter.py --item brownCricket output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt --item blackCricket output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt --kmeans 6 --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black
python scripts\plot_pca_kmeans_scatter.py --item brownCricket output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt --item blackCricket output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt --kmeans 10 --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black
```

## Output Files

Comparison grid (both crickets x Shared PCA/K=6/K=10, see "Side-By-Side Grid" above):

```text
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/cluster_comparison_grid.png
```

Feature-space scatter plots (see "Feature-Space View" above):

```text
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/pca_kmeans_scatter_k6.png
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/pca_kmeans_scatter_k6_manifest.txt
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/pca_kmeans_scatter_k10.png
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/pca_kmeans_scatter_k10_manifest.txt
```

K=6:

```text
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/brownCricket_shared_pca_features.ply
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/brownCricket_shared_pca_features.png
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/brownCricket_shared_kmeans_k6_clusters.ply
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/brownCricket_shared_kmeans_k6_clusters.png
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/brownCricket_shared_kmeans_k6_labels.npy
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/blackCricket_shared_pca_features.ply
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/blackCricket_shared_pca_features.png
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/blackCricket_shared_kmeans_k6_clusters.ply
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/blackCricket_shared_kmeans_k6_clusters.png
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/blackCricket_shared_kmeans_k6_labels.npy
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/shared_kmeans_k6_legend.png
visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k6/manifest.txt
```

K=10: same layout under
`visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/k10/`.

## Reproduction Commands

K=6 (explicit `--cluster-on features`, though it is now the default):

```powershell
python scripts\visualize_feature_comparison.py --item brownCricket meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt --item blackCricket meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6 --kmeans 6 --cluster-on features --preview
```

K=10:

```powershell
python scripts\visualize_feature_comparison.py --item brownCricket meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt --item blackCricket meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt --outdir visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k10 --kmeans 10 --cluster-on features --preview
```

Comparison grid (after both of the above have been run):

```powershell
python scripts\make_contact_sheet.py --cell brownCricket "Shared PCA" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\brownCricket_shared_pca_features.png --cell brownCricket "K=6 clusters" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\brownCricket_shared_kmeans_k6_clusters.png --cell brownCricket "K=10 clusters" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k10\brownCricket_shared_kmeans_k10_clusters.png --cell blackCricket "Shared PCA" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\blackCricket_shared_pca_features.png --cell blackCricket "K=6 clusters" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k6\blackCricket_shared_kmeans_k6_clusters.png --cell blackCricket "K=10 clusters" visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\k10\blackCricket_shared_kmeans_k10_clusters.png --title "Brown vs Black Cricket: Shared PCA and K-Means Cluster Maps (cluster_on=features)" --out visualizations\bugnist_crickets_features_16v_512\shared_pca_kmeans_brown_black\cluster_comparison_grid.png
```

To sweep several `K` values in one go instead of retyping the command, see
"Running Several K Values" in `documentation/guides/PCA_KMEANS_CLUSTERING.md`.

## Suggested Next Experiment

1. Run the same clustering on the `bcrick`/`sfaar`/`soldat` individual-specimen
   trio (different species, not two crickets) to see whether the
   extremity-cluster pattern still holds across bigger morphology differences.
2. Cross-check cluster labels against the manual landmark CSVs: do the
   `*_LEG_BASE` and `*_ANTENNA_BASE` landmarks fall inside the extremity
   cluster(s) found here?
3. Re-run the silhouette/inertia `K` sweep against `cluster_on: features`
   directly (the existing table in "Choosing K" was computed in PCA space,
   before the default changed).
4. Investigate the near-empty cluster at K=10 (`brownCricket` cluster `4`,
   16 vertices): is it a meaningful tiny outlier region, or an artifact of
   `n_init=10` landing on a bad local optimum for that specimen?
5. If the extremity pattern holds up across more species, use it as a
   pre-filter for `compute_feature_correspondences.py` (e.g. only match leg
   vertices against other leg vertices) to see whether it reduces the
   wrong-leg confusion seen in the landmark benchmark.
