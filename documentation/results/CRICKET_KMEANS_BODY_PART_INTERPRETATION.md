# Interpreting the Cricket K-Means Results as Body-Part Evidence

## The short answer

The useful result is **not** the hundred-color image. The strongest and most
interpretable setting in the experiments saved so far is **K=10**.

At K=10, 8 of the 9 corresponding manual landmarks receive the same cluster ID
on the brown and black cricket. The clusters distinguish coarse anatomical
types such as abdomen tip, head tip, antenna base, foreleg base, hind-leg base,
and thorax. However, left and right versions of the same appendage usually
receive the **same** cluster.

The defensible conclusion is therefore:

> The Diff3F descriptors encode coarse, cross-specimen anatomical structure,
> but they are largely invariant to bilateral side. They represent "foreleg-like"
> or "antenna-like" regions more reliably than a unique structure such as
> "left foreleg." This explains why point-to-point matching can select the
> symmetric leg even when it has found the correct general body-part type.

This is promising evidence of part awareness, but it is not yet proof of a
dense semantic segmentation model.

## What a K-means color actually means

The current clustering uses a single K-means model fitted to sampled,
L2-normalized 2048-dimensional descriptors from both crickets. It receives no
vertex coordinates, mesh connectivity, body-part names, or left/right labels.

Consequently:

- One color means only "these vertices are closest to the same descriptor
  centroid."
- The numeric cluster ID and its display color have no semantic or ordinal
  meaning.
- Similar-looking colors are not necessarily similar clusters.
- A cluster can occupy several disconnected areas of a mesh.
- A color becomes anatomically meaningful only after checking where it occurs
  and comparing it with landmarks or dense manual labels.
- Cluster IDs are comparable between these two meshes because the same shared
  model produced both maps. They may be permuted after a different fit or seed.
- Never compare an ID across different K runs: cluster 1 at K=6 is not the same
  object as cluster 1 at K=10.
- The saved `.npy` label array is the authoritative discrete result. A mesh
  viewer may visually blend vertex colors across a triangle; those intermediate
  colors are rendering artifacts, not extra clusters.

K-means is therefore a probe of structure in the descriptor space. It does not
turn Diff3F into a supervised body-part segmenter.

## The direct landmark check

The saved cluster labels were sampled at the nearest mesh vertex to each of the
nine manual landmarks. All landmark-to-mesh snap distances are small
(0.10--0.37% of the corresponding bounding-box diagonal), so the following
result is not explained by markers floating far from the surface. As a boundary
sensitivity check, the calculation was also repeated using the majority cluster
among the 100 Euclidean-nearest vertices to each landmark.

### K=10

| Manual landmark | Brown cluster | Black cluster | Same shared cluster? |
| --- | ---: | ---: | :---: |
| Abdomen tip | 7 | 7 | yes |
| Head tip | 8 | 8 | yes |
| Left antenna base | 1 | 1 | yes |
| Right antenna base | 1 | 1 | yes |
| Left foreleg base | 0 | 0 | yes |
| Right foreleg base | 0 | 0 | yes |
| Left hind-leg base | 9 | 9 | yes |
| Right hind-leg base | 9 | 2 | no |
| Thorax center | 6 | 6 | yes |

This is the clearest result in the current K-means experiment:

- **Cross-specimen consistency:** 8/9 landmark pairs keep the same ID.
- **Coarse part separation:** the six probed anatomical types map to distinct
  clusters, apart from the one black-cricket right-hind-leg exception.
- **Bilateral collapse:** left/right antenna bases share a cluster on both
  meshes; left/right foreleg bases share a cluster on both meshes; hind-leg
  bases share a cluster on the brown cricket but not the black one. Thus 5/6
  bilateral pair checks are merged rather than side-specific.

The isolated K=10 maps are easier to inspect than the all-color image:

- [Brown cricket, one cluster per panel](../../visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/cluster_grid/brownCricket_cluster_grid_k10.png)
- [Black cricket, one cluster per panel](../../visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/cluster_grid/blackCricket_cluster_grid_k10.png)

These landmark results do **not** prove that every vertex in cluster 0 is
foreleg, for example. A single landmark tests the cluster at one anatomical
location. Dense part masks would be needed to measure whole-part purity and
intersection-over-union.

## Why K=100 and K=500 are not useful body-part maps

Increasing K does not ask K-means to discover anatomy in more detail. It only
forces the descriptor cloud into more Voronoi cells. Once K exceeds the number
of stable semantic modes, the cells can represent pose, visibility, local
surface variation, noise, or specimen identity.

The saved runs show this transition quantitatively:

| K | Active IDs: brown | Active IDs: black | IDs active on both | Shared-ID Jaccard | Brown vertices in shared IDs | Black vertices in shared IDs | Matching landmarks: single vertex | Matching landmarks: 100-neighbor majority |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 6 | 6 | 6 | 100.0% | 100.0% | 100.0% | 8/9 | 9/9 |
| 10 | 10 | 10 | 10 | 100.0% | 100.0% | 100.0% | 8/9 | 8/9 |
| 20 | 17 | 18 | 15 | 75.0% | 80.7% | 80.5% | 3/9 | 5/9 |
| 100 | 57 | 57 | 14 | 14.0% | 16.7% | 11.5% | 0/9 | 0/9 |
| 500 | 259 | 244 | 3 | 0.6% | 0.1% | 0.8% | 0/9 | 0/9 |

At K=100, each cricket uses 57 cluster IDs, but only 14 IDs occur on both.
Most vertices belong to an ID that is absent from the other specimen, and none
of the nine corresponding landmarks retains its ID. At K=500, only 3 of the
500 IDs occur on both meshes. These settings are mainly fragmenting
specimen-specific descriptor modes; the additional colors are not evidence of
additional anatomical understanding.

There is also a sampling issue at very high K. The fit uses 8,000 descriptor
samples per cricket (16,000 total), so K=500 has only about 32 fit samples per
centroid on average. That is a fine quantization of two particular descriptor
clouds, not a sensible inventory of insect parts.

## How to read each saved K

### K=6: coarse anatomy

K=6 gives 8/9 cross-specimen landmark agreement at the exact snapped vertices
and 9/9 using the local-neighborhood majority, but it merges related
structures. For example, the head tip and antenna bases all use cluster 3,
while most foreleg and hind-leg bases use cluster 2. Read this as evidence for
broad body versus appendage structure, not individual parts.

### K=10: best current semantic probe

K=10 retains 8/9 agreement while separating the sampled anatomical types much
more clearly. This is the most useful result to show in the report.

### K=20: onset of over-segmentation

K=20 still has substantial cluster presence on both specimens, but only 3/9
exact landmark vertices retain the same ID (5/9 with the neighborhood-majority
check). It can be used to discuss finer substructure, but it is already less
stable as a shared anatomical vocabulary.

### K=100 and K=500: descriptor quantization, not semantic segmentation

These are useful as a negative/control result: they demonstrate that arbitrarily
raising K does not reveal arbitrarily detailed body parts. Do not try to name or
interpret every color.

## What to show in the report

Use the existing [K=6/K=10 comparison
grid](../../visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/cluster_comparison_grid.png)
as the qualitative figure, and place the K=10 landmark table beside or below
it as the quantitative sanity check. The landmark table is much more
informative than a 100-color legend.

If space allows, use the K-versus-consistency table as the ablation: K=6 is
coarse, K=10 gives the best useful granularity, and K=20/100/500 show that more
clusters do not mean more semantic understanding. K=100 can be shown once as a
failure/over-segmentation example, but it should not be the primary result.

## Relation to the point-to-point correspondence failures

The K-means and correspondence results tell a consistent story:

1. K=10 often groups the correct *type* of structure across specimens.
2. It usually groups the left and right copies of that structure together.
3. Point-to-point nearest-neighbor search must choose one exact vertex, so it
   can select the symmetric copy while still being semantically close.

The hard leg mismatches therefore do not mean that the descriptor is random.
They indicate a difference between **part-level semantics** and
**instance-level localization**. The current descriptor is better at the first.

K-means is a coarse quantization, so it cannot prove that no left/right signal
exists anywhere in the continuous 2048-D descriptor. The more precise statement
is that side information is not strong or stable enough to dominate the current
clustering and correspondence results.

## What may be claimed in the report

### Supported by the present experiment

- The descriptor space contains non-random structure correlated with coarse
  insect anatomy.
- That structure transfers surprisingly well between this brown/black cricket
  pair at K=6 and K=10.
- K=10 is more anatomically informative than K=6 for the available landmarks.
- The representation tends to be bilateral or part-type invariant, which is
  consistent with symmetric-leg correspondence errors.
- Very large K values are unstable across specimens and should not be read as
  semantic part decompositions.

### Not supported yet

- That each cluster is a pure, complete body part.
- That the descriptors reliably distinguish left from right.
- That K=10 is universally optimal.
- That the result generalizes beyond this pair of crickets.
- A numerical dense-segmentation accuracy, because dense ground-truth part
  labels do not yet exist.

## Report-ready paragraph

> Shared K-means clustering was used as an unsupervised probe of the Diff3F
> descriptor field rather than as a semantic segmentation method. With K=10,
> eight of nine corresponding manual landmarks on two cricket specimens were
> assigned to the same shared cluster, and landmarks from different coarse
> anatomical categories generally occupied different clusters. In contrast,
> five of six within-specimen bilateral landmark pairs shared a cluster. This
> suggests that the descriptors encode transferable part type, such as
> antenna-, foreleg-, or thorax-like appearance, more strongly than laterality.
> This interpretation is consistent with point-to-point failures that land on
> the symmetric leg. Increasing K did not improve semantic resolution: at
> K=100 only 14 cluster IDs occurred on both specimens and no corresponding
> landmark retained its ID, indicating specimen-specific over-segmentation.
> The result is therefore evidence of coarse anatomical organization, but not
> of reliable dense or left/right-aware body-part segmentation.

## The clean next evaluation

If a stronger segmentation claim is needed, manually label a small number of
meshes with dense classes at two levels:

1. **Side-agnostic:** head, thorax, abdomen, antenna, foreleg, middle leg,
   hind leg, wing.
2. **Side-aware:** split each paired appendage into left and right.

Then report cluster purity or normalized mutual information, plus mean IoU after
matching cluster IDs to semantic classes. Comparing the side-agnostic and
side-aware scores would directly quantify the symmetry problem. The present
nine-landmark analysis is a strong sanity check, but dense labels are the step
that turns "the colors look anatomical" into a segmentation measurement.

## Inputs used for this note

- Shared clustering manifests and label arrays:
  `visualizations/bugnist_crickets_features_16v_512/shared_pca_kmeans_brown_black/`
- Brown landmarks:
  `landmarks/bugnist_crickets/brownCricket_rotated_obj_landmarks.csv`
- Black landmarks:
  `landmarks/bugnist_crickets/blackCricket_rotated_obj_landmarks.csv`
- Clustering method and settings:
  `scripts/visualize_feature_comparison.py`
