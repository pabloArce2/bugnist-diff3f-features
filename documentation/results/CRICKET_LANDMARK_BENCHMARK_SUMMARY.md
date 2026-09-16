# Cricket Landmark Benchmark Summary

This note summarizes the first manual landmark benchmark on the two rotated
cricket meshes:

- `brownCricket`: `meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj`
- `blackCricket`: `meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj`

The goal was to test whether Diff3F descriptors can recover the same anatomical
landmark on another cricket mesh.

The benchmark uses 9 manual landmarks placed in Blender:

- `ABDOMEN_TIP`
- `HEAD_TIP`
- `LEFT_ANTENNA_BASE`
- `LEFT_FORELEG_BASE`
- `LEFT_HIND_LEG_BASE`
- `RIGHT_ANTENNA_BASE`
- `RIGHT_FORELEG_BASE`
- `RIGHT_HIND_LEG_BASE`
- `THORAX_CENTER`

## Short Conclusion

Diff3F is producing meaningful anatomical correspondences, but it is not
perfectly reliable yet.

It works best on distinctive or central anatomy:

- antenna bases
- head tip
- thorax center
- abdomen tip

It struggles on repeated/symmetric leg anatomy:

- foreleg bases
- hind leg bases

This is a useful result. It suggests the descriptors are not random, but the
current setup is still confused by structures that look similar across the body.

## What Was Tested

For each source landmark, the benchmark does this:

1. Snap the manual landmark to the nearest source mesh vertex.
2. Read that vertex's Diff3F descriptor from the source `.pt` file.
3. Search for the closest descriptor in the target `.pt` file.
4. Compare the predicted target vertex to the manual target landmark with the
   same label.

So, for example:

```text
brown HEAD_TIP descriptor -> nearest black descriptor -> compare to black HEAD_TIP
```

The benchmark was run in both directions:

1. `brownCricket` to `blackCricket`
2. `blackCricket` to `brownCricket`

## Main Metrics

### `target_error_bbox_pct`

This is the main number to read.

It means:

```text
distance from predicted point to manual target landmark
divided by the target mesh bounding-box diagonal
reported as a percentage
```

Practical interpretation:

| Error | Meaning |
| --- | --- |
| `0-2%` | very good |
| `2-5%` | usable / promising |
| `5-10%` | weak or borderline |
| `>10%` | failed anatomical correspondence |

### `PCK`

PCK means percentage of correct keypoints.

For example:

```text
PCK@5% = 6/9
```

means 6 of the 9 landmarks landed within 5 percent of the target mesh
bounding-box diagonal.

### `gt_feature_rank`

This asks:

```text
Where did the manual target landmark rank in descriptor similarity?
```

Lower is better.

| Rank | Meaning |
| --- | --- |
| `1-10` | excellent |
| `<100` | meaningful |
| `100-500` | ambiguous |
| `>500` | weak/confused |

Important: a landmark can be spatially close but still have a high rank,
especially when many vertices have very similar descriptors.

### `nn_margin`

This is the difference between the best descriptor match and the second-best
descriptor match.

The margins here are very small, which means many matches are ambiguous. This is
expected for repeated insect anatomy, especially legs.

### `source_snap_distance` and `target_snap_distance`

Manual Blender markers are arbitrary 3D positions. The `.pt` file only has
features for actual mesh vertices.

So the benchmark first snaps each marker to the nearest vertex. Large snap
distances mean the marker may be floating away from the mesh surface or the
nearest available vertex is not exactly at the intended anatomical point.

## Experiment 1: Brown To Black

Files:

- Source mesh: `meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj`
- Source features: `output/bugnist_crickets_features_16v_512/BrownCricket_rotated_diff3f.pt`
- Source landmarks: `landmarks/bugnist_crickets/brownCricket_rotated_obj_landmarks.csv`
- Target mesh: `meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj`
- Target features: `output/bugnist_crickets_features_16v_512/BlackCricket_10_10_rotated_diff3f.pt`
- Target landmarks: `landmarks/bugnist_crickets/blackCricket_rotated_obj_landmarks.csv`

Summary:

| Metric | Value |
| --- | ---: |
| Common landmarks | `9` |
| Source vertices/features | `24683` |
| Target vertices/features | `24761` |
| Feature dimension | `2048` |
| PCK@1% | `1/9` |
| PCK@2% | `3/9` |
| PCK@5% | `6/9` |
| PCK@10% | `7/9` |
| Median target error | `2.10%` |
| Mean target error | `9.50%` |
| GT rank top 100 | `4/9` |
| GT rank top 1000 | `9/9` |
| Median GT feature rank | `210` |

Per-landmark result:

| Landmark | Error | GT rank | Interpretation |
| --- | ---: | ---: | --- |
| `RIGHT_ANTENNA_BASE` | `0.63%` | `8` | excellent |
| `LEFT_FORELEG_BASE` | `1.57%` | `210` | spatially good, descriptor ambiguous |
| `LEFT_ANTENNA_BASE` | `1.75%` | `243` | spatially good, descriptor ambiguous |
| `HEAD_TIP` | `2.08%` | `52` | good |
| `THORAX_CENTER` | `2.10%` | `35` | good |
| `ABDOMEN_TIP` | `2.60%` | `72` | usable |
| `LEFT_HIND_LEG_BASE` | `5.08%` | `376` | borderline |
| `RIGHT_FORELEG_BASE` | `29.35%` | `619` | failed |
| `RIGHT_HIND_LEG_BASE` | `40.36%` | `250` | failed |

Interpretation:

Brown to black works well for 6 of 9 landmarks under the 5 percent threshold.
The failures are concentrated on the right-side leg bases. This suggests the
descriptor is confused by repeated leg structures.

## Experiment 2: Black To Brown

Files:

- Source mesh: `meshes/bugnist_crickets/sfaar_10_010/BlackCricket_10_10_rotated.obj`
- Source features: `output/bugnist_crickets_features_16v_512/BlackCricket_10_10_rotated_diff3f.pt`
- Source landmarks: `landmarks/bugnist_crickets/blackCricket_rotated_obj_landmarks.csv`
- Target mesh: `meshes/bugnist_crickets/bcrick_10_010/BrownCricket_rotated.obj`
- Target features: `output/bugnist_crickets_features_16v_512/BrownCricket_rotated_diff3f.pt`
- Target landmarks: `landmarks/bugnist_crickets/brownCricket_rotated_obj_landmarks.csv`

Summary:

| Metric | Value |
| --- | ---: |
| Common landmarks | `9` |
| Source vertices/features | `24761` |
| Target vertices/features | `24683` |
| Feature dimension | `2048` |
| PCK@1% | `1/9` |
| PCK@2% | `3/9` |
| PCK@5% | `5/9` |
| PCK@10% | `6/9` |
| Median target error | `2.40%` |
| Mean target error | `11.19%` |
| GT rank top 100 | `4/9` |
| GT rank top 1000 | `9/9` |
| Median GT feature rank | `136` |

Per-landmark result:

| Landmark | Error | GT rank | Interpretation |
| --- | ---: | ---: | --- |
| `RIGHT_ANTENNA_BASE` | `0.83%` | `5` | excellent |
| `LEFT_ANTENNA_BASE` | `1.16%` | `53` | good |
| `THORAX_CENTER` | `1.63%` | `20` | good |
| `ABDOMEN_TIP` | `2.39%` | `194` | usable |
| `HEAD_TIP` | `2.40%` | `25` | good |
| `LEFT_HIND_LEG_BASE` | `5.62%` | `345` | weak |
| `RIGHT_FORELEG_BASE` | `26.52%` | `136` | failed |
| `LEFT_FORELEG_BASE` | `26.65%` | `849` | failed |
| `RIGHT_HIND_LEG_BASE` | `33.55%` | `332` | failed |

Interpretation:

Black to brown is slightly weaker than brown to black. It still finds 5 of 9
landmarks under the 5 percent threshold, but both foreleg bases and the right
hind leg base fail badly.

## Cross-Experiment Interpretation

The consistent strong landmarks are:

- `RIGHT_ANTENNA_BASE`
- `LEFT_ANTENNA_BASE`
- `HEAD_TIP`
- `THORAX_CENTER`
- `ABDOMEN_TIP`

The consistent weak landmarks are:

- `RIGHT_FORELEG_BASE`
- `RIGHT_HIND_LEG_BASE`
- `LEFT_HIND_LEG_BASE`

The direction-dependent landmark is:

- `LEFT_FORELEG_BASE`

It works well from brown to black, but fails from black to brown.

## What This Means

The descriptor field is anatomically meaningful, especially on landmarks that
are globally distinctive or close to central body structure.

However, the current setup has trouble separating repeated appendages. Legs have
similar local geometry and similar rendered appearance, so the descriptor nearest
neighbor can jump to the wrong leg region.

The benchmark result is therefore promising but not conclusive:

- Good enough to show that Diff3F is learning useful correspondences.
- Not yet reliable enough for precise full-body anatomical landmark transfer.
- Especially weak on symmetric/repeated leg structures.

## Why The Mean Error Looks Worse Than The Median

The median errors are low:

- Brown to black: `2.10%`
- Black to brown: `2.40%`

But the mean errors are higher:

- Brown to black: `9.50%`
- Black to brown: `11.19%`

This happens because a few landmarks fail very badly. The median says the
typical landmark is decent. The mean says the system still has serious outliers.

For this experiment, the median is a better description of the normal behavior,
and the failed leg landmarks explain the high mean.

## Important Caveats

The benchmark has only 9 landmarks. This is enough for a first sanity check, but
not enough for a final claim.

Some markers have larger snap distances, especially hind-leg landmarks. This may
mean the spheres were not exactly on the mesh surface, or that the intended
anatomical location does not have a nearby vertex.

The nearest-neighbor descriptor margin is small. That means the top match and
second-best match are often almost tied, so some predictions are unstable.

Also, a high cosine score does not guarantee anatomical correctness. A wrong leg
can still have a high descriptor similarity because the surface looks similar.

## Output Files

Brown to black:

- Summary JSON: `visualizations/bugnist_crickets_landmark_benchmark/brownCricket_to_blackCricket/brownCricket_to_blackCricket_landmark_summary.json`
- Benchmark CSV: `visualizations/bugnist_crickets_landmark_benchmark/brownCricket_to_blackCricket/brownCricket_to_blackCricket_landmark_benchmark.csv`
- Predicted matches CSV: `visualizations/bugnist_crickets_landmark_benchmark/brownCricket_to_blackCricket/brownCricket_to_blackCricket_predicted_matches.csv`
- Blender scene: `visualizations/bugnist_crickets_landmark_benchmark/brownCricket_to_blackCricket/brownCricket_to_blackCricket_landmark_scene.blend`

Black to brown:

- Summary JSON: `visualizations/bugnist_crickets_landmark_benchmark/blackCricket_to_brownCricket/blackCricket_to_brownCricket_landmark_summary.json`
- Benchmark CSV: `visualizations/bugnist_crickets_landmark_benchmark/blackCricket_to_brownCricket/blackCricket_to_brownCricket_landmark_benchmark.csv`
- Predicted matches CSV: `visualizations/bugnist_crickets_landmark_benchmark/blackCricket_to_brownCricket/blackCricket_to_brownCricket_predicted_matches.csv`
- Blender scene: `visualizations/bugnist_crickets_landmark_benchmark/blackCricket_to_brownCricket/blackCricket_to_brownCricket_landmark_scene.blend`

## Reproduction Commands

Brown to black:

```powershell
python scripts\evaluate_landmark_benchmark.py --source-name brownCricket --source-geometry meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj --source-features output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt --source-landmarks landmarks\bugnist_crickets\brownCricket_rotated_obj_landmarks.csv --target-name blackCricket --target-geometry meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj --target-features output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt --target-landmarks landmarks\bugnist_crickets\blackCricket_rotated_obj_landmarks.csv --outdir visualizations\bugnist_crickets_landmark_benchmark\brownCricket_to_blackCricket
```

Black to brown:

```powershell
python scripts\evaluate_landmark_benchmark.py --source-name blackCricket --source-geometry meshes\bugnist_crickets\sfaar_10_010\BlackCricket_10_10_rotated.obj --source-features output\bugnist_crickets_features_16v_512\BlackCricket_10_10_rotated_diff3f.pt --source-landmarks landmarks\bugnist_crickets\blackCricket_rotated_obj_landmarks.csv --target-name brownCricket --target-geometry meshes\bugnist_crickets\bcrick_10_010\BrownCricket_rotated.obj --target-features output\bugnist_crickets_features_16v_512\BrownCricket_rotated_diff3f.pt --target-landmarks landmarks\bugnist_crickets\brownCricket_rotated_obj_landmarks.csv --outdir visualizations\bugnist_crickets_landmark_benchmark\blackCricket_to_brownCricket
```

## Suggested Next Experiment

For the next benchmark iteration:

1. Add more landmarks, especially on legs.
2. Recheck hind-leg marker placement and snap distances.
3. Try higher descriptor quality, for example more views or shared prompts.
4. Use mutual matching or spatial constraints to reduce wrong-leg jumps.
5. Compare results across more cricket pairs, not only one pair.

The most important next question is whether leg failures persist across more
individuals, or whether they are specific to this pair and these marker
locations.
