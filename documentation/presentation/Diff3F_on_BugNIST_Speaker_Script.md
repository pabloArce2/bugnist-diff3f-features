# Diff3F on BugNIST — Speaker Script

Planned length: **22:40** (20–25 minute target)

The same dialogue is embedded in the PowerPoint speaker notes.

## Slide 01 — Diff3F on BugNIST

**Time:** 00:00–00:45 (0:45)  
**Presenter cue:** Open with the final descriptor field, not the machinery.

Good morning. This project asks whether a model trained on ordinary two-dimensional images can give semantic meaning to three-dimensional insect scans. The input is only geometry reconstructed from X-ray CT: no texture, no labels, and no part annotations. The coloured cricket on the screen is the output. Every surface point has received a 2,048-dimensional Diff3F descriptor, and the colour is only a three-dimensional PCA view of that much richer feature. The practical question is not simply whether the colours look smooth. It is whether a point that means head, thorax, abdomen, antenna, or leg is represented similarly on another insect. I will first show how Diff3F transfers semantics from 2D models into 3D, then how I adapted the pipeline to BugNIST, and finally where correspondence works and where it fails.

## Slide 02 — Can a surface know its anatomy?

**Time:** 00:45–01:55 (1:10)  
**Presenter cue:** Contrast geometry-only input with a semantic descriptor field.

A CT reconstruction gives us a surface. At each vertex we know a position and usually a normal, but nothing says what that point represents anatomically. Coordinates alone are a poor identity: two corresponding points can be far apart because two specimens have different sizes, poses, or proportions. Classical geometric descriptors can describe local shape, but they do not bring a strong semantic prior and generally prefer shapes that are close to isometric. Diff3F takes a different route. It asks pretrained image models to interpret several views of the object, then transfers their features back to the surface. On the left is the kind of untextured mesh we begin with. On the right, colour reveals an organised descriptor field. The key question for the whole talk is whether this organisation is genuinely anatomical or only visually attractive.

## Slide 03 — Coordinates move. Meaning should not.

**Time:** 01:55–03:00 (1:05)  
**Presenter cue:** Define the desired invariance and the target use: cross-specimen correspondence.

Imagine selecting the head tip on one cricket and asking for the equivalent point on another. A nearest Euclidean coordinate would be meaningless because the bodies are placed differently and their proportions are not identical. What we want is a descriptor whose similarity reflects what a point is, rather than where it happens to be. If the representation works, the two heads should be near each other in feature space, even when they are far apart in world space. The same idea should extend to the thorax, abdomen, and appendages. This is useful for transferring annotations, comparing morphology, and eventually building consistent measurements across specimens. But insects make the problem difficult: legs are repeated, left and right sides are approximately symmetric, and CT-derived surfaces include noise that clean benchmark meshes do not.

## Slide 04 — How does Diff3F move semantics into 3D?

**Time:** 03:00–04:15 (1:15)  
**Presenter cue:** Walk left to right through the pipeline image.

The central idea is a render–interpret–project loop. First, the 3D shape is rendered from several cameras. For every camera we also produce depth, surface-normal, and visibility maps. Stable Diffusion then turns the plain render into a plausible image, while ControlNet uses those geometric maps to keep the generated appearance aligned with the original silhouette. Two feature streams are extracted: intermediate activations from the diffusion UNet and DINOv2 features from the final image. They are normalised, given equal weight, concatenated into a 2,048-dimensional pixel descriptor, and projected back through the known camera onto the visible surface. Finally, all observations of each vertex are averaged across views. No 3D training and no insect part labels are required. The only semantic instruction provided by the user is the text prompt.

## Slide 05 — What does one camera contribute?

**Time:** 04:15–05:35 (1:20)  
**Presenter cue:** Explain aligned geometry maps, the two feature streams, and fusion.

One view produces four perfectly aligned geometric images: the neutral render, depth, normals, and a mask of visible surface. These control the generated appearance and also tell us which pixels can safely return to the 3D object. Diffusion contributes 1,280 channels from a 32-by-32 UNet feature map. It tends to vary smoothly and carries broad spatial organisation. DINOv2 contributes 768 channels from a 37-by-37 patch grid. It sees the completed image and often separates regions more sharply, although it also inherits anything the generator hallucinates. After resizing and separate normalisation, the two streams are concatenated with equal weights. The three small colour maps on the right are PCA visualisations, not the actual descriptor used for matching. Their role is to make the complementary behaviour visible: smooth global structure from diffusion, more discrete image semantics from DINO, and both in the fused descriptor.

## Slide 06 — Are 16 views really 16?

**Time:** 05:35–06:45 (1:10)  
**Presenter cue:** Use the repeated images to explain the camera-grid bug and its implication.

This implementation detail turned out to matter. The original sampler combines a grid of azimuths and elevations, but elevation is swept through a full 360 degrees. Different angle pairs therefore produce the same physical camera. In a run that requests 16 views, I found only six distinct viewpoints: eight cameras collapse to the two poles and the remaining views form repeated pairs. Repetition does not change a simple average very much, but it spends the most expensive part of the computation—diffusion generation—without showing the model any new surface. I implemented Fibonacci and insect-specific samplers to avoid that loss of coverage. For comparability, the two cricket experiments shown later retain the published 16-view grid. The larvae use 16 distinct insect-specific views. This is an important caveat when we compare results.

## Slide 07 — One prompt changes the descriptor

**Time:** 06:45–08:00 (1:15)  
**Presenter cue:** Point out wings/body changes and connect them to downstream features.

The prompt is the only semantic input, so its effect deserves attention. With the broad prompt “insect,” Stable Diffusion may reinterpret the same silhouette as a winged, simplified insect. A more specific cricket prompt follows the body plan more closely. The bottom row shows that the fused feature map changes as well. This is not proof that a longer prompt is always better, because diffusion is stochastic, but it shows that prompt choice changes the signal consumed by both feature extractors. There are two related implementation findings. Image-to-image strength means that only 24 of the requested 30 denoising steps actually execute, and the implementation resets the random seed for every camera, so views reuse the same noise sequence. The process is repeatable, but stochastic errors can remain correlated across views—potentially important when several legs look similar.

## Slide 08 — How does a CT volume become Diff3F-ready?

**Time:** 08:00–09:20 (1:20)  
**Presenter cue:** Describe the CT-to-surface chain while following the three images.

BugNIST contains 512-by-256-by-256 X-ray volumes, while Diff3F expects a surface. I first crop around the specimen and inspect orthogonal slices and maximum-intensity projections. A threshold creates a binary mask; small components are removed, gaps are closed and filled, and only the largest connected component is retained. Otsu is useful as an initial estimate, but it is not reliable enough by itself because the scans contain cuticle, soft tissue, mounting material, air, and artefacts. Marching cubes extracts the surface. The immediate result, in the middle, contains voxel-scale staircasing and far more triangles than the clean objects used in the original Diff3F paper. Taubin smoothing suppresses high-frequency roughness with limited shrinkage, vertex clustering simplifies the mesh, and a final manual rotation establishes a common body orientation. The right image is the geometry used for descriptor extraction.

## Slide 09 — How much geometry is enough?

**Time:** 09:20–10:25 (1:05)  
**Presenter cue:** Use the three big metrics; be careful not to claim linear timing.

I tested the same specimen at three mesh densities. The raw marching-cubes reconstruction had about 216,000 faces and 107,000 vertices. The practical choice retained roughly 50,000 faces and 24,000 vertices. The large-scale anatomy remained recognisable, and PCA visualisations of the final descriptors kept a similar spatial organisation. The smaller mesh also reduced the half-precision descriptor payload from about 418 to 94 mebibytes because there are many fewer descriptor rows. In the recorded single-view run it took 79.8 seconds instead of 134.5 seconds, a reduction of about 41 percent. A 75,000-face version happened to run faster still, which shows normal GPU-run variation, so I do not claim a linear timing law. The evidence supports 50,000 faces as a useful compromise between visible anatomy, storage, and practical runtime.

## Slide 10 — Four specimens. Three tests.

**Time:** 10:25–11:25 (1:00)  
**Presenter cue:** Introduce the two adult crickets and two larvae, then the evaluation layers.

The evaluation uses four processed specimens: a brown cricket, a black cricket, a mealworm, and a soldier fly larva. Every surface point keeps the full 2,048-dimensional descriptor. I evaluate the representation at three levels. First, label-free diagnostics ask whether nearest-neighbour correspondences are internally coherent across all 12 ordered source-to-target pairs. Second, shared PCA and shared k-means ask whether descriptors organise comparable body regions without seeing coordinates or labels. Third, manual anatomical landmarks test the question we ultimately care about: when I select a known point, does its best descriptor match land near the same anatomical point on another specimen? Each test is stricter than the previous one, so agreement across them is more meaningful than a single attractive visualisation.

## Slide 11 — Which pair is most coherent?

**Time:** 11:25–12:50 (1:25)  
**Presenter cue:** Read the heatmap by rows as source and columns as target; highlight asymmetry.

This heatmap shows cycle consistency within 10 percent of the source bounding-box diagonal. For each ordered pair, I sample 300 source vertices, match each one by cosine similarity, and then match it back. A cycle is successful if it returns close to where it started. The adult crickets are clearly the strongest and most symmetric pair: 70 percent in one direction and 72 percent in the other. The larval pair preserves some common structure, but is more direction-dependent. Adult-to-larva comparisons are mostly weaker and show more many-to-one collapse. An important lesson is that median cosine similarity alone can still look convincing when geometry is incoherent. Cycle consistency, target reuse, and distance preservation expose failures that similarity hides. Also note that matching is directional: nearest-neighbour search from A to B is not guaranteed to invert the search from B to A.

## Slide 12 — Do unlabeled features discover body parts?

**Time:** 12:50–14:05 (1:15)  
**Presenter cue:** Explain shared PCA versus clustering and why a common basis/model matters.

Here the two cricket descriptor fields are analysed jointly. The left column uses a PCA basis fitted to both specimens, so equal colours refer to the same directions in descriptor space. The middle and right columns show one k-means model fitted to both insects at K equals 6 and K equals 10. Clustering uses all 2,048 normalised feature dimensions—no vertex coordinates, connectivity, specimen identity, or anatomical labels. The same broad organisation appears on both crickets, and some clusters concentrate on protruding structures such as legs and antennae. That is evidence that the descriptor carries part-level information. But PCA is only a visual projection, and clusters are not automatically anatomical segments. The next slide checks the cluster identities at manual landmarks to see whether this apparent organisation transfers to named points.

## Slide 13 — 8 of 9 landmarks share a cluster

**Time:** 14:05–15:10 (1:05)  
**Presenter cue:** Frame this as coarse type transfer, not dense segmentation.

At K equals 10, eight of the nine cricket landmarks receive the same shared cluster ID on both specimens. Head tip matches head tip, thorax centre matches thorax centre, both antenna bases agree, both foreleg bases agree, and most of the remaining landmarks agree as well. This is a strong sign that broad anatomical type transfers without supervised training. The failure is revealing: left and right copies of an appendage often share the same descriptor category, and one right hind-leg base receives a different cluster. Increasing K does not simply solve the problem; eventually it fragments the two insects differently. So the clustering supports a precise claim: Diff3F organises coarse parts consistently. It does not yet provide a reliable, side-aware dense segmentation.

## Slide 14 — Can it find the exact point?

**Time:** 15:10–16:05 (0:55)  
**Presenter cue:** Explain the manual landmark benchmark in one clean sequence.

The strictest test uses manual landmarks placed in Blender. The crickets have nine each: head tip, thorax centre, abdomen tip, two antenna bases, two foreleg bases, and two hind-leg bases. For a selected source landmark, I take its descriptor and compare it with every target descriptor using cosine similarity. The top match becomes the prediction. Its spatial distance from the manual target is divided by the target bounding-box diagonal, making errors comparable across scale. PCK at a chosen percentage reports how many predictions fall inside that tolerance. I evaluate both directions separately because independent nearest-neighbour matching is asymmetric. The labels define the queries and the expected answers; they never modify the descriptors or the matching rule.

## Slide 15 — Where does correspondence break?

**Time:** 16:05–17:25 (1:20)  
**Presenter cue:** Describe the bar chart as worst error across the two cricket directions.

This chart takes the worse error across the two cricket directions for each landmark. The five central or distinctive landmarks—both antenna bases, head tip, thorax centre, and abdomen tip—remain below three percent. In each direction, seven of nine landmarks are within a 10 percent tolerance. The large errors are concentrated on repeated legs. The worst right hind-leg result reaches almost 39 percent; forelegs can also jump dramatically depending on direction. That pattern is more informative than the mean error. Diff3F has not lost all anatomical information: it is accurate on the central body and distinctive endpoints. Instead, it struggles to identify which instance of a repeated, approximately symmetric structure is intended. This is the difference between recognising “leg-like” and recognising “this particular right hind leg.”

## Slide 16 — Same idea. Opposite outcome.

**Time:** 17:25–18:45 (1:20)  
**Presenter cue:** Use the screenshots to make direction-dependent appendage confusion concrete.

These two examples come from the same black-to-brown direction. On the left, the right foreleg prediction lands close to the manual target, with only 2.02 percent normalised error. On the right, the left foreleg prediction lands in a different region and produces 27.94 percent error. The source descriptor is still finding something semantically plausible, but not the correct instance. Independent nearest-neighbour matching imposes no mutuality, no left–right constraint, and no spatial smoothness across neighbouring queries. Several source points can collapse onto the same attractive target. This is why a high cosine score is not sufficient evidence of anatomical correctness. For repeated structures, correspondence needs additional context—global consistency, orientation, geodesic relations, or weak side labels—not just local descriptor similarity.

## Slide 17 — Smooth features can still be wrong

**Time:** 18:45–20:05 (1:20)  
**Presenter cue:** Contrast the attractive shared-PCA gradients with the wrong-end failure.

The larvae provide the clearest warning against over-interpreting colour maps. Their shared-PCA fields vary smoothly along the main body axes, which looks organised and suggests a meaningful head-to-tail gradient. Yet exact matching is asymmetric. The head transfers reasonably in both directions, but the thorax fails in both. In the soldier-fly-to-mealworm direction, the abdomen-tip prediction reaches the opposite end of the mealworm, producing an error of 86.49 percent of the target diagonal. A descriptor field can therefore be smooth and structured while still assigning the wrong semantic orientation. The general image generator may not understand specialised larval anatomy well enough to distinguish two visually similar ends, and the pipeline has no explicit head-tail constraint. This is a small three-landmark test, not a population accuracy claim, but the failure mode is unambiguous.

## Slide 18 — Part identity is not point identity

**Time:** 20:05–21:05 (1:00)  
**Presenter cue:** Deliver the main conclusion in one sentence, then qualify it.

The three experiments converge on one interpretation. Diff3F transfers useful part-level semantics to unseen CT-derived geometry without task-specific training. The two adult crickets show strong cycle consistency, shared clusters line up at eight of nine landmarks, and central anatomical points match accurately. But broad part identity is not the same as exact point identity. A descriptor can correctly say “this is a leg” while choosing the wrong leg, the wrong side, or—in the larval case—the wrong end. Input preparation, camera coverage, prompts, diffusion priors, and the matching rule all affect that boundary. So the method is convincing as an exploratory semantic feature generator, but it is not yet a complete side-aware dense-correspondence solution.

## Slide 19 — What would make it reliable?

**Time:** 21:05–22:10 (1:05)  
**Presenter cue:** Present three concrete next steps tied directly to observed failure modes.

Three next steps follow directly from the results. First, stronger ground truth: more specimens, multiple annotators, and region-based labels would separate model error from uncertainty in choosing one exact vertex. Second, a more suitable semantic prior: comparing the general Stable Diffusion model with an insect- or specimen-specialised image model would test whether larval ambiguity comes from the generator. Third, side-aware and globally coherent matching: top-k candidates, geodesic relationships, cycle constraints, and a few weak semantic anchors could distinguish repeated appendages without requiring dense manual segmentation. I would also ablate camera samplers, prompts, seeds, and the separate diffusion and DINO contributions. The goal is to preserve Diff3F’s zero- or low-label advantage while adding exactly the structure that the failure cases show is missing.

## Slide 20 — Can a CT insect gain semantics?

**Time:** 22:10–22:40 (0:30)  
**Presenter cue:** Close with the qualified answer and invite questions.

So, can an untextured CT insect gain semantics from pretrained 2D models? Yes—coarsely and usefully. Diff3F discovers organised body regions and transfers several distinctive landmarks, but exact correspondence remains vulnerable to symmetry, repeated legs, and specialised anatomy. That qualified result is the main contribution of this investigation. Thank you, and I’m happy to take questions.

