# What The Hell Are The `.pt` Descriptor Files?

This document explains what we are building in this repository, especially the
Diff3F `.pt` files produced for BugNIST meshes and point clouds.

The short version:

```text
A .pt file is a table of learned feature vectors.

Rows    = one mesh vertex or one point-cloud point
Columns = 2048 learned descriptor values

So if a point cloud has 20,000 points:

  output.pt  ->  tensor shape [20000, 2048]

Row 0 describes point 0.
Row 1 describes point 1.
...
Row 19999 describes point 19999.
```

These descriptors are not colors, not coordinates, not labels, and not a new
mesh. They are learned numerical summaries of what the diffusion/DINO models
"see" at each location on the shape.

Think of each row as a high-dimensional fingerprint for a tiny place on the
insect surface.

```text
               one point on the insect
                         |
                         v
        +-----------------------------------+
        | 2048 numbers describing that spot |
        +-----------------------------------+
          0.12  -0.03  0.48  ...  -0.19

        Not human-readable by itself.
        Useful because we can compare it to other fingerprints.
```

## The Pipeline In One Picture

For meshes:

```text
BugNIST TIFF volume
        |
        v
crop ROI + threshold + keep largest component
        |
        v
marching cubes
        |
        v
mesh vertices + faces
        |
        v
render many 2D views
        |
        v
Stable Diffusion / ControlNet + DINOv2 features per image pixel
        |
        v
map image-pixel features back to mesh vertices
        |
        v
mesh_diff3f.pt
shape: [num_vertices, 2048]
```

For point clouds:

```text
BugNIST TIFF volume
        |
        v
crop ROI + threshold + keep largest component
        |
        v
sample surface points
        |
        v
point cloud
        |
        v
render many 2D views
        |
        v
Stable Diffusion / ControlNet + DINOv2 features per image pixel
        |
        v
map image-pixel features back to point indices
        |
        v
pointcloud_diff3f.pt
shape: [num_points, 2048]
```

The mesh and point-cloud versions are conceptually the same. The only real
difference is how we attach pixel features back to 3D geometry.

## What Is Inside A `.pt` File?

`.pt` is PyTorch's saved-file format. In this project, we usually save a single
tensor:

```python
features = torch.load("some_shape_diff3f.pt", map_location="cpu")
print(features.shape)
```

Example outputs:

```text
mesh:
  bcrick mesh has 107,066 vertices
  bcrick_..._diff3f.pt has shape [107066, 2048]

point cloud:
  bcrick point cloud has 20,000 points
  bcrick_..._20k_diff3f.pt has shape [20000, 2048]
```

The important contract is:

```text
geometry row i  <---->  descriptor row i
```

For a mesh:

```text
mesh.vertices[i]  <---->  features[i]
```

For a point cloud:

```text
pointcloud.points[i]  <---->  features[i]
```

This is why the exact geometry file matters. If you resample, reorder, clean,
decimate, or re-export the geometry after computing features, the `.pt` file no
longer lines up with it.

## What Is A Feature Vector?

A feature vector is a list of numbers used to describe something.

A simple handmade feature vector for a point on an insect could be:

```text
[height, local_curvature, distance_to_center, brightness]
```

That would be easy to interpret, but weak.

Diff3F gives us a learned vector instead:

```text
[f0, f1, f2, f3, ..., f2047]
```

Each value is a coordinate in a learned feature space. The dimensions do not have
simple names like:

```text
f12 = legness
f93 = antennaness
f427 = shellness
```

That is not how neural features usually work.

Instead, meaning is distributed across many dimensions. A rough mental model is:

```text
one coordinate by itself:        usually not meaningful
the full 2048-number pattern:    meaningful
distance between two patterns:   useful
```

Like a chord in music: one note is not the whole identity, but the combination
of notes creates a recognizable structure.

## Why 2048 Numbers?

The descriptor is made by concatenating two different feature sources:

```text
Diffusion UNet features:   1280 dimensions
DINOv2 image features:      768 dimensions
-------------------------------------------
Total:                     2048 dimensions
```

So each row is:

```text
features[i] = [diffusion_feature_0 ... diffusion_feature_1279,
               dino_feature_0      ... dino_feature_767]
```

In the code this is the constant:

```python
FEATURE_DIMS = 1280 + 768
```

## What Does The Diffusion Part Contribute?

The diffusion part comes from Stable Diffusion with ControlNet.

The idea is:

```text
1. Render the 3D shape into a 2D image.
2. Use depth and normal maps to tell Stable Diffusion the shape structure.
3. Ask it to interpret the image with a prompt, for example "insect".
4. Extract intermediate UNet features while it processes that image.
```

These intermediate UNet features are not the final generated image. They are
internal activations from the diffusion model. They tend to contain semantic and
visual information learned from huge image datasets.

The prompt matters. If we use:

```text
"insect"
```

then the model is encouraged to interpret the rendered form as insect-like.

The prompt does not create ground-truth anatomy. It biases the feature space.
For correspondence experiments, keep the prompt fixed across all specimens
unless you are deliberately testing prompt sensitivity.

## What Does The DINOv2 Part Contribute?

DINOv2 is a self-supervised vision model. It produces image features that are
often useful for matching parts between images.

In our case:

```text
rendered / diffusion-processed image
        |
        v
DINOv2
        |
        v
features per image patch
        |
        v
aligned back to every pixel
```

DINOv2 tends to add visual consistency and part-like information. It is not
trained specifically on BugNIST insects, but its learned visual descriptors can
still carry useful structure.

## How Does A 2D Pixel Become A 3D Descriptor?

Diff3F works through rendered views.

Imagine the 3D object being photographed from many angles:

```text
                 camera
                   |
                   v
             +-----------+
             |  2D view  |
             +-----------+
                  /|\
                   |
              3D insect
```

For each rendered view:

```text
visible 3D surface location
        |
        v
2D pixel in rendered image
        |
        v
2048-D pixel descriptor
        |
        v
copy / accumulate descriptor back onto the 3D location
```

After many views, a vertex or point may have been seen several times. We average
the descriptors collected for it.

```text
descriptor(point i) =
    average of all pixel descriptors that saw point i
```

This averaging is important because one view may see only one side or one
lighting/depth configuration. Multiple views give a more stable descriptor.

## Mesh Mapping Vs Point-Cloud Mapping

For meshes, we render triangles. A visible pixel has a depth. The code
unprojects that pixel back into 3D and finds nearby mesh vertices.

```text
pixel -> depth -> 3D world coordinate -> nearby mesh vertices
```

For point clouds, PyTorch3D's point rasterizer directly tells us which point
landed on each pixel.

```text
pixel -> point index
```

That makes the point-cloud mapping more direct:

```text
pixel descriptor -> point row in the original point cloud
```

## What Happens To Points That Are Never Seen?

Some vertices or points may not receive features:

```text
hidden by another part
too small at render resolution
missed because the point radius is too small
not visible from the chosen views
```

The code fills missing descriptors by copying the descriptor from the nearest
3D location that did receive one.

```text
missing point
     |
     v
find nearest visible point
     |
     v
copy its descriptor
```

This is practical, but it is also a limitation. A copied descriptor is not as
trustworthy as one directly observed in rendered views.

If many points are missing, try:

```text
more views:        --num-views 9, 16, 25
higher resolution: --height 512 --width 512
larger point size: --point-radius 0.012 or 0.015
```

## Why Do We Use Several Views?

A single 2D render cannot see the whole insect.

```text
front view sees front surface
back view sees back surface
side view sees side structures
top view sees dorsal structure
```

Diff3F uses a grid of camera views around the object.

Important:

```text
--num-views must be a perfect square

4   = 2 x 2 view grid
9   = 3 x 3 view grid
16  = 4 x 4 view grid
25  = 5 x 5 view grid
100 = 10 x 10 view grid
```

More views usually means better coverage, but also more GPU time.

For local experiments:

```text
4 views:  smoke test, fast, low quality
9 views:  better first real test
16+ views: more serious, preferably on a stronger GPU/HPC
```

## What Are We Developing?

We are developing a pipeline that turns BugNIST insect CT geometry into dense
learned descriptors.

In plain terms:

```text
We want every small place on an insect surface to have a numerical identity.
```

That identity can then be used for:

```text
1. Visualizing learned structure on one insect.
2. Comparing descriptor patterns between insects.
3. Finding possible corresponding points between two insects.
4. Measuring whether those correspondences look stable or collapse.
```

The final ambition is anatomical correspondence:

```text
this point on insect A  <---->  matching point on insect B
```

But we are not there yet in a proven way. Right now, we have a descriptor
pipeline and exploratory diagnostics. We do not yet have ground-truth landmarks
that prove anatomical accuracy.

## What Does A Colored Feature `.ply` Show?

The `.pt` file is 2048-dimensional, so we cannot directly look at it.

For visualization, we compress the 2048 numbers down to 3 numbers using PCA:

```text
2048-D descriptor
        |
        v
PCA
        |
        v
3-D color coordinate
        |
        v
RGB color
```

This produces colored meshes or colored point clouds.

Important:

```text
The colors are a visualization of the descriptors.
The colors are not the descriptors themselves.
```

If two points have similar colors in a shared-PCA visualization, it suggests
their descriptors are similar along the main PCA directions. But the real
matching still happens in the full 2048-D space.

## Single PCA Vs Shared PCA

Single-shape PCA:

```text
fit PCA on one insect
color that insect
```

This is useful to inspect structure within one shape. But colors are not
comparable across insects because each insect gets its own PCA basis.

Shared PCA:

```text
fit one PCA basis using features from all insects
color every insect with that same basis
```

This is what we want for visual comparison.

```text
Single PCA:
  red on bcrick does not necessarily mean red on sfaar

Shared PCA:
  red on bcrick and red on sfaar are at least using the same color coordinate
```

## How Correspondence Works

Once every point has a descriptor, we can compare descriptors.

For source point `i`:

```text
source_features[i] = 2048-D vector
```

We search the target for the most similar vector:

```text
target_features[j] = nearest descriptor in feature space
```

Then we say:

```text
source point i matches target point j
```

The current scripts use cosine similarity:

```text
cosine similarity close to 1: vectors point in a similar direction
cosine similarity close to 0: weak relation
cosine similarity negative: very different direction
```

Cosine similarity cares more about descriptor direction than descriptor length.
This is common for neural descriptors.

## What The Metrics Mean

Without ground-truth landmarks, the metrics are diagnostics, not proof.

### Cosine Similarity

```text
How similar is the matched pair in feature space?
```

Higher is usually better, but high cosine alone does not prove anatomical
correctness.

### Mutual Match

```text
A -> B gives target point j
B -> A from that point gives back source point i
```

If many matches are mutual, the descriptor space is more stable. If zero are
mutual, matches are likely ambiguous.

### Cycle Consistency

```text
source A -> target B -> source A again
```

If we return near the original source point, the match is more believable.

### Unique Target Ratio

```text
Do many source points collapse onto the same few target points?
```

A low unique target ratio is a warning sign. It means the descriptor may be
matching many places to one "popular" location.

### Pairwise Distance Preservation

```text
If two source points are far apart, are their matched target points also far apart?
```

This checks whether the matching keeps rough geometry structure.

## What These Features Are Not

They are not:

```text
manual landmarks
segmentation labels
explicit insect anatomy labels
physical measurements
voxel intensities
surface normals
coordinates
proof of correspondence
```

They are:

```text
learned descriptors distilled from image foundation models
attached densely to 3D geometry through rendering
useful for visualization and matching experiments
```

## The Most Important Fragility

The descriptor tensor and the geometry must stay married.

```text
GOOD:
  pointcloud_A.ply
  pointcloud_A_diff3f.pt

BAD:
  pointcloud_A_resampled.ply
  pointcloud_A_diff3f.pt
```

Even if the resampled cloud looks visually identical, its point order and point
count probably changed. Then row `i` no longer means the same 3D location.

## A More Intuitive Analogy

Imagine the insect surface as a city at night.

```text
each point/vertex = one house
each rendered view = a drone photo
Diff3F/DINO = a system that reads visual context from the photo
.pt file = a notebook with one fingerprint per house
```

The fingerprint does not say:

```text
house 413 is a bakery
```

It says something more like:

```text
house 413 has a visual/context signature similar to these other places
```

Then correspondence is:

```text
Find the house in another city whose fingerprint looks most similar.
```

This can work beautifully when the cities are organized similarly. It becomes
fragile when the cities have different shapes, missing neighborhoods, different
view coverage, or ambiguous repeated structures.

In insects, repeated legs, antennae, and body segments create exactly that kind
of ambiguity.

## Why Point Clouds Are Interesting Here

Meshes give us connected triangle surfaces. Point clouds give us sampled
surface locations without faces.

Meshes:

```text
pros:
  connected surface
  nice rendering
  normals are easier

cons:
  huge meshes can be heavy
  marching-cubes topology may contain artifacts
  vertex density depends on segmentation geometry
```

Point clouds:

```text
pros:
  fixed size, for example 20k or 100k points
  easier to compare computationally
  avoids huge triangle meshes

cons:
  no connectivity
  normals are approximate
  sparse point rendering can miss points unless radius/views are tuned
```

The point-cloud Diff3F pipeline is useful because it lets us test descriptors on
a controlled number of surface samples.

## How To Read A `.pt` Shape

Use this:

```powershell
python -c "import torch; f=torch.load('output\bugnist_pointcloud_features\bcrick_10_001_thr29_roi_keeplargest_ds1_20k_diff3f.pt', map_location='cpu'); print(f.shape, f.dtype)"
```

Expected:

```text
torch.Size([20000, 2048]) torch.float16
```

That means:

```text
20000 surface points
2048 descriptor values per point
float16 storage to save memory
```

For computation, the scripts often load them as `float32`:

```python
features = torch.load(path, map_location="cpu").float()
```

## How To Inspect One Row

```python
import torch

features = torch.load("some_diff3f.pt", map_location="cpu")
row = features[0]

print(row.shape)       # [2048]
print(row[:10])        # first 10 values
print(row.norm())      # vector magnitude
```

The first 10 values do not tell you much as humans. The row becomes useful when
compared to other rows.

## How To Compare Two Points

```python
import torch

features_a = torch.load("a_diff3f.pt", map_location="cpu").float()
features_b = torch.load("b_diff3f.pt", map_location="cpu").float()

a = torch.nn.functional.normalize(features_a[123], dim=0)
b = torch.nn.functional.normalize(features_b[456], dim=0)

score = torch.dot(a, b)
print(score)
```

That `score` is cosine similarity.

## What A Good Future Evaluation Looks Like

Right now we can say:

```text
These descriptors show structure.
Some matches look less random than pure noise.
The diagnostics are mixed.
```

What we cannot honestly say yet:

```text
This is anatomically correct.
```

To say that, we need manual landmarks:

```text
bcrick landmark 1: left antenna base
sfaar landmark 1: left antenna base
soldat landmark 1: left antenna base
...
```

Then we can measure:

```text
Does descriptor matching recover known landmarks?
How far is the predicted point from the true landmark?
What percentage of matches land within 5 percent of body size?
```

That would turn the project from "interesting descriptor experiment" into
"measured anatomical correspondence experiment."

## Current Mental Model

This is the thing to keep in your head:

```text
                 3D insect geometry
                         |
                         v
        many rendered images from many angles
                         |
                         v
         foundation-model image feature fields
                         |
                         v
         features copied back onto 3D locations
                         |
                         v
      one 2048-D descriptor per vertex or point
                         |
                         v
       visualization, matching, and diagnostics
```

Or, more compact:

```text
We are painting each 3D location with an invisible 2048-color paint.
PCA lets us see a 3-color shadow of it.
Correspondence compares the invisible full paint, not just the visible colors.
```

## The One-Sentence Explanation

We are building a rendering-based pipeline that uses pretrained image models to
assign a 2048-dimensional learned descriptor to every mesh vertex or point-cloud
point, so that insect surfaces can be visualized and compared by feature
similarity rather than by raw coordinates alone.

