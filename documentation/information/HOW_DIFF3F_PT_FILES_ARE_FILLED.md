# How Diff3F `.pt` Files Are Filled

This note explains how the final Diff3F `.pt` tensor is created and filled for
meshes and point clouds.

The short answer is:

```text
Yes, the output tensor size is known before the views are processed.

For a mesh:
  shape = number of mesh vertices x 2048

For a point cloud:
  shape = number of points x 2048
```

Then each rendered view contributes feature vectors to the rows that are visible
from that camera. At the end, every row is averaged. Rows that were never seen
are filled by copying the descriptor from the nearest observed 3D location.

## The Output Tensor

In `diff3f.py`, the descriptor size is:

```python
FEATURE_DIMS = 1280 + 768
```

That means:

```text
1280 dimensions from Stable Diffusion UNet features
 768 dimensions from DINOv2 image features
----
2048 dimensions total
```

So if a point cloud has 20,000 points:

```text
features.shape = [20000, 2048]
```

If a mesh has 107,066 vertices:

```text
features.shape = [107066, 2048]
```

The row order is the geometry order:

```text
mesh.vertices[i]       <-> features[i]
pointcloud.points[i]   <-> features[i]
```

This is why the `.pt` file must stay paired with the exact mesh or point cloud
that produced it.

## Two Tensors Are Created

The code does not only create the final feature table. It also creates a count
table.

For meshes:

```python
ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS)).half()
ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()
```

For point clouds:

```python
ft_per_point = torch.zeros((len(points), FEATURE_DIMS)).half()
ft_per_point_count = torch.zeros((len(points), 1)).half()
```

The meaning is:

```text
ft_per_vertex[i]       = sum of all feature vectors assigned to vertex i
ft_per_vertex_count[i] = number of assignments received by vertex i
```

The same logic applies to point clouds:

```text
ft_per_point[i]       = sum of all feature vectors assigned to point i
ft_per_point_count[i] = number of assignments received by point i
```

So during the view loop, the tensor is not immediately storing the final average.
It stores accumulated evidence.

## What Happens In One View

For each camera view, the pipeline roughly does this:

```text
3D geometry
  |
  v
render 2D image + depth map + optional normal map
  |
  v
run Stable Diffusion / ControlNet on that view
  |
  v
extract UNet features and DINOv2 features per image pixel
  |
  v
combine them into one 2048-D descriptor per visible pixel
  |
  v
send each visible pixel descriptor back to the 3D vertex/point it came from
```

A useful mental model:

```text
Each view creates a temporary 2D feature image.

Each visible pixel in that image has:
  - a position in the image
  - a depth
  - a 2048-D descriptor
  - a link back to some 3D surface location
```

Then the 2048-D pixel descriptor is added to the relevant 3D row.

## How The 2048-D Pixel Feature Is Made

For one rendered view:

```text
rendered image + depth/normal controls + prompt
  |
  v
Stable Diffusion / ControlNet
  |
  v
UNet feature map, 1280 channels
```

The final generated image is also sent through DINOv2:

```text
final diffusion image
  |
  v
DINOv2 feature map, 768 channels
```

Both feature maps are aligned back to the render resolution, then concatenated:

```python
aligned_features = torch.hstack([
    diffusion_features * 0.5,
    dino_features * 0.5,
])
```

So each visible pixel receives:

```text
2048-D pixel descriptor =
  [1280-D diffusion descriptor, 768-D DINO descriptor]
```

The `.pt` descriptor does not come from the saved debug PNGs. The PNGs are human
visualizations. The real tensor comes from internal model feature maps.

## Mesh Case: Pixels Are Mapped Back To Nearby Vertices

For meshes, the renderer sees triangles and pixels, not isolated vertices.

The code takes every visible pixel:

```text
pixel x/y + pixel depth
  |
  v
unproject through the camera
  |
  v
3D world coordinate on the rendered surface
```

Then it finds nearby mesh vertices using a ball query:

```text
rendered surface coordinate
  |
  v
all mesh vertices inside a small radius
```

The radius is controlled by:

```text
ball_drop_radius = mesh_size * tolerance
```

where `mesh_size` is estimated from the maximum distance across the mesh.

Important detail:

```text
One visible pixel can contribute to several nearby vertices.
```

This happens because the default mesh path uses `ball_query(..., K=100, radius=...)`.
So the mapping is not always:

```text
one pixel -> exactly one vertex
```

It is more like:

```text
one visible surface pixel -> nearby mesh vertices within tolerance
```

For every matched vertex:

```text
ft_per_vertex[vertex_id] += pixel_feature
ft_per_vertex_count[vertex_id] += 1
```

So a vertex may receive contributions from many pixels and many views.

## Point-Cloud Case: Pixels Contain Point Indices

For point clouds, the mapping is more direct.

The point-cloud renderer returns a `point_indices` image. For each pixel, it says
which original point landed there.

```text
visible pixel
  |
  v
point index from renderer
  |
  v
row in ft_per_point
```

For every visible point assignment:

```text
ft_per_point[point_id] += pixel_feature
ft_per_point_count[point_id] += 1
```

With the default:

```text
--points-per-pixel 1
```

each pixel stores only the closest visible point. If `--points-per-pixel` is
larger, more than one point can receive the same pixel feature.

The point-cloud mapping is therefore:

```text
pixel descriptor -> original point row
```

while the mesh mapping is:

```text
pixel descriptor -> nearby mesh vertex rows
```

## What Happens Across Many Views

The same accumulation process repeats for every camera view.

Example with 16 views:

```text
view 0 sees some points/vertices -> add their descriptors
view 1 sees some points/vertices -> add their descriptors
view 2 sees some points/vertices -> add their descriptors
...
view 15 sees some points/vertices -> add their descriptors
```

A single row can receive:

```text
0 assignments
1 assignment
many assignments from one view
many assignments from several views
```

The count is not exactly "number of views that saw this point." It is the number
of feature assignments that row received.

For a mesh, one view can assign many nearby pixel descriptors to the same vertex.
For a point cloud, one point can appear in several pixels or views depending on
point radius, resolution, and camera position.

## Averaging At The End

After all views are processed, the accumulated sum is divided by the count:

```python
idxs = (ft_per_vertex_count != 0)[:, 0]
ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
```

For point clouds:

```python
idxs = (ft_per_point_count != 0)[:, 0]
ft_per_point[idxs, :] = ft_per_point[idxs, :] / ft_per_point_count[idxs, :]
```

So the final descriptor for a seen vertex or point is:

```text
final descriptor =
  average of all pixel descriptors assigned to that row
```

In equation form:

```text
features[i] =
  sum(all descriptors assigned to row i) / count(row i)
```

## What Happens To Rows That Were Never Seen

Some rows may have count zero:

```text
ft_per_vertex_count[i] == 0
```

or:

```text
ft_per_point_count[i] == 0
```

This means that vertex or point never received direct feature evidence from any
rendered view.

Reasons this can happen:

```text
the point was fully occluded
the render resolution was too low
the point cloud radius was too small
the mesh tolerance was too strict
the camera views missed that region
the geometry had tiny structures that were not rasterized clearly
```

The code then fills missing rows by copying from the nearest observed 3D
location:

```text
missing vertex/point
  |
  v
find nearest vertex/point with count > 0
  |
  v
copy that descriptor
```

This is implemented in:

```python
copy_missing_features_from_nearest(...)
```

This fill step is practical, but it is weaker than direct observation.

Safe interpretation:

```text
A directly observed descriptor is supported by rendered image evidence.
A copied descriptor is an approximation based only on nearest 3D position.
```

## Final Save

After averaging and filling missing rows, the tensor is returned and saved:

```python
torch.save(features, save_path)
```

The saved `.pt` file usually contains one PyTorch tensor:

```text
shape: [N, 2048]
dtype: usually float16
```

where `N` is:

```text
number of mesh vertices
```

or:

```text
number of point-cloud points
```

## Your Mental Model Is Correct

Your understanding can be written like this:

```text
1. Load a geometry with N vertices or N points.
2. Create an empty N x 2048 feature table.
3. Create an empty N x 1 count table.
4. Render the geometry from many views.
5. For each view, compute a 2048-D feature vector for each visible pixel.
6. Map each visible pixel back to the mesh vertices or point-cloud points it represents.
7. Add that pixel feature to the corresponding row.
8. Increase that row's count.
9. After all views, divide each row by its count.
10. If a row was never seen, copy the feature from the nearest seen row.
11. Save the final N x 2048 tensor as `.pt`.
```

The only correction is this:

```text
The view does not directly "look at vertices" in the mesh case.
It looks at rendered pixels on triangle surfaces, then maps those pixels back to nearby vertices.
```

For point clouds, it is closer to directly looking at points because the renderer
returns point indices for visible pixels.

## Small Example

Imagine a tiny point cloud with only 5 points and 3-D descriptors instead of
2048-D descriptors.

Start:

```text
features = zeros([5, 3])
counts   = zeros([5, 1])
```

After view 0:

```text
point 0 receives [1, 0, 0]
point 1 receives [0, 1, 0]
point 3 receives [0, 0, 1]
```

Accumulated:

```text
features[0] = [1, 0, 0], count[0] = 1
features[1] = [0, 1, 0], count[1] = 1
features[2] = [0, 0, 0], count[2] = 0
features[3] = [0, 0, 1], count[3] = 1
features[4] = [0, 0, 0], count[4] = 0
```

After view 1:

```text
point 0 receives [0.8, 0.2, 0]
point 2 receives [0, 1, 1]
```

Accumulated:

```text
features[0] = [1.8, 0.2, 0], count[0] = 2
features[1] = [0, 1, 0],     count[1] = 1
features[2] = [0, 1, 1],     count[2] = 1
features[3] = [0, 0, 1],     count[3] = 1
features[4] = [0, 0, 0],     count[4] = 0
```

Average:

```text
features[0] = [0.9, 0.1, 0]
features[1] = [0, 1, 0]
features[2] = [0, 1, 1]
features[3] = [0, 0, 1]
features[4] = missing
```

Fill missing point 4:

```text
point 4 copies the descriptor from its nearest observed neighbor
```

The real project does the same thing, except:

```text
5 rows become 20,000 or 100,000 rows
3 descriptor dimensions become 2048 descriptor dimensions
2 views become 4, 16, 25, or more views
```

## Why This Matters For Correspondence

When we later match two insects, we compare rows:

```text
source_features[i]  vs  target_features[j]
```

That comparison only makes sense if:

```text
features[i] really describes geometry location i
```

So for every geometry change:

```text
rotate and re-export with changed vertex order
smooth
decimate
resample
convert mesh to point cloud
```

compute a fresh `.pt` descriptor.

The `.pt` file is not a detachable texture. It is a row-by-row descriptor table
attached to one exact geometry file.

