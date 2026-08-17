# What The 2D Debug Images Mean

This document explains the 2D image-debug step added in:

```text
scripts/debug_2d_diffusion_view.py
```

The goal of that script is simple:

```text
pause the Diff3F pipeline in the middle
and save the 2D images/features that the AI sees
```

It is not meant to produce beautiful images. It is meant to answer:

```text
What is Stable Diffusion seeing?
What is ControlNet conditioning it with?
What does the generated image look like?
What do the extracted feature maps look like?
```

## The Big Picture

Diff3F does not look directly at the 3D insect.

It first turns the 3D shape into 2D views:

```text
3D point cloud / mesh
        |
        v
2D render from one camera
        |
        v
Stable Diffusion + ControlNet
        |
        v
2D AI-generated image + internal feature maps
        |
        v
DINOv2 image features
        |
        v
2048-D pixel descriptors
        |
        v
map those descriptors back to 3D points / vertices
```

So the 2D debug script captures this middle part:

```text
3D geometry -> 2D render -> AI image process -> 2D feature maps
```

## The Mental Model

Imagine we take a tiny studio photo of the insect.

But instead of only saving the photo, we save several layers:

```text
photo of insect silhouette
depth map
surface orientation map
AI's generated interpretation of the photo
AI feature maps
DINO feature maps
combined feature map
```

The final 3D descriptors come from these 2D feature maps.

The pictures are therefore diagnostic windows into the descriptor factory.

## What The AI Receives

The 2D AI gets three main pieces of information:

```text
1. input render
2. depth control image
3. normal control image
```

Plus a text prompt:

```text
"insect"
```

This means the AI is being told:

```text
Here is a rendered shape.
Here is its depth.
Here are approximate surface directions.
Please interpret it as an insect-like image.
```

## What ControlNet Is Doing

Stable Diffusion normally generates images from text.

ControlNet lets us add extra visual constraints.

In our case:

```text
Depth ControlNet:
  "Respect this 3D depth structure."

Normal ControlNet:
  "Respect this surface orientation structure."
```

So the model is not freely inventing any insect. It is pushed to follow the
rendered geometry.

That said, it is still a generative image model, so it may hallucinate texture,
color, contrast, or visual details. That hallucination is part of why the debug
images can look strange.

## What Each Saved Image Means

The debug folder contains images named in order.

Example folder:

```text
debug/pointcloud_bcrick_view0_ai
```

### `01_input_render.png`

This is the raw 2D render of the mesh or point cloud.

It is the closest thing to:

```text
what the camera sees before AI touches anything
```

For point clouds, it may look sparse, dotted, or soft depending on:

```text
point count
point radius
render resolution
camera angle
```

If this image is bad, everything after it becomes suspect.

Things to check:

```text
Does the insect silhouette look complete?
Are legs / antennae visible?
Is the object too tiny in the frame?
Is it too sparse or broken?
```

If the render is too sparse, increase:

```text
--point-radius 0.015
```

or use a denser point cloud.

### `02_depth_control.png`

This image tells ControlNet the relative depth of the visible surface.

Usually:

```text
brighter / darker = closer / farther depth
black background = no geometry
```

The exact brightness direction is less important than the depth structure.

This image answers:

```text
Does the AI know the 3D shape's front/back structure?
```

If depth is noisy, broken, or flat, the AI receives weak geometric guidance.

### `03_normal_control.png`

This image encodes approximate surface direction as color.

It is not normal RGB color. It is geometry color:

```text
red/green/blue channels represent x/y/z normal direction
```

For meshes, normals come from triangle surface normals.

For point clouds, our debug path estimates a screen-space normal map from depth.
That is less perfect than mesh normals, but useful for giving ControlNet more
shape information.

This image answers:

```text
Does the AI know how the visible surface is oriented?
```

If the normal map looks chaotic, the point cloud may be too sparse, the point
radius too small, or the view resolution too low.

### `04_visible_mask.png`

This is the most practical image.

White pixels:

```text
pixels that correspond to actual rendered geometry
```

Black pixels:

```text
background
```

Only white pixels are meaningful for mapping back to 3D.

This is important because Stable Diffusion and DINO produce features for the
whole square image, including the background. But Diff3F only wants features for
the insect surface.

So the mask says:

```text
Use features here.
Ignore features there.
```

### `05_denoise_step_*.png`

These are snapshots from inside the Stable Diffusion denoising process.

They are decoded versions of the latent image while the model is still working.

They are not exactly "what the AI is thinking." A better phrase is:

```text
rough previews of the image latent as it is being refined
```

A simplified diffusion process looks like:

```text
noisy latent
    |
    v
less noisy latent
    |
    v
more structured latent
    |
    v
final generated image
```

In the debug output:

```text
05_denoise_step_000...
05_denoise_step_002...
05_denoise_step_004...
...
```

These show the image becoming more structured over time.

Why can they look wild and colorful?

```text
low resolution
few denoising steps
strong generative model
the input render is very abstract
the model is trying to make an "insect photo" from a grey geometry render
```

Do not judge descriptor quality only by whether these snapshots look beautiful.
They are useful because they show whether the model is preserving the insect
shape or drifting away from it.

### `06_final_generated.png`

This is the final image produced by Stable Diffusion image-to-image.

This is the image DINOv2 sees.

It is also the image from which the custom pipeline extracts the diffusion UNet
feature map.

Important:

```text
The final generated image is not the final result of the project.
```

It is an intermediate image used to obtain better feature descriptors.

The generated image can look visually strange but still produce useful features.
The pipeline is not trying to make art. It is using the image model as a feature
extractor.

### `07_ai_change_map.png`

This image compares:

```text
raw input render
vs
final generated image
```

It shows amplified pixel difference.

Bright / colorful areas mean:

```text
the AI changed that part a lot
```

Dark / quiet areas mean:

```text
the AI kept that part closer to the original render
```

This helps answer:

```text
Where did Stable Diffusion hallucinate or reinterpret the input?
```

If the change map is extremely active everywhere, the AI may be overpowering the
geometry. Possible fixes:

```text
increase render quality
increase resolution
try more denoising steps
try a more specific prompt
reduce weird point-cloud sparsity
```

### `08_unet_feature_pca.png`

This visualizes the diffusion UNet feature map.

The real UNet feature at each pixel is:

```text
1280 numbers
```

We cannot display 1280 dimensions directly, so the script compresses them to RGB
using PCA:

```text
1280-D UNet feature
        |
        v
PCA to 3 dimensions
        |
        v
RGB color
```

This image answers:

```text
Which visible pixels does the diffusion model consider similar?
```

If two body regions have similar colors here, their diffusion features are
similar in this view.

Important:

```text
The colors are not labels.
```

Red does not mean "leg", blue does not mean "head", green does not mean
"abdomen". The colors are just a 3D projection of a 1280-D space.

### `09_dino_feature_pca.png`

This visualizes DINOv2 features.

The real DINO feature at each pixel is:

```text
768 numbers
```

Again, PCA compresses that to RGB.

DINOv2 is often good at visual grouping and part-like similarity in images.

This image answers:

```text
How does DINO group the visible insect pixels?
```

Sometimes DINO features look sharper or more part-aware than UNet features.
Sometimes they are noisy, especially if the generated image is strange.

### `10_combined_diff3f_feature_pca.png`

This is closest to the actual descriptor used by Diff3F.

It combines:

```text
1280-D diffusion UNet feature
768-D DINOv2 feature
-----------------------------
2048-D combined feature
```

Then it compresses the 2048-D vector to RGB for visualization.

This image answers:

```text
What does the full pixel descriptor structure look like in this view?
```

This is the most important feature-map image in the debug folder.

But remember:

```text
matching does not use this RGB image
matching uses the full 2048-D vector
```

The RGB image is only a shadow of the real descriptor.

### `contact_sheet.png`

This is just a visual summary of the main outputs.

Read it left-to-right, top-to-bottom:

```text
input render       depth control       normal control
visible mask       denoising preview   final generated
AI change map      UNet PCA            DINO PCA
combined PCA
```

The contact sheet is the quickest way to ask:

```text
Did the 2D view make sense?
Did the AI preserve the insect?
Are the feature maps structured on the insect body?
```

## What The Pictures Tell Us

The pictures tell us four things.

### 1. Whether The Geometry View Is Good

Look at:

```text
01_input_render.png
04_visible_mask.png
```

Good signs:

```text
clear insect silhouette
body parts visible
not too tiny
not too fragmented
```

Bad signs:

```text
missing legs or antennae
object cropped
point cloud looks like dust
only a tiny blob appears
```

If the geometry view is bad, the descriptors from that view will also be bad.

### 2. Whether ControlNet Has Useful Shape Guidance

Look at:

```text
02_depth_control.png
03_normal_control.png
```

Good signs:

```text
depth has clear body structure
normal map changes smoothly across surfaces
background is clean
```

Bad signs:

```text
depth is mostly flat
normal map is random speckle
shape is barely visible
```

If control images are poor, Stable Diffusion has less reason to respect the 3D
object.

### 3. Whether Stable Diffusion Is Preserving Or Inventing

Look at:

```text
05_denoise_step_*.png
06_final_generated.png
07_ai_change_map.png
```

Good signs:

```text
final image still roughly follows the insect silhouette
main body orientation is preserved
AI change map is strongest inside/near the object, not random everywhere
```

Bad signs:

```text
final image becomes unrelated scenery
insect shape disappears
AI invents large structures outside the mask
```

Some hallucination is expected. Total drift is a problem.

### 4. Whether Feature Maps Have Structure

Look at:

```text
08_unet_feature_pca.png
09_dino_feature_pca.png
10_combined_diff3f_feature_pca.png
```

Good signs:

```text
different regions have different colors
colors vary smoothly across body parts
left/right or repeated structures show some related patterns
background is muted
```

Bad signs:

```text
all visible pixels have almost the same color
colors look purely random
features mostly respond to background
body parts have no coherent regions
```

The combined PCA is the best one to inspect first.

## Why The AI Image Can Look Weird

This is important.

The generated image may look strange, colorful, or not like a clean insect.

That does not automatically mean the descriptor is useless.

Why?

Because we are not using Stable Diffusion as an artist. We are using it as a
feature machine.

The model's internal activations may still encode useful structure even when the
decoded image looks odd.

However:

```text
if the generated image completely ignores the input geometry,
then the features become suspicious
```

So the visual question is not:

```text
Is this a beautiful insect image?
```

The visual question is:

```text
Did the AI keep enough of the rendered insect structure
for its internal features to be meaningful?
```

## The Most Important Relationship

The 2D debug images are not separate from the final `.pt`.

They are the source material.

For one view:

```text
combined 2048-D pixel features
        |
        v
visible pixels only
        |
        v
map pixels back to 3D points / vertices
        |
        v
accumulate into .pt rows
```

After many views:

```text
point i seen in view 0 -> descriptor contribution
point i seen in view 1 -> descriptor contribution
point i seen in view 2 -> descriptor contribution
...
average all contributions
```

That average becomes:

```text
features[i]
```

inside the `.pt` file.

## A Tiny Example

Suppose one point on the insect is visible in three rendered views.

```text
view 0 pixel descriptor: [2048 numbers]
view 1 pixel descriptor: [2048 numbers]
view 2 pixel descriptor: [2048 numbers]
```

The final point descriptor is approximately:

```text
average(view0, view1, view2)
```

So the `.pt` row for that point is built from all the times that point appeared
in the 2D AI views.

## How To Use This Debug Script

Render/control-only check:

```powershell
python scripts\debug_2d_diffusion_view.py `
  --kind pointcloud `
  --input pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --prompt insect `
  --outdir debug\pointcloud_bcrick_view0_controls `
  --num-views 4 `
  --view-index 0 `
  --height 128 `
  --width 128 `
  --skip-ai
```

Full AI trace:

```powershell
python scripts\debug_2d_diffusion_view.py `
  --kind pointcloud `
  --input pointclouds\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1_20k.ply `
  --prompt insect `
  --outdir debug\pointcloud_bcrick_view0_ai `
  --num-views 4 `
  --view-index 0 `
  --height 128 `
  --width 128 `
  --num-inference-steps 10 `
  --denoise-interval 2
```

For a mesh:

```powershell
python scripts\debug_2d_diffusion_view.py `
  --kind mesh `
  --input meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --prompt insect `
  --outdir debug\mesh_bcrick_view0_ai `
  --num-views 4 `
  --view-index 0 `
  --height 128 `
  --width 128 `
  --num-inference-steps 10 `
  --denoise-interval 2
```

## What To Try When Something Looks Wrong

If the render is too sparse:

```text
increase --point-radius
try 0.015 or 0.02
```

If the insect is too small:

```text
use a different view
increase render resolution
check whether the point cloud is centered/scaled correctly
```

If the AI image drifts too much:

```text
try more denoising steps
try 20 or 30 instead of 10
try a more specific prompt
for example "micro CT render of an insect"
```

If feature PCA is mostly one color:

```text
the view may not contain enough visual variation
try another view index
try higher resolution
try more complete geometry
```

If the normal map is chaotic for point clouds:

```text
increase point radius
use a denser point cloud
compare against mesh mode
```

## What Not To Overinterpret

Do not read the PCA colors as anatomical labels.

This is wrong:

```text
red = legs
green = abdomen
blue = antennae
```

This is safer:

```text
similar colors = similar projected high-dimensional features
different colors = different projected high-dimensional features
```

Also, PCA colors from one debug run are not automatically comparable to PCA
colors from another debug run, because each run fits its own PCA visualization.

## The One-Sentence Explanation

The debug images show how one 3D view becomes a 2D AI interpretation, how that
interpretation produces diffusion and DINO feature maps, and how those maps
become the 2048-dimensional descriptors that are later attached back onto the
3D insect.

