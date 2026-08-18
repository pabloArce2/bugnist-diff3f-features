# Otsu Thresholding Explained

Otsu thresholding is an automatic way to choose an intensity threshold.

In our BugNIST CT pipeline, thresholding means:

```text
voxel intensity > threshold  -> insect/object
voxel intensity <= threshold -> background/noise
```

Until now, many commands used manual thresholds like:

```text
bcrick threshold 29
sfaar threshold 11
soldat threshold 28
```

Otsu tries to choose this number automatically from the intensity histogram.

## The Idea

Imagine the CT volume has two main intensity groups:

```text
dark voxels   -> air/background/cotton/noise
bright voxels -> insect material
```

The histogram might look roughly like this:

```text
voxel count
   ^
   |
   |     background
   |       peak
   |      /\
   |     /  \                 insect
   |    /    \                 peak
   |___/      \____      _____/\
   |               \____/       \____
   +------------------------------------> intensity
                    ^
                    |
              Otsu threshold
```

Otsu searches for the threshold that best separates the histogram into two
classes. It tries to make each side internally compact and the two sides as
different as possible.

In simpler words:

```text
Otsu asks:

"Where is the best cut between dark stuff and bright stuff?"
```

## Why It Is Useful

Otsu is useful because it gives us a reproducible automatic threshold:

```text
same input volume + same ROI -> same threshold
```

It avoids manually guessing values like `11`, `28`, or `29`.

It is especially useful for first-pass segmentation and for comparing whether
manual thresholds were reasonable.

## Why It Can Fail

Otsu assumes the histogram can be separated into roughly two meaningful groups.

BugNIST CT scans are messy. A crop may contain:

```text
insect
cotton
air
scan artifacts
multiple materials
partial body structures
```

If the ROI contains lots of cotton or background, Otsu may choose a threshold
that separates background from cotton instead of cotton from insect.

So Otsu is not magic. It is a good automatic baseline, not guaranteed truth.

## Otsu And ROI

Otsu is computed on the volume you give it.

That means the ROI matters a lot.

```text
large messy ROI -> Otsu sees many irrelevant voxels
clean insect ROI -> Otsu sees a clearer histogram
```

For this project:

```text
manual ROI + Otsu threshold
```

is usually more defensible than:

```text
whole CT volume + Otsu threshold
```

because the ROI removes unrelated material before the histogram is analyzed.

## How To Use It

For previews:

```powershell
python scripts\preview_tif_volume.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --outdir previews\bugnist_raw_ct_otsu `
  --roi-start 185 30 29 `
  --roi-size 180 220 200 `
  --threshold-method otsu
```

For mesh conversion:

```powershell
python scripts\bugnist_tif_to_mesh.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --out meshes\bugnist_individual\bcrick_10_001\otsu\bcrick_10_001_otsu_roi_keeplargest_ds1.obj `
  --roi-start 185 30 29 `
  --roi-size 180 220 200 `
  --threshold-method otsu `
  --keep-largest `
  --downsample 1
```

For point-cloud conversion:

```powershell
python scripts\bugnist_tif_to_pointcloud.py `
  --tif bugNIST\raw\bcrick_10_001.tif `
  --out pointclouds\bugnist_individual\bcrick_10_001\otsu\bcrick_10_001_otsu_roi_keeplargest_ds1_20k.ply `
  --npy pointclouds\bugnist_individual\bcrick_10_001\otsu\bcrick_10_001_otsu_roi_keeplargest_ds1_20k.npy `
  --roi-start 185 30 29 `
  --roi-size 180 220 200 `
  --threshold-method otsu `
  --keep-largest `
  --downsample 1 `
  --num-points 20000 `
  --method mesh-surface `
  --center
```

## How To Compare Otsu Against Manual Thresholds

Generate an Otsu mesh and a manual-threshold mesh with the same ROI.

Then preview both:

```powershell
python scripts\preview_geometry_points.py `
  --kind mesh `
  --input meshes\bugnist_individual\bcrick_10_001\otsu\bcrick_10_001_otsu_roi_keeplargest_ds1.obj `
  --out previews\bugnist_otsu_compare\bcrick_otsu.png
```

Compare against:

```powershell
python scripts\preview_geometry_points.py `
  --kind mesh `
  --input meshes\bugnist_individual\bcrick_10_001\preprocessed\bcrick_10_001_thr29_roi_keeplargest_ds1.obj `
  --out previews\bugnist_otsu_compare\bcrick_manual_thr29.png
```

Things to check:

```text
Does Otsu keep the full insect?
Does it remove cotton/background?
Are thin legs or antennae lost?
Does it create too much surface fuzz?
```

## What The Scripts Now Do

The conversion scripts support:

```text
--threshold-method manual      use --threshold
--threshold-method percentile  use --threshold-percentile
--threshold-method otsu        compute Otsu automatically
--threshold-method auto        use manual if supplied, percentile if supplied, else Otsu
```

`auto` is the default for mesh and point-cloud conversion.

The scripts now print the threshold used:

```text
Segmentation threshold: 18.000 (otsu)
```

That printed number is important. Save it in notes if you want the experiment to
be explainable later.

## The One-Sentence Explanation

Otsu thresholding automatically chooses the intensity cut that best separates
the CT crop into dark and bright voxel groups, giving us a reproducible first
guess for insect segmentation without manually choosing the threshold.

