# Diff3F on BugNIST — Speaker Script

Planned length: **23:50** (20–25 minute target)

The dialogue through Slide 07.1 is Pablo's wording and is kept verbatim. The `//` lines describe what should appear on screen and are not meant to be spoken.

## Slide 01 — Diff3F on BugNIST

**Time:** 00:00–00:55 (0:55)

**Dialogue:**

Hello, my name is Pablo Arce de Aldecoa, I am a master student in Autonomous Systems, and today I am going to talk about Diff3f, walk you through my investigation on this project, and try to awnser the question of whether a descriptor that uses a model trained on ordinary two-dimensional images can give a semantic meaning to three-dimensional scans.

During this presentation I will be explaining a little bit with my words, how this has been done, how it works, the specific dataset I put it into practice, and the experiments and metrics that I carried out to draw some conclusions, where I was surprised, and where I think it fails.

// Here I would put images of the differents bugs, the 4 of them, on grey mesh form, and then
an arrow, and the image of the pca when the descriptor han been drawn.

## Slide 02 — What are we going to answer?

**Time:** 00:55–01:45 (0:50)

**Dialogue:**

So just to be clear on  what we are actually going to respond today:

1. How does the Dif3F pipeline transform an untextured 3D shape into a semantic descriptor
feld, and which implementation choices afect the result?
2. How should the dataset volumes be prepared before the descriptor extraction?
3. Do the descriptors support meaningful grouping and correspondence across diferent
specimens?

// Here I would just put the bullet points of the project in words, no images necessary

## Slide 03 — Can a surface know its anatomy?

**Time:** 01:45–02:35 (0:50)

**Dialogue:**

The whole point of this architecture/method is to achieve an efficient zero-shot semantic segmentation. When we are treating reconstructed surfaces, we normally have information like the coordinates or the normal of each vertex, but nothing that really corresponds anatomically, and with different species, two corresponding points can differ, depending on their size, poses or proportions.

// Here I would just include different images of untreated different grey meshes, like without being rotated, without being smoothed, etc, just a bunch of them. And the sentence Can a surface know its anatomy is fine ig

## Slide 04 — What is Diff3f?

**Time:** 02:35–03:25 (0:50)

**Dialogue:**

In short words, Diff3f is a semantic feature descriptor, that its able to transfer semantic information from pretrained 2D models to untextured 3D geometry, with a very low input level, and in a relatively short time.

So unlike other intrinsic descriptors, we are using Dino Features, and Stable Difussion features, to actually gain information about what those vertexes represent, and not just what the surface around them look like. 

//In this slide, just a big text saying, So what is Diff3f? and the image the owners of the first paper have. Like the actual front of their report image, you know.

## Slide 05 — How does it work?

**Time:** 03:25–04:25 (1:00)

**Dialogue:**

I am going to explain this as an overview, and then we will actually get into detail of each one of the steps.

The central idea is simple, We render the shape from several viewpoints, turn each plain render into a plausible image, generate and extract image features, and project those features back to the visible surface. When we repeat this process fromn many viewpoints, we give every surface point a descriptor from not only one observation, but the average of many.

The result, is a 2048-dimensional sized vector per mesh vertex, or cloud point.

// Use the pipeline overview from the report. Keep only four simple labels below it: render, generate, extract, return to 3D.

## Slide 06 — Stage 1: Rendering the shape

**Time:** 04:25–05:20 (0:55)

**Dialogue:**

The first stage is placing different cameras around the object, and render the geometry from each viewpoint. The cameras, look at the center of the bounding box, from a distance equal to 0.65 x the bounding box. As an obervation, the original implementation places cameras on a grid of azimuth and elevation angles, which sometimes led to repeated views, this was solved by adding other styles of capturing these renderings, like searching for angles which could be also more informative..

// Here include images of different camera views, they exist throught the project report. 

## Slide 06.1 — Normal and dept maps

**Time:** 05:20–06:00 (0:40)

**Dialogue:**

Not only that, but every viewpoint not only produces a plain render, but also a depth map, a normal map, and a foreground mask, which will help ControlNet tell where the visible surface is, and how it is oriented.

//There is a good image in the report, Figure 4, you could use

## Slide 07 — Generating a visible appearance

**Time:** 06:00–06:50 (0:50)

**Dialogue:**

In this stage, Diff3f uses Stable Difussion, an image-to-image model to construct, or "hallucinate" an appearance. The inputs, are simply, the depth and normal maps, which provide geometric constraints, and then a text prompt, which is the only semantic information we are passing to the whole pipeline, which specifies the object category.

// Figure 6 in the report 

## Slide 07.1 — Why does the prompt matter?

**Time:** 06:50–07:55 (1:05)

**Dialogue:**

Over different tests I carried through my experiments, I realised that there is a big variation on the prompt you use, not only on the final generated image, but also on the final generated feature map (which I will explain later). It does not mean that longer prompts are always going to perform better, because the two generations are completly stochastic, but a more detailed one definetely gives a better output. There is although a big barrier here that you have most likely realised by now, Stable difussion is always going to perform worse on things it has not "seen" that much, and even if you give a super accurate description, sometimes it wont make up for that fact.

// Figure 7

## Slide 08 — The recipe I used

**Time:** 07:55–08:45 (0:50)

**Dialogue:**

Before continuing, this table is just the recipe I used for the experiments. I do not expect you to remember every number. The important ones are that I rendered sixteen views at a resolution of 512 by 512, used Stable Diffusion together with the depth and normal maps, and kept one descriptor for every vertex or point. Everything else in the table explains where those final 2048 values come from.

// Here I would put the Table 1 we have on the report, just literally that.

## Slide 08.1 — Where do the features come from?

**Time:** 08:45–09:45 (1:00)

**Dialogue:**

Once the visible image has been generated, Diff3f takes information from two places. The first one is Stable Diffusion itself, while the image is being created. This gives a smooth idea of the general structure of the object. The second one is Dino, which looks at the final image and normally separates parts in a sharper way.

These two maps are then put together. One has 1280 values and the other has 768, which gives the final 2048-dimensional descriptor. Again, the exact number is not the interesting part. What matters is that one source gives more global information, the other gives more local image information, and the final descriptor keeps both.

// Use the three feature images from the report: Diffusion, DINO and the fused Diff3f map. Keep 1280 + 768 = 2048 as the only large text.

## Slide 09 — Returning the information to 3D

**Time:** 09:45–10:35 (0:50)

**Dialogue:**

At this point the features still live on a two-dimensional image, but our original object is three-dimensional. Because we know the camera that produced every render, we can send every visible pixel back to the point or vertex that created it.

We repeat that for all the views and average the observations. So a point is not described from only one image, but from all the cameras that were able to see it. This is the moment where the plain grey mesh becomes the coloured descriptor field that I showed at the beginning.

// Show the two final cricket descriptor fields in 3D, using the shared PCA colours from the report.

## Slide 10 — What is BugNIST?

**Time:** 10:35–11:30 (0:55)

**Dialogue:**

Up to this point I have explained the method. Now I want to explain the data I actually used. BugNIST is a dataset of X-ray CT scans of insects and larvae. Instead of giving us a ready mesh, every specimen is stored as a three-dimensional volume of intensity values.

That is useful because we can see the insect internally and from any direction, but Diff3f cannot work directly on the volume. It needs a surface. So before running any descriptor, I first had to separate the insect from the background and turn the selected voxels into geometry.

// Use the orthogonal CT slices and maximum-intensity projections from the BugNIST section of the report.

## Slide 11 — From a CT scan to a surface

**Time:** 11:30–12:30 (1:00)

**Dialogue:**

The preparation follows a simple chain. First I crop the scan around the specimen. Then I choose a threshold that separates the insect from most of the background. I remove small disconnected pieces, fill gaps, and keep the largest connected object.

After that, marching cubes converts the selected volume into a triangle surface. The first result is recognisable, but it is still very rough because it follows the voxel grid. Finally, I smooth and simplify it, and rotate all the specimens into a comparable orientation. This last step matters because otherwise the same camera could see the side of one insect and the end of another.

// Show the CT threshold overlay, then the raw marching-cubes mesh, then the cleaned mesh, connected by arrows.

## Slide 11.1 — Why simplify the mesh?

**Time:** 12:30–13:40 (1:10)

**Dialogue:**

The original surface had around 216 thousand faces. That is much heavier and much noisier than the clean objects where Diff3f was originally tested. I compared that surface with versions of 75 thousand and 50 thousand faces.

The 50 thousand face version still kept the head, the body and the legs, but reduced the number of descriptor rows by around 77 percent. In the recorded test it also went from 134.5 seconds to 79.8 seconds for one view. I would not say that the timing is perfectly linear, because the 75 thousand version happened to be faster in that run. The point is simply that a much lighter mesh kept the large anatomical structure and made the rest of the experiments more practical.

// Use the smoothing comparison from the report. Next to it show only: 216k to 50k faces, 418 to 94 MiB, and 134.5 to 79.8 seconds.

## Slide 12 — What did I test?

**Time:** 13:40–14:30 (0:50)

**Dialogue:**

For the final experiments I used four specimens: a brown cricket, a black cricket, a mealworm, and a soldier fly larva.

I looked at the descriptors in three different ways. First, I tested if a point could travel to another specimen and come back to the same area. Second, I checked if the descriptors grouped similar body regions without using labels. And third, I placed anatomical landmarks manually and tested if the best match actually reached the point I expected.

// Show the four grey specimens. Under them use three short questions: Does it come back? Does it group body parts? Does it find the point?

## Slide 13 — First test: does the match come back?

**Time:** 14:30–15:50 (1:20)

**Dialogue:**

For this first test I did not use any manual labels. I selected 300 points from one specimen, matched each descriptor to the most similar descriptor on the other specimen, and then matched it back again. If it comes back close to the original point, the match is at least internally consistent.

The clearest result is the cricket pair. Around 70 percent of the points came back within ten percent of the body size in both directions. The larvae still shared some structure, but the result depended much more on the direction. Comparisons between an adult cricket and a larva were normally weaker.

This already suggested that the descriptor works better when the two shapes have a similar body plan. It also showed me that a high similarity number alone is not enough. A match can look similar in feature space and still be wrong on the geometry.

// Use the cycle-consistency heatmap, but keep the cricket pair visually highlighted.

## Slide 14 — Second test: does it discover body parts?

**Time:** 15:50–17:05 (1:15)

**Dialogue:**

For the second test I put the descriptors from the two crickets into the same visual space. That is what the smooth PCA colours show. Then I used one shared k-means model to divide both insects into groups.

The model did not know where the vertices were, which insect they came from, or the name of any body part. It only saw the descriptors. Even with that limitation, the same colours appeared in many of the same regions on both insects.

When I checked the nine manual landmarks, eight of them received the same cluster in both crickets. That is a good result for coarse anatomy. But it also exposed the main problem: left and right legs often receive the same kind of description. The method can recognise a leg-like region without always knowing which leg it is.

// Use the shared PCA and K=6/K=10 comparison. Put 8 out of 9 in one corner, without another separate slide.

## Slide 15 — Third test: can it find the exact point?

**Time:** 17:05–18:25 (1:20)

**Dialogue:**

The last test is more direct. I manually placed nine landmarks on each cricket in Blender: the head, the centre of the thorax, the end of the abdomen, the antenna bases, and four leg bases. For the larvae I used the three points that I could identify consistently: head, centre and abdomen.

For every source landmark, I took its descriptor and searched the complete target for the most similar one. Then I measured the distance between the prediction and the manual target. I divided that distance by the size of the target specimen, so the errors remain comparable even when the meshes have different scales.

This is the strictest experiment because a smooth colour map is not enough anymore. The prediction has to land close to the anatomical point I selected by hand.

// Use a very simple flow: manual source point, descriptor match, predicted point, error to the manual target. Avoid formulas.

## Slide 16 — What happened on the crickets?

**Time:** 18:25–19:40 (1:15)

**Dialogue:**

The cricket result was better than I expected on the central parts of the body. In both directions, seven of the nine landmarks were within ten percent of the target size. The antenna bases, the head, the thorax and the abdomen stayed below three percent.

The errors were not spread equally over the whole insect. They were concentrated on the legs. Some leg matches were still good, but the largest mistakes jumped to another leg or another side of the body. This is why the mean error looks much worse than the median: most of the central points are close, and a few leg failures are very far away.

So I would not describe the result as a general failure. The descriptor does understand a lot of the body. The difficulty is choosing between repeated structures that look and mean almost the same thing.

// Use the landmark error chart. Keep the five values below 3 percent in one colour and the large leg errors in another.

## Slide 16.1 — It knows “leg”, but which leg?

**Time:** 19:40–20:55 (1:15)

**Dialogue:**

These two examples make the problem very clear. They both go from the black cricket to the brown cricket. On the left, the right foreleg lands almost exactly where it should, with an error of 2.02 percent. On the right, the left foreleg lands in a different region and the error becomes 27.94 percent.

The descriptor is not choosing a completely random place. It is often choosing something that is still leg-like. But the matching rule looks at every point independently. It has no rule saying that a right leg should remain on the right, or that neighbouring source points should stay together on the target.

This was probably the most useful failure in the project, because it shows exactly what information is there and what information is still missing.

// Put the successful and failed leg correspondence images side by side, with only 2.02% and 27.94% as text.

## Slide 17 — The larvae gave a warning

**Time:** 20:55–22:10 (1:15)

**Dialogue:**

The larvae gave a different and more serious warning. If we only look at their PCA colours, both descriptor fields seem smooth and organised from one end of the body to the other. It is very tempting to interpret that as a correct correspondence.

But when I tested the manual points, only the head transferred well in both directions. The centre failed, and when I matched the abdomen from the soldier fly larva to the mealworm, the prediction went to the opposite end. The error was 86.49 percent of the target size.

So a descriptor can look organised without understanding the direction of the anatomy. This is also where the limitation of Stable Diffusion becomes important. A general image model has seen many common animals, but probably much less useful information about the two ends of specialised larvae.

// Show the two smooth larval PCA fields next to the wrong-end correspondence, with 86.49% as the only large number.

## Slide 18 — What did I learn?

**Time:** 22:10–23:10 (1:00)

**Dialogue:**

My main conclusion is that Diff3f does transfer useful semantic information to these CT-derived insects without training a new 3D model and without giving it anatomical labels.

It works especially well for broad and distinctive regions. The head, the central body and the abdomen can be very consistent between the two crickets. The clustering also shows that the descriptors contain a real idea of body-part type.

But part identity is not the same as point identity. Knowing that something is a leg does not automatically tell us which leg, and a smooth body gradient does not automatically tell us which end is the head. The result also depends on the quality of the mesh, the camera views, the prompt, and how familiar the image model is with the object.

// Use one simple statement on screen: Part identity is not point identity. Put the two coloured crickets behind it.

## Slide 19 — What would I do next?

**Time:** 23:10–23:50 (0:40)

**Dialogue:**

The next step would be to label more specimens, try an image model that knows more about insects, and add some form of side or global consistency so repeated legs are not treated independently.

But for this project, the answer is yes: an untextured CT insect can gain useful semantics from pretrained 2D models, as long as we are honest about the difference between recognising a body part and finding the exact point.

Thank you, and I am happy to take questions.

// Keep three short ideas: more ground truth, an insect-aware image model, and side-aware matching. Finish with Questions?

