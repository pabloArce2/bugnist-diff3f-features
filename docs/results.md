# What the experiments showed

Four specimens were processed as described in [pipeline.md](pipeline.md): a brown and a
black cricket, a mealworm (MEL) and a soldier fly larva (SL). The full analysis is in the
project report; this is the short version.

## The descriptors organise the body

With a shared PCA, the two crickets get the same colour pattern over head, thorax, abdomen
and appendages, and the two larvae both vary smoothly along the body axis. With a shared
k-means (K=10, fitted on the full descriptors of both crickets), the nearest vertex to eight
of the nine hand-placed cricket landmarks falls in the same cluster on both specimens. What
is not captured is the side: left and right copies of a leg or antenna usually share a
cluster.

## Matching works for similar specimens

Label-free diagnostics, 300 farthest-point samples per direction:

| Direction | median cosine | cycle within 10 % | unique targets |
| --- | --- | --- | --- |
| brown cricket -> black cricket | 0.810 | 70.0 % | 0.893 |
| black cricket -> brown cricket | 0.817 | 72.0 % | 0.917 |
| MEL -> SL | 0.745 | 62.7 % | 0.647 |
| brown cricket -> SL | 0.584 | 34.3 % | 0.667 |

The cricket pair is the strongest and most symmetric; cricket-larva pairs are much weaker,
with many source points collapsing onto a few targets.

## Landmarks: parts yes, sides no

| Direction | PCK@2 % | PCK@5 % | PCK@10 % | median error |
| --- | --- | --- | --- | --- |
| brown -> black cricket | 3/9 | 5/9 | 7/9 | 2.55 % |
| black -> brown cricket | 3/9 | 6/9 | 7/9 | 2.39 % |
| MEL -> SL | 0/3 | 2/3 | 2/3 | 3.84 % |
| SL -> MEL | 0/3 | 1/3 | 1/3 | 23.28 % |

Errors are a percentage of the target's bounding-box diagonal. On the crickets the head,
thorax, abdomen tip and antenna bases all land within 3 %; the failures are legs matched to
the wrong leg (up to 39 %). On the larvae the head transfers well, but SL -> MEL sends the
abdomen tip to the wrong end of the body.

## In short

Diff3F transfers part identity (this is a leg, this is the head) to CT-derived insect
meshes with nothing but a text prompt, but it does not reliably tell repeated or
mirror-image parts apart, and it depends on how well Stable Diffusion knows the kind of
animal: it did clearly better on crickets than on larvae. Clean, simplified and consistently
oriented meshes were a precondition for any of this to work.

The benchmark is small (one cricket pair, one larva pair, 12 landmark pairs), so these
numbers describe these specimens, not the method in general.
