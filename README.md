# Cutest picture: kitten and chick

Entry for make.fun's [Make the cutest picture](https://make.fun/bounty/cutest-picture-01) bounty.

## What it does

`main.py` (Python 3.13, standard library only) writes one picture to `/output/entry.png`:
a chubby cream kitten with big glossy blue eyes and a little head tilt, hugging a baby
chick, on a soft pastel background. The same bytes come out on every run.

The picture itself is made by `render3d.py`, an offline raymarcher over a signed-distance
scene: the kitten's head, ears, body, paws and feet are ellipsoids joined with smooth
unions, the chick, scarf and eyes are their own parts, and each pixel is lit with a warm
wrapped key light, soft shadows, ambient occlusion, a sky fill and a rim, at 2x2
supersampling. Face details (iris, highlights, blush, mouth, stripes) are painted in the
head's own frame. Its output is committed as `picture.png`, which `main.py` copies.
`render.py` is the earlier flat 2D version of the same picture.

## How it is scored

A gate checks the file is a whole PNG or JPEG of at most 1 MiB and 1024x1024 pixels.
Valid pictures are then compared two at a time by make.fun's house AI judge, which picks
the cuter one (clear cute subject first, then the subject's cuteness, charm, craft and
how well it reads small). Entries rank by win rate over those games.

## Run it locally

```sh
python main.py            # writes /output/entry.png
pip install numpy pillow  # only needed to redraw the picture
python render3d.py picture.png --bg=blue   # about 2-3 minutes; --bg=pink for the pink version
```
