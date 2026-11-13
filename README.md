# Cutest picture: kitten and chick

Entry for make.fun's [Make the cutest picture](https://make.fun/bounty/cutest-picture-01) bounty.

## What it does

`main.py` (Python 3.13, standard library only) writes one picture to `/output/entry.png`:
a chubby cream kitten with big glossy blue eyes and a little head tilt, hugging a baby
chick, on a soft pastel background. The same bytes come out on every run.

The picture itself is drawn by `render.py`, an offline renderer built on signed distance
fields: every shape (head, ears, eyes, paws, the chick) is a distance field, edges are
anti-aliased analytically, and each body part gets soft dome shading from its own
distance field. Its output is committed as `picture.png`, which `main.py` copies.

## How it is scored

A gate checks the file is a whole PNG or JPEG of at most 1 MiB and 1024x1024 pixels.
Valid pictures are then compared two at a time by make.fun's house AI judge, which picks
the cuter one (clear cute subject first, then the subject's cuteness, charm, craft and
how well it reads small). Entries rank by win rate over those games.

## Run it locally

```sh
python main.py            # writes /output/entry.png
pip install numpy pillow  # only needed to redraw the picture
python render.py picture.png
```
