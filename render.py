"""Offline renderer for the entry picture (dev tool: numpy + Pillow).

Draws a chubby kitten hugging a baby chick with signed distance fields,
analytic anti-aliasing and soft dome shading, and writes picture.png.
main.py (stdlib only) copies that file to /output/entry.png.
"""
import math
import sys

import numpy as np
from PIL import Image

S = 1024
BUBBLE = "--bubble" in sys.argv
PY, PX = np.mgrid[0:S, 0:S].astype(np.float64) + 0.5
# with the speech bubble the scene shrinks about its bottom centre to free the top
SCALE = 0.86 if BUBBLE else 1.0
XS = 512 + (PX - 512) / SCALE
YS = 1010 + (PY - 1010) / SCALE


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


# ---------------------------------------------------------------- SDF helpers
def ellipse(cx, cy, rx, ry, X=XS, Y=YS):
    px, py = (X - cx) / rx, (Y - cy) / ry
    k0 = np.sqrt(px * px + py * py)
    k1 = np.sqrt((px / rx) ** 2 + (py / ry) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def circle(cx, cy, r, X=XS, Y=YS):
    return np.hypot(X - cx, Y - cy) - r


def segment(ax, ay, bx, by, r, X=XS, Y=YS):
    pax, pay = X - ax, Y - ay
    bax, bay = bx - ax, by - ay
    h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay), 0, 1)
    return np.hypot(pax - bax * h, pay - bay * h) - r


def arc(cx, cy, rad, a0, a1, th, X=XS, Y=YS):
    """Stroke of a circular arc from angle a0 to a1 (radians, y down)."""
    ang = np.arctan2(Y - cy, X - cx)
    mid = (a0 + a1) / 2
    half = (a1 - a0) / 2
    da = np.abs((ang - mid + np.pi) % (2 * np.pi) - np.pi)
    d_ring = np.abs(np.hypot(X - cx, Y - cy) - rad) - th
    e0 = (cx + rad * math.cos(a0), cy + rad * math.sin(a0))
    e1 = (cx + rad * math.cos(a1), cy + rad * math.sin(a1))
    d_end = np.minimum(np.hypot(X - e0[0], Y - e0[1]), np.hypot(X - e1[0], Y - e1[1])) - th
    return np.where(da <= half, d_ring, d_end)


def triangle(p0, p1, p2, X=XS, Y=YS):
    pts = [p0, p1, p2]
    d = np.full(X.shape, 1e9)
    cs = []
    for i in range(3):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 3]
        ex, ey = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1)
        d = np.minimum(d, np.hypot(wx - ex * h, wy - ey * h))
        cs.append(ex * wy - ey * wx)
    inside = ((cs[0] > 0) & (cs[1] > 0) & (cs[2] > 0)) | ((cs[0] < 0) & (cs[1] < 0) & (cs[2] < 0))
    return np.where(inside, -d, d)


def heart(cx, cy, size, X=XS, Y=YS):
    """Heart distance (approximate), point down, centred near (cx, cy)."""
    px = np.abs(X - cx) / size
    py = (cy - Y) / size + 0.55  # y up, origin at the bottom tip
    a = np.hypot(px - 0.25, py - 0.75) - math.sqrt(2) / 4
    m = 0.5 * np.maximum(px + py, 0)
    b = np.sqrt(np.minimum((px - 1) ** 2 + (py - 1) ** 2, (px - m) ** 2 + (py - m) ** 2)) * np.sign(px - py)
    return np.where(py + px > 1, a, b) * size


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def rot(cx, cy, deg):
    """Coordinates rotated about (cx, cy): shapes drawn in them appear rotated by deg."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    dx, dy = XS - cx, YS - cy
    return cx + c * dx + s * dy, cy - s * dx + c * dy


def cov(d, soft=1.0):
    return np.clip(0.5 - d / soft, 0, 1)


def blur(a, r):
    """Approximate Gaussian blur: three separable box blurs of radius r."""
    r = max(1, int(r))
    for _ in range(3):
        for axis in (0, 1):
            p = np.pad(a, [(r + 1, r)] if False else ((r + 1, r), (0, 0)) if axis == 0 else ((0, 0), (r + 1, r)), mode="edge")
            c = np.cumsum(p, axis=axis)
            if axis == 0:
                a = (c[2 * r + 1:, :] - c[:-2 * r - 1, :]) / (2 * r + 1)
            else:
                a = (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)
    return a


# ---------------------------------------------------------------- painting
img = np.zeros((S, S, 3))


def paint(color, alpha):
    global img
    col = color if np.ndim(color) == 3 else np.broadcast_to(color, img.shape)
    a = alpha[..., None]
    img = img * (1 - a) + col * a


LIGHT = np.array([-0.42, -0.62, 0.66])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def shaded(d, base, shadow, depth=110.0, light=None, gloss=0.22):
    """Soft dome shading: height field from the distance field, blurred, lit from the top left."""
    t = np.clip(-d / depth, 0, 1)
    h = np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1)) * depth
    h = blur(h, depth * 0.12)
    gy, gx = np.gradient(h)
    n = np.sqrt(gx * gx + gy * gy + 1)
    nx, ny, nz = -gx / n, -gy / n, 1 / n
    lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    k = np.clip((lam - 0.15) / 0.8, 0, 1)[..., None] ** 0.8
    col = shadow[None, None, :] * (1 - k) + base[None, None, :] * k
    if light is not None:
        top = np.clip((lam - 0.85) / 0.15, 0, 1)[..., None]
        col = col * (1 - top) + light[None, None, :] * top
    spec = np.clip(lam, 0, 1) ** 40 * gloss
    return np.clip(col + spec[..., None], 0, 1)


def outlined(d, base, shadow, line, width=5.0, **kw):
    paint(line, cov(d - width))
    paint(shaded(d, base, shadow, **kw), cov(d))


# ---------------------------------------------------------------- the scene
def draw():
    # background: soft pastel radial gradient, lighter behind the subject
    r = np.hypot(XS - 512, YS - 460) / 760
    c_in, c_out = hexc("#FFF3F6"), hexc("#F8C3D6")
    t = np.clip(r, 0, 1)[..., None] ** 1.2
    paint(c_in * (1 - t) + c_out * t, np.ones((S, S)))

    # soft bokeh circles
    for (bx_, by_, br, ba) in [(170, 330, 90, 0.18), (860, 470, 110, 0.15), (230, 880, 70, 0.14),
                               (800, 850, 80, 0.12), (700, 90, 60, 0.16)]:
        paint(hexc("#FFFFFF"), np.clip(0.5 - circle(bx_, by_, br) / 18, 0, 1) * ba)

    # a few sparkles and floating hearts
    for (sx, sy, sr) in [(150, 190, 30), (890, 640, 26), (120, 700, 18)]:
        dd = np.minimum(ellipse(sx, sy, sr, sr * 0.22), ellipse(sx, sy, sr * 0.22, sr))
        paint(hexc("#FFFFFF"), cov(dd, 1.5) * 0.9)
    for (hx_, hy_, hs, col) in [(870, 170, 44, "#FF7FA3"), (930, 300, 26, "#FF9DB8"), (95, 420, 30, "#FF9DB8")]:
        hd = heart(hx_, hy_, hs)
        paint(hexc("#FFFFFF"), cov(hd - 4) * 0.9)
        paint(shaded(hd, hexc(col), hexc("#E0507A"), depth=hs * 0.6, gloss=0.5), cov(hd))

    # floor shadow
    paint(hexc("#D88AA6"), np.clip(0.5 - ellipse(512, 980, 350, 46) / 40, 0, 1) * 0.35)

    cream = hexc("#FFF9F0")
    cream_sh = hexc("#EBCDB9")
    white = hexc("#FFFFFF")
    line = hexc("#6E4A48")
    pink = hexc("#FF93AE")

    HX, HY, TILT = 512, 440, 7.0
    hx, hy = rot(HX, HY, TILT)

    # kitten: head + body + ears as one soft blob
    body = ellipse(512, 800, 270, 200)
    head = ellipse(HX, HY, 320, 262, hx, hy)
    earL = triangle((HX - 300, HY - 90), (HX - 238, HY - 345), (HX - 92, HY - 222), hx, hy) - 26
    earR = triangle((HX + 300, HY - 90), (HX + 238, HY - 345), (HX + 92, HY - 222), hx, hy) - 26
    kit = smin(smin(head, earL, 34), earR, 34)
    kit = smin(kit, body, 110)
    outlined(kit, cream, cream_sh, line, width=6, depth=150, light=white)

    # inner ears
    for sgn in (-1, 1):
        ie = triangle((HX + sgn * 250, HY - 140), (HX + sgn * 230, HY - 285), (HX + sgn * 148, HY - 215), hx, hy) - 12
        paint(shaded(ie, hexc("#FFC2D0"), hexc("#F29AB2"), depth=40, gloss=0.0), cov(ie))

    # ginger forehead stripes
    for (ax, ay, bx, by) in [(0, -245, 0, -188), (-50, -238, -40, -192), (50, -238, 40, -192)]:
        st = segment(HX + ax, HY + ay, HX + bx, HY + by, 12, hx, hy)
        paint(hexc("#F7B06A"), cov(st) * 0.9)

    # mint scarf around the neck
    mint, mint_sh, mint_ln = hexc("#A6EBD5"), hexc("#62C2A6"), hexc("#3F7F6E")
    scarf = np.maximum(ellipse(512, 640, 252, 74), -ellipse(512, 600, 262, 64))
    outlined(scarf, mint, mint_sh, mint_ln, width=5, depth=34, gloss=0.1)
    tail = smin(circle(372, 690, 30), ellipse(342, 758, 36, 62), 18)
    outlined(tail, mint, mint_sh, mint_ln, width=5, depth=30, gloss=0.1)
    for k in (-1, 0, 1):
        paint(mint_ln, cov(segment(342 + k * 16, 796, 342 + k * 16, 816, 2.6)) * 0.85)

    # chick held in front
    CX, CY = 512, 800
    chick = circle(CX, CY, 118)
    # tuft of three feathers behind the outline
    for (ang, ln) in [(-100, 52), (-78, 44), (-122, 40)]:
        a = math.radians(ang)
        fx, fy = CX + 6 + ln * math.cos(a), CY - 112 + ln * math.sin(a)
        f = segment(CX + 4, CY - 108, fx, fy, 11)
        outlined(f, hexc("#FFE680"), hexc("#F5B83A"), hexc("#9A6A2A"), width=4, depth=10, gloss=0.0)
    outlined(chick, hexc("#FFE36E"), hexc("#F2A93A"), hexc("#9A6A2A"), width=5, depth=100,
             light=hexc("#FFF6C8"), gloss=0.25)
    # chick wings
    for sgn in (-1, 1):
        w = ellipse(CX + sgn * 96, CY + 30, 28, 44)
        paint(hexc("#F7C447"), cov(w) * cov(chick))
    # chick eyes: big and sparkly
    for sgn in (-1, 1):
        ex, ey = CX + sgn * 44, CY - 20
        paint(hexc("#2B1D1D"), cov(ellipse(ex, ey, 19, 23)))
        paint(white, cov(circle(ex - 6, ey - 8, 7.5)))
        paint(white, cov(circle(ex + 6, ey + 8, 3.5)))
    # chick beak
    beak = smin(triangle((CX - 18, CY + 2), (CX + 18, CY + 2), (CX, CY + 24)) - 4, circle(CX, CY + 5, 9), 4)
    paint(hexc("#C2581F"), cov(beak - 2.5))
    paint(hexc("#FF8A3D"), cov(beak))
    # chick blush
    for sgn in (-1, 1):
        paint(hexc("#FF8FA3"), np.clip(0.5 - ellipse(CX + sgn * 74, CY + 14, 20, 12) / 6, 0, 1) * 0.65)

    # kitten paws hugging the chick
    for sgn in (-1, 1):
        px_, py_ = CX + sgn * 112, CY + 46
        pw = ellipse(px_, py_, 50, 56)
        outlined(pw, cream, cream_sh, line, width=5, depth=46, light=white)
        for k in (-1, 0, 1):
            tx, ty = px_ - sgn * 26, py_ + k * 18
            paint(line, cov(segment(tx - sgn * 12, ty, tx + sgn * 4, ty, 2.6)) * 0.85)

    # kitten feet
    for sgn in (-1, 1):
        fx_ = 512 + sgn * 132
        ft = ellipse(fx_, 966, 76, 40)
        outlined(ft, cream, cream_sh, line, width=5, depth=36, light=white)
        for k in (-1, 1):
            paint(line, cov(segment(fx_ + k * 22, 940, fx_ + k * 22, 955, 2.6)) * 0.85)

    # eyes: huge and glossy
    for sgn in (-1, 1):
        ex, ey = HX + sgn * 132, HY + 44
        eye = ellipse(ex, ey, 78, 90, hx, hy)
        paint(hexc("#2A1E2E"), cov(eye - 3))
        tt = np.clip((hy - (ey - 90)) / 180, 0, 1)[..., None]
        iris_col = hexc("#1C2C6C") * (1 - tt) + hexc("#5CC8FF") * tt
        iris = ellipse(ex, ey + 6, 66, 76, hx, hy)
        paint(iris_col, cov(iris))
        pupil = ellipse(ex, ey + 4, 38, 46, hx, hy)
        paint(hexc("#121426"), cov(pupil))
        paint(hexc("#B4F0FF"), np.clip(0.5 - ellipse(ex, ey + 56, 36, 14, hx, hy) / 10, 0, 1) * 0.6)
        paint(white, cov(ellipse(ex - 27, ey - 32, 27, 31, hx, hy)))
        paint(white, cov(circle(ex + 28, ey + 32, 13, hx, hy)))
        paint(white, cov(circle(ex + 34, ey - 40, 6, hx, hy)) * 0.9)

    # blush
    for sgn in (-1, 1):
        bl = ellipse(HX + sgn * 222, HY + 132, 60, 34, hx, hy)
        paint(hexc("#FF8FAB"), np.clip(0.5 - bl / 16, 0, 1) * 0.6)

    # nose and a small open smile
    nose = smin(triangle((HX - 19, HY + 112), (HX + 19, HY + 112), (HX, HY + 132), hx, hy) - 5,
                circle(HX, HY + 116, 10, hx, hy), 5)
    paint(pink, cov(nose))
    paint(line, cov(segment(HX, HY + 132, HX, HY + 144, 3.2, hx, hy)))
    # small open mouth with a tongue under the w
    mouth = np.maximum(ellipse(HX, HY + 150, 21, 26, hx, hy), (HY + 150) - hy)
    paint(line, cov(mouth - 3))
    paint(hexc("#A8445A"), cov(mouth))
    paint(hexc("#FF8FA6"), cov(np.maximum(circle(HX, HY + 178, 15, hx, hy), mouth)))
    for sgn in (-1, 1):
        paint(line, cov(arc(HX + sgn * 18, HY + 144, 18, 0.0, math.pi, 3.4, hx, hy)))

    # whiskers
    for sgn in (-1, 1):
        for k in (-1, 0, 1):
            ax, ay = HX + sgn * 268, HY + 108 + k * 22
            bx, by = HX + sgn * 335, HY + 95 + k * 32
            paint(line, cov(segment(ax, ay, bx, by, 2.4, hx, hy)) * 0.75)

    # tiny flower by the right ear
    FX, FY = HX + 212, HY - 212
    for i in range(5):
        a = i * 2 * math.pi / 5 - math.pi / 2
        pd = circle(FX + 24 * math.cos(a), FY + 24 * math.sin(a), 20, hx, hy)
        paint(hexc("#C76A86"), cov(pd - 3))
        paint(hexc("#FFE3EC"), cov(pd))
    paint(hexc("#FFC93C"), cov(circle(FX, FY, 13, hx, hy)))


def draw_bubble():
    """A speech bubble from the kitten, in pixel coordinates."""
    from PIL import ImageDraw, ImageFont
    x0, y0, x1, y1 = 52, 26, 600, 168
    box = np.maximum(np.abs(PX - (x0 + x1) / 2) - ((x1 - x0) / 2 - 40), np.abs(PY - (y0 + y1) / 2) - ((y1 - y0) / 2 - 40))
    rr = np.hypot(np.maximum(np.abs(PX - (x0 + x1) / 2) - ((x1 - x0) / 2 - 40), 0),
                  np.maximum(np.abs(PY - (y0 + y1) / 2) - ((y1 - y0) / 2 - 40), 0))
    rbox = np.where(box > 0, rr, box) - 40
    tail = triangle((390, 150), (470, 150), (452, 215), PX, PY) - 4
    bub = smin(rbox, tail, 8)
    paint(hexc("#6E4A48"), cov(bub - 5))
    paint(hexc("#FFFFFF"), cov(bub))
    layer = Image.new("L", (S, S), 0)
    dr = ImageDraw.Draw(layer)
    f1 = ImageFont.truetype("C:/Windows/Fonts/seguibl.ttf", 46)
    f2 = ImageFont.truetype("C:/Windows/Fonts/seguibl.ttf", 40)
    t1, t2 = "We're best friends!", "Pick us?"
    w1 = dr.textlength(t1, font=f1)
    w2 = dr.textlength(t2, font=f2)
    cx = (x0 + x1) / 2
    dr.text((cx - w1 / 2, 40), t1, font=f1, fill=255)
    dr.text((cx - (w2 + 46) / 2, 100), t2, font=f2, fill=255)
    a = np.asarray(layer, dtype=np.float64) / 255
    paint(hexc("#5A3A44"), a)
    hd = heart(cx + (w2 + 46) / 2 - 18, 126, 34, PX, PY)
    paint(hexc("#FF6F98"), cov(hd))


draw()
if BUBBLE:
    draw_bubble()
out = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0] if args else "picture.png"
out.save(path, optimize=True)
print(path)
