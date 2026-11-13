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
YS, XS = np.mgrid[0:S, 0:S].astype(np.float64) + 0.5


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
    on = da <= half
    d_ring = np.abs(np.hypot(X - cx, Y - cy) - rad) - th
    e0 = (cx + rad * math.cos(a0), cy + rad * math.sin(a0))
    e1 = (cx + rad * math.cos(a1), cy + rad * math.sin(a1))
    d_end = np.minimum(np.hypot(X - e0[0], Y - e0[1]), np.hypot(X - e1[0], Y - e1[1])) - th
    return np.where(on, d_ring, d_end)


def triangle(p0, p1, p2, X=XS, Y=YS):
    pts = [p0, p1, p2]
    d = np.full(X.shape, 1e9)
    s = np.ones(X.shape)
    for i in range(3):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 3]
        ex, ey = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1)
        d = np.minimum(d, np.hypot(wx - ex * h, wy - ey * h))
        c = ex * wy - ey * wx
        s = s * np.where(c > 0, 1, -1)
    # inside when all cross products share a sign
    sgn = np.ones(X.shape)
    c0 = (pts[1][0] - pts[0][0]) * (Y - pts[0][1]) - (pts[1][1] - pts[0][1]) * (X - pts[0][0])
    c1 = (pts[2][0] - pts[1][0]) * (Y - pts[1][1]) - (pts[2][1] - pts[1][1]) * (X - pts[1][0])
    c2 = (pts[0][0] - pts[2][0]) * (Y - pts[2][1]) - (pts[0][1] - pts[2][1]) * (X - pts[2][0])
    inside = ((c0 > 0) & (c1 > 0) & (c2 > 0)) | ((c0 < 0) & (c1 < 0) & (c2 < 0))
    return np.where(inside, -d, d)


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


# ---------------------------------------------------------------- painting
img = np.zeros((S, S, 3))


def paint(color, alpha):
    global img
    col = color if np.ndim(color) == 3 else np.broadcast_to(color, img.shape)
    a = alpha[..., None]
    img = img * (1 - a) + col * a


LIGHT = np.array([-0.45, -0.65, 0.62])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def shaded(d, base, depth=110.0, amb=0.62, spec=0.18, warm=None):
    """Soft dome shading of a filled shape from its distance field."""
    gy, gx = np.gradient(d)
    g = np.hypot(gx, gy) + 1e-9
    gx, gy = gx / g, gy / g
    t = np.clip(-d / depth, 0, 1)
    edge = np.sqrt(np.clip(1 - t, 0, 1)) * (1 - t) ** 0.5
    nx, ny = gx * edge, gy * edge
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    diff = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    shade = amb + (1 - amb) * diff
    col = base[None, None, :] * shade[..., None]
    if warm is not None:  # warm bounce light on the shadow side
        k = np.clip(1 - diff, 0, 1)[..., None] * 0.18
        col = col * (1 - k) + warm[None, None, :] * k
    h = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1) ** 24 * spec
    col = col + h[..., None]
    return np.clip(col, 0, 1)


def outlined(d, base, line, width=5.0, **kw):
    paint(line, cov(d - width))
    paint(shaded(d, base, **kw), cov(d))


# ---------------------------------------------------------------- the scene
def draw():
    # background: soft pastel radial gradient with a light glow behind the subject
    r = np.hypot(XS - 512, YS - 470) / 720
    c_in, c_out = hexc("#FFF1F4"), hexc("#F9C9DA")
    t = np.clip(r, 0, 1)[..., None] ** 1.3
    paint(c_in * (1 - t) + c_out * t, np.ones((S, S)))

    # sparkles in the background
    for (sx, sy, sr) in [(140, 170, 34), (880, 210, 26), (110, 640, 20), (915, 600, 30), (820, 110, 14)]:
        dd = np.minimum(ellipse(sx, sy, sr, sr * 0.22), ellipse(sx, sy, sr * 0.22, sr))
        paint(hexc("#FFFFFF"), cov(dd, 1.5) * 0.85)

    # floor shadow
    paint(hexc("#D98AA8"), np.clip(0.5 - ellipse(512, 975, 360, 48) / 40, 0, 1) * 0.35)

    cream = hexc("#FFF6EA")
    line = hexc("#6B4A4A")
    warm = hexc("#FFC9A8")
    ginger = hexc("#F6B26B")
    pink = hexc("#FF9EB5")

    HX, HY, TILT = 512, 430, 7.0
    hx, hy = rot(HX, HY, TILT)

    # kitten: head + body + ears as one soft blob
    body = ellipse(512, 800, 255, 205)
    head = ellipse(HX, HY, 305, 250, hx, hy)
    earL = triangle((HX - 285, HY - 95), (HX - 225, HY - 330), (HX - 85, HY - 215), hx, hy) - 26
    earR = triangle((HX + 285, HY - 95), (HX + 225, HY - 330), (HX + 85, HY - 215), hx, hy) - 26
    kit = smin(smin(head, earL, 30), earR, 30)
    kit = smin(kit, body, 70)
    outlined(kit, cream, line, width=6, warm=warm, depth=140)

    # inner ears
    for sgn in (-1, 1):
        ie = triangle((HX + sgn * 250, HY - 120), (HX + sgn * 222, HY - 290), (HX + sgn * 125, HY - 205), hx, hy) - 14
        paint(shaded(ie, hexc("#FFB8C8"), depth=40, amb=0.8), cov(ie))

    # ginger forehead stripes
    for (ax, ay, bx, by) in [(0, -235, 0, -175), (-48, -228, -38, -180), (48, -228, 38, -180)]:
        st = segment(HX + ax, HY + ay, HX + bx, HY + by, 12, hx, hy)
        paint(ginger, cov(st) * 0.9)

    # chick held in front
    CX, CY = 512, 790
    chick = circle(CX, CY, 112)
    outlined(chick, hexc("#FFD84D"), hexc("#9A6A2A"), width=5, warm=hexc("#FFB347"), depth=90)
    # tuft
    for (a0, a1, cx, cy, rr) in [(-math.pi * 0.95, -math.pi * 0.35, CX - 6, CY - 112, 22),
                                 (-math.pi * 0.85, -math.pi * 0.15, CX + 14, CY - 118, 18)]:
        paint(hexc("#9A6A2A"), cov(arc(cx, cy, rr, a0, a1, 4.5)))
    # chick wings
    for sgn in (-1, 1):
        w = ellipse(CX + sgn * 88, CY + 22, 30, 44)
        paint(hexc("#F4C030"), cov(w) * cov(chick))
    # chick eyes
    for sgn in (-1, 1):
        ex, ey = CX + sgn * 40, CY - 18
        paint(hexc("#2B1D1D"), cov(ellipse(ex, ey, 14, 17)))
        paint(hexc("#FFFFFF"), cov(circle(ex - 4, ey - 6, 5.5)))
    # chick beak
    beak = smin(triangle((CX - 17, CY + 2), (CX + 17, CY + 2), (CX, CY + 24)) - 4, circle(CX, CY + 5, 9), 4)
    paint(hexc("#FF8A3D"), cov(beak))
    # chick blush
    for sgn in (-1, 1):
        paint(hexc("#FF8FA3"), np.clip(0.5 - ellipse(CX + sgn * 66, CY + 16, 18, 11) / 6, 0, 1) * 0.6)

    # kitten paws hugging the chick
    for sgn in (-1, 1):
        pw = ellipse(CX + sgn * 112, CY + 25, 58, 66)
        outlined(pw, cream, line, width=5, warm=warm, depth=50)
        for k in (-1, 0, 1):
            tx = CX + sgn * (112 - 30) + k * 0 + (sgn * k * 0)
            ty = CY + 25 + k * 22
            paint(line, cov(segment(tx - sgn * 14, ty, tx + sgn * 2, ty, 2.6)) * 0.8)

    # kitten feet
    for sgn in (-1, 1):
        ft = ellipse(512 + sgn * 120, 965, 78, 42)
        outlined(ft, cream, line, width=5, warm=warm, depth=40)
        for k in (-1, 1):
            paint(line, cov(segment(512 + sgn * 120 + k * 22, 935, 512 + sgn * 120 + k * 22, 952, 2.6)) * 0.8)

    # eyes: huge and glossy
    for sgn in (-1, 1):
        ex, ey = HX + sgn * 128, HY + 40
        eye = ellipse(ex, ey, 74, 86, hx, hy)
        paint(hexc("#2A1E2E"), cov(eye - 3))
        # iris gradient, dark at the top, bright blue at the bottom
        tt = np.clip((hy - (ey - 86)) / 172, 0, 1)[..., None]
        iris_col = hexc("#1E2F6E") * (1 - tt) + hexc("#5EC4FF") * tt
        iris = ellipse(ex, ey + 6, 62, 72, hx, hy)
        paint(iris_col, cov(iris))
        pupil = ellipse(ex, ey + 4, 36, 44, hx, hy)
        paint(hexc("#121426"), cov(pupil))
        # glow at the bottom of the iris
        paint(hexc("#A8ECFF"), np.clip(0.5 - ellipse(ex, ey + 52, 34, 14, hx, hy) / 10, 0, 1) * 0.55)
        # highlights
        paint(hexc("#FFFFFF"), cov(ellipse(ex - 26, ey - 30, 25, 29, hx, hy)))
        paint(hexc("#FFFFFF"), cov(circle(ex + 26, ey + 30, 12, hx, hy)))
        paint(hexc("#FFFFFF"), cov(circle(ex + 32, ey - 38, 6, hx, hy)) * 0.9)

    # blush
    for sgn in (-1, 1):
        bl = ellipse(HX + sgn * 215, HY + 128, 58, 32, hx, hy)
        paint(hexc("#FF8FAB"), np.clip(0.5 - bl / 16, 0, 1) * 0.55)

    # nose and mouth
    nose = smin(triangle((HX - 19, HY + 108), (HX + 19, HY + 108), (HX, HY + 128), hx, hy) - 5,
                circle(HX, HY + 112, 10, hx, hy), 5)
    paint(pink, cov(nose))
    paint(line, cov(segment(HX, HY + 128, HX, HY + 140, 3.2, hx, hy)))
    for sgn in (-1, 1):
        paint(line, cov(arc(HX + sgn * 18, HY + 140, 18, 0.0, math.pi, 3.4, hx, hy)))

    # whiskers
    for sgn in (-1, 1):
        for k in (-1, 0, 1):
            ax, ay = HX + sgn * 255, HY + 105 + k * 22
            bx, by = HX + sgn * 320, HY + 92 + k * 32
            paint(line, cov(segment(ax, ay, bx, by, 2.4, hx, hy)) * 0.75)

    # tiny flower by the right ear
    FX, FY = HX + 205, HY - 205
    for i in range(5):
        a = i * 2 * math.pi / 5 - math.pi / 2
        pd = circle(FX + 24 * math.cos(a), FY + 24 * math.sin(a), 20, hx, hy)
        paint(hexc("#C76A86"), cov(pd - 3))
        paint(hexc("#FFE0EA"), cov(pd))
    paint(hexc("#FFC93C"), cov(circle(FX, FY, 13, hx, hy)))


draw()
out = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
path = sys.argv[1] if len(sys.argv) > 1 else "picture.png"
out.save(path, optimize=True)
print(path)
