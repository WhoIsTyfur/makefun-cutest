"""Offline 3D renderer for the entry picture (dev tool: numpy + Pillow).

Raymarches a signed-distance scene of a chubby kitten hugging a baby chick,
with soft shadows, ambient occlusion and wrap lighting for a soft toy look,
and writes picture.png. main.py (stdlib only) copies that file.
"""
import math
import sys

import numpy as np
from PIL import Image

OUT = 1024
SS = 2 if "--fast" not in sys.argv else 1
N = OUT * SS
F = np.float32


def v3(x, y, z):
    return np.array([x, y, z], dtype=F)


def norm(v):
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ SDF parts
def ell(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-6)


def sph(p, c, r):
    return np.linalg.norm(p - c, axis=1) - r


def rotz(p, c, deg):
    t = math.radians(deg)
    cs, sn = math.cos(t), math.sin(t)
    q = p - c
    x = cs * q[:, 0] + sn * q[:, 1]
    y = -sn * q[:, 0] + cs * q[:, 1]
    return np.stack([x, y, q[:, 2]], axis=1) + c


def capsule(p, a, b, r):
    pa, ba = p - a, b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


HEAD_C = v3(0, 0.62, 0)
TILT = 8.0  # head tilt, degrees

# materials
M_FUR, M_INEAR, M_EYE, M_NOSE, M_CHICK, M_BEAK, M_SCARF, M_GROUND, M_CEYE, M_PETAL, M_FCEN = range(11)


def head_frame(p):
    return rotz(p, HEAD_C, -TILT)


def scene(p, want_mat=False):
    hp = head_frame(p)
    head = ell(hp, HEAD_C, v3(1.28, 1.02, 1.0))
    ears = np.full(len(p), 1e9, dtype=F)
    inner = np.full(len(p), 1e9, dtype=F)
    for s in (-1, 1):
        ec = HEAD_C + v3(s * 0.8, 0.8, 0.08)
        ep = rotz(hp, ec, s * 24)
        ears = np.minimum(ears, ell(ep, ec, v3(0.42, 0.6, 0.22)))
        inner = np.minimum(inner, ell(ep, ec + v3(0, -0.04, -0.15), v3(0.26, 0.42, 0.1)))
    fur = smin(head, ears, 0.12)
    body = ell(p, v3(0, -0.78, 0.12), v3(1.0, 0.86, 0.82))
    fur = smin(fur, body, 0.32)
    feet = np.minimum(ell(p, v3(-0.5, -1.58, -0.42), v3(0.36, 0.2, 0.42)),
                      ell(p, v3(0.5, -1.58, -0.42), v3(0.36, 0.2, 0.42)))
    fur = smin(fur, feet, 0.08)
    paws = np.minimum(ell(p, v3(-0.47, -0.66, -0.96), v3(0.25, 0.27, 0.25)),
                      ell(p, v3(0.47, -0.66, -0.96), v3(0.25, 0.27, 0.25)))
    fur = np.minimum(fur, paws)
    # a small daisy at the base of the right ear
    fc = HEAD_C + v3(0.6, 0.6, -0.68)
    petals = np.full(len(p), 1e9, dtype=F)
    for i in range(5):
        a = i * 2 * math.pi / 5 + 0.3
        petals = np.minimum(petals, ell(hp, fc + v3(0.13 * math.cos(a), 0.13 * math.sin(a), 0.02),
                                        v3(0.09, 0.09, 0.045)))
    fcen = sph(hp, fc + v3(0, 0, -0.035), 0.065)
    eyes = np.minimum(ell(hp, HEAD_C + v3(-0.47, -0.06, -0.84), v3(0.31, 0.38, 0.16)),
                      ell(hp, HEAD_C + v3(0.47, -0.06, -0.84), v3(0.31, 0.38, 0.16)))
    nose = ell(hp, HEAD_C + v3(0, -0.36, -0.99), v3(0.075, 0.05, 0.05))
    chick = sph(p, v3(0, -0.54, -0.9), 0.47)
    beak = ell(p, v3(0, -0.54, -1.38), v3(0.085, 0.06, 0.08))
    ceyes = np.minimum(sph(p, v3(-0.17, -0.43, -1.3), 0.06), sph(p, v3(0.17, -0.43, -1.3), 0.06))
    tuft = np.minimum(np.minimum(capsule(p, v3(0, -0.1, -0.92), v3(-0.07, -0.03, -0.95), 0.033),
                                 capsule(p, v3(0, -0.1, -0.92), v3(0.0, 0.0, -0.95), 0.036)),
                      capsule(p, v3(0, -0.1, -0.92), v3(0.07, -0.04, -0.94), 0.033))
    chick = smin(chick, tuft, 0.05)
    # scarf: a flattened torus around the neck
    q = p - v3(0, -0.12, 0.05)
    rxz = np.sqrt((q[:, 0] / 1.0) ** 2 + (q[:, 2] / 0.88) ** 2)
    scarf = np.sqrt((rxz - 0.84) ** 2 * 0.8 + (q[:, 1] / 1.0) ** 2) - 0.16
    ground = p[:, 1] + 1.74

    parts = [fur, inner, eyes, nose, chick, beak, scarf, ground, ceyes, petals, fcen]
    d = parts[0]
    for x in parts[1:]:
        d = np.minimum(d, x)
    if not want_mat:
        return d
    mats = np.argmin(np.stack(parts, axis=1), axis=1)
    return d, mats


# ------------------------------------------------------------------ camera
CAM = v3(0, 0.5, -9.0)
FOCAL = 5.5  # larger = narrower field of view
ys, xs = np.mgrid[0:N, 0:N].astype(F)
u = ((xs + 0.5) / N - 0.5) * 2
v = -((ys + 0.5) / N - 0.5) * 2
rd = np.stack([u, v, np.full_like(u, FOCAL)], axis=-1).reshape(-1, 3)
rd /= np.linalg.norm(rd, axis=1, keepdims=True)
ro = np.broadcast_to(CAM, rd.shape).astype(F)

# ------------------------------------------------------------------ march
t = np.full(len(rd), 6.0, dtype=F)
hit = np.zeros(len(rd), bool)
alive = np.arange(len(rd))
for i in range(160):
    if len(alive) == 0:
        break
    p = ro[alive] + rd[alive] * t[alive, None]
    d = scene(p)
    t[alive] += d * 0.9
    done = d < 0.0015
    hit[alive[done]] = True
    alive = alive[~done & (t[alive] < 16)]
print("march done", hit.mean())

P = ro + rd * t[:, None]
idx = np.nonzero(hit)[0]
ph = P[idx]
_, mat = scene(ph, want_mat=True)

# normals
e = F(0.0015)
nrm = np.stack([
    scene(ph + v3(e, 0, 0)) - scene(ph - v3(e, 0, 0)),
    scene(ph + v3(0, e, 0)) - scene(ph - v3(0, e, 0)),
    scene(ph + v3(0, 0, e)) - scene(ph - v3(0, 0, e)),
], axis=1)
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9

KEY = norm(v3(-0.45, 0.95, -0.55))
FILL = norm(v3(0.7, 0.2, -0.5))
RIM = norm(v3(0.3, 0.5, 0.9))


def soft_shadow(p, ldir, k=4.0):
    res = np.ones(len(p), dtype=F)
    tt = np.full(len(p), 0.02, dtype=F)
    for _ in range(48):
        h = scene(p + ldir * tt[:, None])
        res = np.minimum(res, k * h / tt)
        tt += np.clip(h, 0.01, 0.25)
    return np.clip(res, 0, 1)


def ambient_occ(p, n):
    occ = np.zeros(len(p), dtype=F)
    sc = 1.0
    for i in range(1, 6):
        hh = 0.03 + 0.07 * i
        dd = scene(p + n * hh)
        occ += (hh - dd) * sc
        sc *= 0.75
    return np.clip(1 - 1.6 * occ, 0, 1)


po = ph + nrm * 0.004
sh = soft_shadow(po, KEY)
ao = ambient_occ(ph, nrm)
print("lighting done")


def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=F)


def srgb_to_lin(c):
    return np.power(c, 2.2)


# base colours per hit
base = np.zeros((len(ph), 3), dtype=F)
pal = {
    M_FUR: "#FFEAD6", M_INEAR: "#FFB3C6", M_EYE: "#1A2250", M_NOSE: "#FF8FAB", M_CHICK: "#FFD84A",
    M_BEAK: "#FF8A3D", M_SCARF: "#8FE3C8", M_GROUND: "#F7C9D8", M_CEYE: "#20161A",
    M_PETAL: "#FFD3E2", M_FCEN: "#FFC93C",
}
for m, c in pal.items():
    base[mat == m] = srgb_to_lin(hexc(c))

# face decals in the head frame: blush, mouth, forehead stripes
hp = head_frame(ph)
fx, fy = hp[:, 0] - HEAD_C[0], hp[:, 1] - HEAD_C[1]
front = (hp[:, 2] < 0) & (mat == M_FUR)
for s in (-1, 1):
    bd = ((fx - s * 0.8) / 0.22) ** 2 + ((fy + 0.36) / 0.13) ** 2
    a = np.clip(1.4 - bd, 0, 1)[:, None] * 0.65 * front[:, None]
    base = base * (1 - a) + srgb_to_lin(hexc("#FF8FAB")) * a
for s in (-1, 1):
    md = np.abs(np.hypot(fx - s * 0.07, fy + 0.42) - 0.07)
    lower = fy < -0.42
    a = (np.clip((0.016 - md) / 0.006, 0, 1) * lower * front)[:, None]
    base = base * (1 - a) + srgb_to_lin(hexc("#5A3A3A")) * a
# ginger ear tips: warm colour toward the top outer corners of the head
furh = mat == M_FUR
g = np.clip((fy - 0.55) / 0.25, 0, 1) * np.clip((np.abs(fx) - 0.4) / 0.2, 0, 1) * furh
base = base * (1 - 0.8 * g[:, None]) + srgb_to_lin(hexc("#E9893A"))[None] * (0.8 * g[:, None])
# toe lines on the paws hugging the chick
for sx in (-0.47, 0.47):
    for k in (-1, 1):
        on = (np.abs(ph[:, 0] - (sx + k * 0.065)) < 0.011) & (ph[:, 1] > -0.9) & (ph[:, 1] < -0.8) \
             & (ph[:, 2] < -1.0) & furh
        base = np.where(on[:, None], srgb_to_lin(hexc("#8A6460"))[None], base)
# a small open smile with a tongue under the w
mr = np.hypot(fx / 0.062, (fy + 0.47) / 0.085)
mouth = (mr < 1) & (fy < -0.475) & front
rim_m = (mr < 1.18) & (fy < -0.465) & front & ~mouth
tongue = mouth & (np.hypot(fx, fy + 0.565) < 0.045)
base = np.where(rim_m[:, None], srgb_to_lin(hexc("#5A3A3A"))[None], base)
base = np.where(mouth[:, None], srgb_to_lin(hexc("#9E3E52"))[None], base)
base = np.where(tongue[:, None], srgb_to_lin(hexc("#FF8FA6"))[None], base)
for sx in (-0.17, 0, 0.17):
    sd = np.hypot(np.maximum(np.abs(fy - 0.78) - 0.12, 0), fx - sx * (1 + 0.15 * (fy - 0.78)))
    a = (np.clip((0.05 - sd) / 0.01, 0, 1) * front)[:, None] * 0.95
    base = base * (1 - a) + srgb_to_lin(hexc("#E07A2C")) * a

# eyes: iris gradient, pupil and painted highlights in each eye's local frame
eyem = mat == M_EYE
for s in (-1, 1):
    ex, ey = s * 0.47, -0.06
    lx, ly = (fx - ex) / 0.31, (fy - ey) / 0.38
    inside = eyem & (np.hypot(lx, ly) < 1.25)
    tt = np.clip((ly + 1) / 2, 0, 1)  # 0 bottom, 1 top
    iris = srgb_to_lin(hexc("#5CC8FF"))[None] * (1 - tt[:, None]) + srgb_to_lin(hexc("#1C2C6C"))[None] * tt[:, None]
    r = np.hypot(lx, ly)
    col = np.where((r < 0.84)[:, None], iris, srgb_to_lin(hexc("#141022"))[None])
    pup = np.hypot(lx / 0.52, (ly - 0.02) / 0.6) < 1
    col = np.where(pup[:, None], srgb_to_lin(hexc("#0E0F1E"))[None], col)
    h1 = np.hypot((lx + 0.36) / 0.3, (ly - 0.36) / 0.34) < 1
    h2 = np.hypot(lx - 0.38, ly + 0.4) < 0.15
    col = np.where((h1 | h2)[:, None], F(1.0), col)
    base = np.where(inside[:, None], col, base)

# chick: blush on its cheeks and a catchlight in each eye
for s in (-1, 1):
    bd = ((ph[:, 0] - s * 0.29) / 0.09) ** 2 + ((ph[:, 1] + 0.58) / 0.055) ** 2
    a = (np.clip(1.4 - bd, 0, 1) * 0.7 * (mat == M_CHICK))[:, None]
    base = base * (1 - a) + srgb_to_lin(hexc("#FF8FA3")) * a
    cl = np.hypot(ph[:, 0] - (s * 0.17 - 0.022), ph[:, 1] - (-0.41)) < 0.022
    base = np.where((cl & (mat == M_CEYE))[:, None], F(1.0), base)

# toe lines on the feet
for sx in (-0.5, 0.5):
    for k in (-1, 1):
        td = np.abs(ph[:, 0] - (sx + k * 0.1))
        on = (td < 0.014) & (ph[:, 1] > -1.6) & (ph[:, 1] < -1.47) & (ph[:, 2] < -0.6) & (mat == M_FUR)
        base = np.where(on[:, None], srgb_to_lin(hexc("#8A6460"))[None], base)

# plush fur: a little noise in the normals
furm = mat == M_FUR
q = ph * 38.0
nz = np.stack([np.sin(q[:, 1] * 1.3 + np.sin(q[:, 2] * 0.7)) * np.cos(q[:, 0] * 0.9),
               np.sin(q[:, 2] * 1.1 + np.sin(q[:, 0] * 0.8)) * np.cos(q[:, 1] * 1.2),
               np.sin(q[:, 0] * 1.2 + np.sin(q[:, 1] * 0.9)) * np.cos(q[:, 2] * 1.0)], axis=1)
nrm = np.where(furm[:, None], nrm + 0.025 * nz, nrm)
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

# ------------------------------------------------------------------ shading
col = np.zeros_like(base)
wrap = np.where(np.isin(mat, [M_FUR, M_CHICK, M_SCARF, M_INEAR]), 0.45, 0.1).astype(F)
ndl = nrm @ KEY
shs = np.where(mat == M_GROUND, sh, 0.45 + 0.55 * sh)
diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1) * shs
fill = np.clip(nrm @ FILL, 0, 1)
sky = 0.5 + 0.5 * nrm[:, 1]
rim = np.clip(1 - np.sum(nrm * -rd[idx], axis=1), 0, 1) ** 3  # fresnel-ish edge
key_col = np.array([1.0, 0.95, 0.88], dtype=F) * 1.85
fill_col = np.array([0.85, 0.9, 1.0], dtype=F) * 0.35
sky_col = np.array([1.0, 0.92, 0.95], dtype=F) * 0.55
col = base * (key_col * diff[:, None] + fill_col * fill[:, None] + sky_col * (sky * ao)[:, None])
# warm subsurface tint where the light wraps into shadow
sss = (np.clip(wrap - np.abs(ndl), 0, 1) * (wrap > 0.2))[:, None] * np.array([0.35, 0.12, 0.08], dtype=F)
col += sss * base * 0.9
col += (rim * ao)[:, None] * np.array([1.0, 0.9, 0.95], dtype=F) * 0.25 * (mat != M_GROUND)[:, None]
# velvet sheen on the fur
vdn = np.clip(np.sum(nrm * -rd[idx], axis=1), 0, 1)
sheen = (1 - vdn) ** 2 * furm
col += (sheen * ao)[:, None] * np.array([1.0, 0.85, 0.8], dtype=F) * 0.22
# specular
hv = KEY - rd[idx]
hv /= np.linalg.norm(hv, axis=1, keepdims=True)
gloss = np.where(np.isin(mat, [M_EYE, M_CEYE, M_NOSE]), 200.0, 24.0).astype(F)
ks = np.where(np.isin(mat, [M_EYE, M_CEYE, M_NOSE]), 1.6, 0.06).astype(F)
col += (np.clip(np.sum(nrm * hv, axis=1), 0, 1) ** gloss * ks * sh)[:, None]

# ground: fade into the background
bg_top, bg_bot = srgb_to_lin(hexc("#FFF0F5")), srgb_to_lin(hexc("#F7BCD2"))
vy = (v.reshape(-1) + 1) / 2
bg = bg_bot[None] * (1 - vy[:, None]) + bg_top[None] * vy[:, None]
gm = mat == M_GROUND
gsh = (sh * ao)[gm]
col[gm] = bg[idx[gm]] * (0.62 + 0.38 * gsh[:, None])

# soft bokeh in the background
px = (u.reshape(-1) + 1) / 2
py = 1 - vy
for (bx, by, br, ba) in [(0.13, 0.3, 0.09, 0.22), (0.88, 0.42, 0.1, 0.2), (0.1, 0.72, 0.06, 0.18),
                         (0.9, 0.12, 0.06, 0.2), (0.82, 0.75, 0.07, 0.16), (0.2, 0.08, 0.05, 0.2)]:
    dd = np.hypot(px - bx, py - by) - br
    bg = bg + (np.clip(0.5 - dd / 0.012, 0, 1) * ba)[:, None] * (1.0 - bg)
img = bg.copy()
img[idx] = col
# tone map and gamma
img = img / (1 + img * 0.12)
img = np.power(np.clip(img, 0, 1), 1 / 2.2)
img = img.reshape(N, N, 3)


def fluff(img, which, behind, count_scale, length, seed):
    """Short hair strands along the silhouette of material `which` where it meets `behind`."""
    from PIL import ImageDraw
    matfull = np.full(N * N, -1, dtype=np.int16)
    matfull[idx] = mat
    matfull = matfull.reshape(N, N)
    nfull = np.zeros((N * N, 2), dtype=F)
    nfull[idx] = nrm[:, :2]
    nfull = nfull.reshape(N, N, 2)
    m = matfull == which
    sep = 2 * SS
    edge = np.zeros_like(m)
    for dy, dx in ((sep, 0), (-sep, 0), (0, sep), (0, -sep)):
        nb = np.roll(np.roll(matfull, dy, axis=0), dx, axis=1)
        edge |= m & np.isin(nb, behind)
    ys_, xs_ = np.nonzero(edge)
    rng = np.random.default_rng(seed)
    keep = rng.random(len(ys_)) < count_scale
    ys_, xs_ = ys_[keep], xs_[keep]
    order = rng.permutation(len(ys_))
    ys_, xs_ = ys_[order], xs_[order]
    base_img = Image.fromarray((img * 255 + 0.5).astype(np.uint8), "RGB")
    layer = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    for y, x in zip(ys_, xs_):
        nx, ny = nfull[y, x]
        ln = math.hypot(nx, ny)
        if ln < 0.2:
            continue
        dx, dy = nx / ln, -ny / ln
        ang = rng.normal(0, 0.35)
        ca, sa = math.cos(ang), math.sin(ang)
        dx, dy = dx * ca - dy * sa, dx * sa + dy * ca
        L = length * SS * (0.5 + rng.random())
        x0, y0 = x - dx * 3 * SS, y - dy * 3 * SS
        c = img[min(N - 1, max(0, int(y - dy * 4 * SS))), min(N - 1, max(0, int(x - dx * 4 * SS)))]
        rgb = tuple(int(min(255, v * 255 * 1.02)) for v in c)
        dr.line([(x0, y0), (x0 + dx * L * 0.6, y0 + dy * L * 0.6)], fill=rgb + (235,), width=max(1, SS))
        dr.line([(x0 + dx * L * 0.6, y0 + dy * L * 0.6), (x0 + dx * L, y0 + dy * L)], fill=rgb + (120,), width=1)
    base_img.paste(layer, (0, 0), layer)
    return np.asarray(base_img, dtype=F) / 255


if "--nofluff" not in sys.argv:
    misses = [-1, M_GROUND]
    img = fluff(img, M_FUR, misses, 0.55, 9, 11)
    img = fluff(img, M_CHICK, misses + [M_FUR, M_SCARF], 0.6, 7, 12)
out = Image.fromarray((img * 255 + 0.5).astype(np.uint8), "RGB")
if SS > 1:
    out = out.resize((OUT, OUT), Image.LANCZOS)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0] if args else "picture.png"
out.save(path, optimize=True)
print(path)
