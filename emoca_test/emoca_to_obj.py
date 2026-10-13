"""
EMOCA test: photos of DREAM -> textured 3D head (OBJ).

Separate test program, not part of DREAM itself. Uses only EMOCA's encoders and
the FLAME head model, so it needs no pytorch3d (which is hard to install on Windows).

For each photo it writes a textured OBJ of the face as it is in the photo, and then
one combined head: the average face shape, neutral expression, with the texture
blended from every photo (each part taken from the photo that sees it best).

Run with DREAM's venv311:
    venv311\\Scripts\\python.exe emoca_test\\emoca_to_obj.py
    venv311\\Scripts\\python.exe emoca_test\\emoca_to_obj.py --images images\\dream.jpg

Models (non-commercial research licence, see https://emoca.is.tue.mpg.de):
    emoca_test/assets/FLAME/...                      from FLAME.zip
    emoca_test/assets/EMOCA/models/EMOCA_v2_lr_mse_20/  from EMOCA_v2_lr_mse_20.zip
"""

import argparse
import os
import pickle
import sys
import types
from pathlib import Path

# face_alignment calls torch.compile, which needs Triton (not available on Windows)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage.transform import estimate_transform, warp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "emoca"))

# EMOCA's ResNet builder downloads ImageNet weights we overwrite straight away - skip that.
import gdl.models.ResNet as _resnet  # noqa: E402
_resnet.load_ResNet50Model = lambda: _resnet.ResNet(_resnet.Bottleneck, [3, 4, 6, 3])
from gdl.models.DecaEncoder import ResnetEncoder  # noqa: E402
import multiview_fit  # noqa: E402

CROP = 224          # what the encoders were trained on
HI_RES = 1536       # crop the texture is sampled from
TEX_CROP = 1.6      # ...and it's this much wider than the face crop, so it takes in the whole head
EYE_IRIS_R = 0.44   # the eye picture's iris radius, as a fraction of the eyeball's
BACK_FADE = (-0.06, -0.10)  # FLAME depth (z) where the profiles fade from full to nothing (ears ~-0.07)
FEATHER = 0.04      # each photo fades out over this fraction of the crop near its edges
TEX_SIZE = 1024
TONE_DIP = 81       # texels: darker dips in the scalp's skin tone narrower than this are lifted (even_tone)


# ---------------------------------------------------------------- loading

class _Stub:
    """Stands in for any class a pickle names but we don't have (chumpy, lightning)."""
    def __init__(self, *a, **k):
        self.__dict__["_state"] = {}

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __array__(self, dtype=None, copy=None):
        s = self.__dict__["_state"]
        x = s.get("x", s) if isinstance(s, dict) else s
        return np.asarray(x, dtype=dtype)

    def __getattr__(self, k):
        s = self.__dict__.get("_state", {})
        if isinstance(s, dict) and k in s:
            return s[k]
        raise AttributeError(k)


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_Stub,), {})


_lenient = types.SimpleNamespace(Unpickler=_LenientUnpickler,
                                 load=lambda f, **kw: _LenientUnpickler(f, **kw).load(),
                                 __name__="lenient_pickle")


def load_pickle(path):
    with open(path, "rb") as f:
        return _LenientUnpickler(f, encoding="latin1").load()


def find_checkpoint(run_dir):
    ckpts = sorted((run_dir / "detail").rglob("*.ckpt")) or sorted(run_dir.rglob("*.ckpt"))
    if not ckpts:
        sys.exit(f"No .ckpt found under {run_dir}")
    def loss(p):
        try:
            return float(p.stem[p.stem.rfind("=") + 1:])
        except ValueError:
            return float("inf")
    best = [p for p in ckpts if p.stem != "last"]
    return min(best, key=loss) if best else ckpts[-1]


def load_model(assets, model_name, device):
    from omegaconf import OmegaConf
    run_dir = assets / "EMOCA" / "models" / model_name
    cfg = OmegaConf.load(run_dir / "cfg.yaml").detail.model
    n = {k: int(cfg[k]) for k in ("n_shape", "n_tex", "n_exp", "n_pose", "n_cam", "n_light")}
    jaw, glob = bool(cfg.get("exp_deca_jaw_pose", False)), bool(cfg.get("exp_deca_global_pose", False))
    n_exp_param = n["n_exp"] + (n["n_pose"] if (jaw and glob) else 3 if (jaw or glob) else 0)
    if cfg.get("expression_backbone", "deca_clone") != "deca_clone":
        sys.exit(f"Unsupported expression_backbone {cfg.expression_backbone}")

    ckpt_path = find_checkpoint(run_dir)
    print(f"checkpoint: {ckpt_path.relative_to(assets)}")
    ckpt = torch.load(ckpt_path, map_location="cpu", pickle_module=_lenient, weights_only=False)
    sd = ckpt["state_dict"]

    def sub(prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    e_flame = ResnetEncoder(outsize=sum(n.values()))
    e_exp = ResnetEncoder(outsize=n_exp_param)
    e_flame.load_state_dict(sub("deca.E_flame."))
    e_exp.load_state_dict(sub("deca.E_expression."))
    e_flame.to(device).eval()
    e_exp.to(device).eval()
    return e_flame, e_exp, n, jaw, glob


def load_flame(assets, n, device):
    from gdl.models import DecaFLAME
    DecaFLAME.pickle.load = lambda f, **kw: _LenientUnpickler(f, **kw).load()
    geo = assets / "FLAME" / "geometry"
    cfg = types.SimpleNamespace(flame_model_path=str(geo / "generic_model.pkl"),
                                flame_lmk_embedding_path=str(geo / "landmark_embedding.npy"),
                                n_shape=n["n_shape"], n_exp=n["n_exp"])
    return DecaFLAME.FLAME(cfg).to(device).eval()


def load_uv_template(assets):
    """FLAME's UV layout from head_template.obj: uv coords and the uv index per face corner."""
    vt, ft = [], []
    with open(assets / "FLAME" / "geometry" / "head_template.obj") as f:
        for line in f:
            if line.startswith("vt "):
                vt.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                ft.append([int(c.split("/")[1]) - 1 for c in line.split()[1:4]])
    return np.array(vt, np.float32), np.array(ft, np.int64)


# ---------------------------------------------------------------- face crop

def detect_crop(fa, img):
    """EMOCA's crop: FAN 68 landmarks -> square box x1.1, then x1.25."""
    lmks = fa.get_landmarks(img)
    if not lmks:
        return None
    kpt = max(lmks, key=lambda k: np.ptp(k[:, 0]) * np.ptp(k[:, 1]))
    left, right = kpt[:, 0].min(), kpt[:, 0].max()
    top, bottom = kpt[:, 1].min(), kpt[:, 1].max()
    old_size = (right - left + bottom - top) / 2 * 1.1
    cx, cy = (left + right) / 2, (top + bottom) / 2
    size = old_size * 1.25
    src = np.array([[cx - size / 2, cy - size / 2], [cx - size / 2, cy + size / 2], [cx + size / 2, cy - size / 2]])
    return src, kpt


def crop(img, src, res):
    dst = np.array([[0, 0], [0, res - 1], [res - 1, 0]])
    tform = estimate_transform("similarity", src, dst)
    out = warp(img / 255.0, tform.inverse, output_shape=(res, res))
    return out.astype(np.float32), tform


# ---------------------------------------------------------------- rasterising

def rasterize(px, z, faces, h, w):
    """Z-buffer: per pixel, the nearest face (bigger z = closer). Returns depth and face id."""
    zbuf = np.full((h, w), -np.inf, np.float32)
    fid = np.full((h, w), -1, np.int64)
    tri = px[faces]
    tz = z[faces]
    for i in range(len(faces)):
        (x0, y0), (x1, y1), (x2, y2) = tri[i]
        xmin, xmax = int(max(np.floor(min(x0, x1, x2)), 0)), int(min(np.ceil(max(x0, x1, x2)), w - 1))
        ymin, ymax = int(max(np.floor(min(y0, y1, y2)), 0)), int(min(np.ceil(max(y0, y1, y2)), h - 1))
        if xmin > xmax or ymin > ymax:
            continue
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
        a = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / d
        b = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / d
        c = 1 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        zz = a * tz[i, 0] + b * tz[i, 1] + c * tz[i, 2]
        region = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        closer = inside & (zz > region)
        region[closer] = zz[closer]
        fid[ymin:ymax + 1, xmin:xmax + 1][closer] = i
    return zbuf, fid


def uv_raster(vt, ft, size):
    """For each texel: which face covers it and its barycentric coords."""
    uv = np.stack([vt[:, 0] * size, (1 - vt[:, 1]) * size], 1)
    fid = np.full((size, size), -1, np.int64)
    bary = np.zeros((size, size, 3), np.float32)
    for i, (i0, i1, i2) in enumerate(ft):
        (x0, y0), (x1, y1), (x2, y2) = uv[i0], uv[i1], uv[i2]
        xmin, xmax = int(max(np.floor(min(x0, x1, x2)) - 1, 0)), int(min(np.ceil(max(x0, x1, x2)) + 1, size - 1))
        ymin, ymax = int(max(np.floor(min(y0, y1, y2)) - 1, 0)), int(min(np.ceil(max(y0, y1, y2)) + 1, size - 1))
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
        a = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / d
        b = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / d
        c = 1 - a - b
        inside = (a >= -1e-3) & (b >= -1e-3) & (c >= -1e-3)
        sub_f = fid[ymin:ymax + 1, xmin:xmax + 1]
        sub_b = bary[ymin:ymax + 1, xmin:xmax + 1]
        sub_f[inside] = i
        sub_b[inside] = np.stack([a, b, c], -1)[inside]
    return fid, bary


def vertex_normals(v, faces):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


# ---------------------------------------------------------------- output

def fill_unseen(tex, seen, covered):
    """Texels no photo saw get a smooth blend of the colours around them (push-pull:
    average down to coarse levels, then fill holes back up from them)."""
    if not seen.any():
        return tex
    levels = [(tex * seen[..., None], seen.astype(np.float32))]
    while levels[-1][1].shape[0] > 1:
        c, w = levels[-1]
        h = c.shape[0] // 2
        c2 = c.reshape(h, 2, h, 2, 3).sum((1, 3))
        w2 = w.reshape(h, 2, h, 2).sum((1, 3))
        levels.append((c2, w2))
    fill = levels[-1][0] / np.maximum(levels[-1][1], 1e-9)[..., None]
    for c, w in reversed(levels[:-1]):
        up = ndimage.zoom(fill, (2, 2, 1), order=1)
        mine = c / np.maximum(w, 1e-9)[..., None]
        a = np.clip(w, 0, 1)[..., None]
        fill = a * mine + (1 - a) * up
    # texels outside the UV islands keep the fill too, so filtering at the seams doesn't pull in black
    return np.where(seen[..., None], tex, fill)


def write_obj(path, verts, faces, vt, ft, tex):
    path = Path(path)
    png = path.with_suffix(".png").name
    Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8)).save(path.with_suffix(".png"))
    with open(path.with_suffix(".mtl"), "w") as f:
        f.write(f"newmtl head\nKd 1 1 1\nKa 0 0 0\nKs 0 0 0\nmap_Kd {png}\n")
    with open(path, "w") as f:
        f.write(f"mtllib {path.with_suffix('.mtl').name}\nusemtl head\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in vt:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for (a, b, c), (ta, tb, tc) in zip(faces + 1, ft + 1):
            f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")


# ---------------------------------------------------------------- back views

def square(cx, cy, s):
    return np.array([[cx - s / 2, cy - s / 2], [cx - s / 2, cy + s / 2], [cx + s / 2, cy - s / 2]])


def project(v, c, res):
    """DECA's orthographic camera: (xy + t) * s, image y points down."""
    px = np.stack([(v[:, 0] + c[1]) * c[0], -(v[:, 1] + c[2]) * c[0]], 1)
    return (px + 1) * res / 2


def studio_mask(img):
    """Her silhouette when the photo has a black studio background, else None. Holes are filled so
    dark parts of her (brows, lashes, shadows) still count as her."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32).sum(1)
    if np.median(border) >= 60:
        return None
    mask = ndimage.binary_opening(img.astype(np.float32).sum(2) > 90, iterations=2)
    return ndimage.binary_fill_holes(mask)


def fit_silhouette(target, verts_of, x0, steps, faces, keep_frac):
    """Line the head up with her silhouette (the photo's mask, cropped like the mesh's view).
    x = head rotation (axis-angle, 3) + camera (3); verts_of(rotation) gives the mesh. Only the head
    is compared: the mesh is cut at the neck, the photo carries on to the shoulders.
    Returns the best x and the match before and after."""
    import cv2
    from scipy.optimize import minimize
    res = target.shape[0]

    def cost(x):
        v = verts_of(x[:3])
        hy = v[:, 1]
        keep = hy > hy.min() + keep_frac * np.ptp(hy)
        cut = int(np.clip(project(v[keep], x[3:], res)[:, 1].max(), 0, res))
        sil = np.zeros((res, res), np.uint8)
        # one triangle at a time: fillPoly with all of them cancels out wherever they overlap
        for tri in np.round(project(v, x[3:], res)[faces] * 8).astype(np.int32):
            cv2.fillConvexPoly(sil, tri, 1, shift=3)
        sil, t = sil[:cut].astype(bool), target[:cut]
        return 1 - (sil & t).sum() / max((sil | t).sum(), 1)

    simplex = np.array([x0] + [x0 + np.eye(6)[i] * steps[i] for i in range(6)])
    best = minimize(cost, x0, method="Nelder-Mead",
                    options=dict(initial_simplex=simplex, xatol=1e-4, fatol=1e-4, maxiter=800))
    return best.x, 1 - cost(x0), 1 - best.fun


def back_start(mask, v):
    """Back views have no face for the encoders. Crop square and starting camera from her
    silhouette's bounding box: same width, same top, centred. v is the head turned round."""
    rows = np.nonzero(mask.any(1))[0]
    top = rows[0]
    # the skull: rows from the top down to roughly the ears, where the neck hasn't started
    upper = mask[top:top + int(0.45 * (rows[-1] - top))]
    xs = np.nonzero(upper.any(0))[0]
    width = xs[-1] - xs[0]
    src = square((xs[0] + xs[-1]) / 2, top + 0.6 * width, 1.6 * width)
    hy = v[:, 1]
    keep = hy > hy.min() + 0.45 * np.ptp(hy)
    side = src[2, 0] - src[0, 0]
    s0 = (2 * width / side) / np.ptp(v[keep, 0])
    top_norm = 2 * (top - src[0, 1]) / side - 1
    return src, np.array([s0, -v[keep, 0].mean(), -top_norm / s0 - v[:, 1].max()])


def _eye_mask(hi, poly, erode=3):
    import cv2
    mask = np.zeros(hi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return ndimage.binary_erosion(mask.astype(bool), iterations=erode)


def measure_iris(hi, poly):
    """Her iris in the front photo's texture crop: centre and radius in pixels. It is the dark
    disc between her lids; its width is safe to measure, the lids only cut its top and bottom."""
    ys, xs = np.nonzero(_eye_mask(hi, poly))
    lum = hi[ys, xs].mean(1)
    dark = lum < np.percentile(lum, 45)
    cx, cy = np.median(xs[dark]), np.median(ys[dark])
    # width only in a band through its middle: lash shadows at the corners are dark too
    band = dark & (np.abs(ys - cy) < 0.15 * np.ptp(ys))
    x0, x1 = np.percentile(xs[band], [5, 95])
    return np.array([cx, cy]), (x1 - x0) / 2


def her_eye_picture(eye_img, hi, polys, irises, bins=24):
    """The eye picture recoloured as her eye. Her iris is measured ring by ring out from its
    centre (pupil, lighter middle, darker outer ring) over the visible parts of both eyes, minus
    lashes and the glint; the picture keeps only its fine fibre detail. Its white becomes her
    eye white, with just a trace of the picture's veins."""
    acc, n = np.zeros((bins, 3)), np.zeros(bins)
    white = []
    for poly, (c, r) in zip(polys, irises):
        ys, xs = np.nonzero(_eye_mask(hi, poly))
        px = hi[ys, xs]
        lum = px.mean(1)
        d = np.hypot(xs - c[0], ys - c[1]) / r
        sel = (d < 1) & (lum < 0.8)
        b = np.minimum((d[sel] * bins).astype(int), bins - 1)
        np.add.at(acc, b, px[sel])
        np.add.at(n, b, 1)
        out_ = (d > 1.2) & (lum > np.percentile(lum, 60))
        white.append(px[out_])
    have = n > 5
    prof = acc / np.maximum(n, 1)[:, None]
    prof = np.stack([np.interp(np.arange(bins), np.nonzero(have)[0], prof[have, k]) for k in range(3)], 1)
    white = np.concatenate(white).mean(0)

    h, w = eye_img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    t = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2) / (w / 2) / EYE_IRIS_R     # 1 at the iris edge
    pic = eye_img.mean(2)
    tb = np.minimum((t * bins).astype(int), bins - 1)
    ring = np.bincount(tb[t < 1].ravel(), pic[t < 1].ravel(), bins) / np.maximum(
        np.bincount(tb[t < 1].ravel(), minlength=bins), 1)
    detail = np.clip(pic / np.maximum(ring[tb], 1e-3), 0.75, 1.25)[..., None]
    iris = np.clip(prof[tb] * detail, 0, 1)
    rest = pic[(t > 1.2) & (t < 2.0)].mean()
    clean = np.clip(white * (0.9 + 0.1 * pic[..., None] / max(rest, 1e-3)), 0, 1)
    edge = np.clip((t - 1) / 0.08, 0, 1)[..., None]                               # soft limbus
    return ((1 - edge) * iris + edge * clean).astype(np.float32)


def iris_aim(poly, lids3d, iris):
    """Where her iris is, as a 3D point on this head: how far it sits between her eye corners
    and lids in the photo, put between the head's own eyelid landmarks (lids3d, in the same
    order as poly: corner, two upper, corner, two lower)."""
    c = iris[0]
    a, b = poly[0], poly[3]
    axis = (b - a) / np.linalg.norm(b - a)
    t = np.dot(c - a, axis) / np.linalg.norm(b - a)
    down_px = np.dot(c - poly.mean(0), [-axis[1], axis[0]])         # image y points down
    gap_px = np.linalg.norm(poly[[1, 2]].mean(0) - poly[[4, 5]].mean(0))
    A, B = lids3d[0], lids3d[3]
    gap3d = np.linalg.norm(lids3d[[1, 2]].mean(0) - lids3d[[4, 5]].mean(0))
    point = lids3d.mean(0) + (t - 0.5) * (B - A)
    point[1] -= down_px / max(gap_px, 1e-6) * gap3d
    return point


def wrap_eyeball(eye_img, v, faces, eyeball, uv, ball_t, aims=None, iris_r=None):
    """Colour for each eyeball texel from a front-on picture of an eye (iris centred, the
    eyeball's outline touching the picture's edges). Each iris points at its eye's aim (a 3D
    point: where her iris is), or straight ahead without one, and is iris_r across (radius,
    per aim) or a typical size. The front half is projected straight on; the back half, which
    the lids always hide, takes sclera from near the rim."""
    tf, tbary, _ = uv
    ball_v = np.unique(faces[eyeball])
    p = (v[faces[tf[ball_t]]] * tbary[ball_t][..., None]).sum(1)      # 3D point per texel
    out = np.zeros((len(p), 3), np.float32)
    h, w = eye_img.shape[:2]
    for side in (v[ball_v, 0] > 0, v[ball_v, 0] <= 0):              # each eye separately
        ev = v[ball_v[side]]
        centre = ev.mean(0)
        radius = np.linalg.norm(ev - centre, axis=1).mean()
        mine = (p[:, 0] > 0) == (centre[0] > 0)
        fwd = np.array([0.0, 0.0, 1.0])                               # FLAME looks down +z
        zoom = EYE_IRIS_R / 0.49                                      # a typical iris: ~half the eyeball
        if aims:
            k = min(range(len(aims)), key=lambda i: np.linalg.norm(aims[i] - centre))
            fwd = (aims[k] - centre) / np.linalg.norm(aims[k] - centre)
            if iris_r:
                zoom = EYE_IRIS_R * radius / iris_r[k]
        right = np.cross([0, 1, 0], fwd)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        d = (p[mine] - centre) / radius
        d = np.stack([d @ right, d @ up, d @ fwd], 1)                 # in the eye's own frame
        back = d[:, 2] < 0
        xy = d[:, :2].copy()
        xy[back] *= 0.92 / np.maximum(np.linalg.norm(xy[back], axis=1, keepdims=True), 1e-6)
        xy = np.clip(xy * zoom, -0.98, 0.98)
        out[mine] = bilinear(eye_img, (0.5 + 0.5 * xy[:, 0]) * (w - 1), (0.5 - 0.5 * xy[:, 1]) * (h - 1))
    return out


def head_parts(assets):
    """Regions of the FLAME head the texturing treats specially:
    eyeball / eye_area: per face, eyeballs and the skin around them (lids, within ~1 cm)
    ear_side: per face, +1 / -1 for the left / right ear, else 0
    face_tex: per texel, FLAME's face mask (softened) - the area the front photo owns"""
    from scipy.spatial import cKDTree
    m = load_pickle(assets / "FLAME" / "geometry" / "generic_model.pkl")
    f = np.asarray(m["f"]).astype(np.int64)
    v = np.asarray(m["v_template"])
    ball = np.asarray(m["weights"])[:, 3:].max(1) > 0.5
    near = cKDTree(v[ball]).query(v, distance_upper_bound=0.012)[0] < 0.012

    # ears: around the point sticking out furthest sideways at ear height (the rim)
    ear_side = np.zeros(len(f), np.int64)
    for sgn in (1, -1):
        band = (v[:, 1] > -0.10) & (v[:, 1] < 0.06) & (sgn * v[:, 0] > 0)
        tip = v[band][np.argmax(sgn * v[band, 0])]
        c = tip - np.array([sgn * 0.012, 0, 0])
        ear = (np.linalg.norm(v - c, axis=1) < 0.04) & (sgn * v[:, 0] > sgn * c[0] - 0.012)
        ear_side[ear[f].all(1)] = sgn

    mask = Image.open(assets / "FLAME" / "mask" / "uv_face_mask.png").convert("L").resize((TEX_SIZE, TEX_SIZE))
    # pulled in from the mask's edge and faded over a wide band, so front and side photos blend
    # across the cheeks instead of meeting at a line
    face = ndimage.binary_erosion(np.asarray(mask) > 127, iterations=TEX_SIZE // 32)
    face_tex = ndimage.gaussian_filter(face.astype(np.float32), TEX_SIZE / 40)
    return dict(eyeball=ball[f].all(1), eye_area=(near & ~ball)[f].any(1) & ~ball[f].all(1), template_z=v[:, 2],
                ear_side=ear_side, face_tex=np.clip(face_tex, 0, 1))


def match_colour(colour, weight, ref, ref_w, covered):
    """Scale a view's colours to agree with ref where both see the head. Only the broad tone
    is matched (a heavily blurred gain), then extended smoothly over the rest of the view."""
    both = ((weight > 0) & (ref_w > 0)).astype(np.float32)
    if both.sum() < 500:
        return colour
    sigma = TEX_SIZE / 24
    norm = np.maximum(ndimage.gaussian_filter(both, sigma), 1e-6)
    gain = np.stack([ndimage.gaussian_filter(ref[..., k] * both, sigma)
                     / np.maximum(ndimage.gaussian_filter(colour[..., k] * both, sigma), 1e-6)
                     for k in range(3)], -1)
    known = norm > 0.05
    gain = fill_unseen(np.where(known[..., None], gain, 0), known, covered)
    return np.clip(colour * np.clip(gain, 0.6, 1.6), 0, 1)


def even_tone(tex, seen, covered, face_tex, keep):
    """The profiles see the skin behind her ears, up to the crown, in shadow, and leave a dark band
    there. Skin tone over the bald scalp should change only slowly, so lift dips in its broad
    brightness narrower than TONE_DIP (a grey closing); the fine detail (pores) stays. Not on
    the face (face_tex), whose shading the front photo gets right, nor near keep (ears, eyeballs)."""
    sigma = TEX_SIZE / 64
    filled = fill_unseen(np.where(seen[..., None], tex, 0), seen, covered)
    lum = ndimage.gaussian_filter(filled.mean(-1), sigma)
    lifted = ndimage.gaussian_filter(ndimage.grey_closing(lum, size=(TONE_DIP, TONE_DIP)), sigma)
    free = (1 - face_tex) * ~ndimage.binary_dilation(keep, iterations=8)
    free = ndimage.gaussian_filter(free.astype(np.float32), sigma / 2)
    gain = 1 + free * (np.clip(lifted / np.maximum(lum, 1e-3), 1, 1.8) - 1)
    return np.where(seen[..., None], np.clip(tex * gain[..., None], 0, 1), tex)


# ---------------------------------------------------------------- main

def bake_view(name, hi, inside, v, cam, faces, vt, ft, uv, out, parts, eye_polys=None):
    """Texture from one view, its OBJ and a preview of the fit. Returns colour and per-texel weight.
    parts: head_parts(); eye_polys: her eye outlines in the crop, known for the front photo only.
    The front photo owns the face and eyes; the others texture the rest (scalp, ears, neck)."""
    tf, tbary, covered = uv
    eyeball, eye_area = parts["eyeball"], parts["eye_area"]
    front = eye_polys is not None
    px = project(v, cam, HI_RES)
    z = v[:, 2]

    zbuf, fid = rasterize(px, z, faces, HI_RES, HI_RES)
    vn = vertex_normals(v, faces)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9

    b = tbary[..., :, None]
    f3 = faces[tf]
    tp = (px[f3] * b).sum(-2)
    tz = (z[f3] * tbary).sum(-1)
    tn = (vn[f3] * b).sum(-2)
    tn /= np.linalg.norm(tn, axis=-1, keepdims=True) + 1e-9
    ix = np.clip(tp[..., 0].astype(int), 0, HI_RES - 1)
    iy = np.clip(tp[..., 1].astype(int), 0, HI_RES - 1)
    depth_ok = tz >= zbuf[iy, ix] - 0.004
    in_img = (tp[..., 0] >= 0) & (tp[..., 0] < HI_RES) & (tp[..., 1] >= 0) & (tp[..., 1] < HI_RES)
    in_img &= inside[iy, ix]
    facing = np.clip(tn[..., 2], 0, 1)
    weight = np.where(covered & depth_ok & in_img, facing ** 2, 0).astype(np.float32)
    # fade out towards the edge of what this photo shows, so photos blend instead of meeting at a line
    fade = np.clip(ndimage.distance_transform_edt(inside) / (FEATHER * HI_RES), 0, 1)
    weight *= fade[iy, ix]
    weight[weight < 0.05] = 0
    if eye_polys is None:
        # Seen nearly edge-on from the sides and back, skin darkens as it turns from the light; that
        # strip (behind the ears) is better filled from the skin around it than taken from there
        weight[facing < 0.35] = 0
    if parts.get("side_view"):
        # the back of her skull is in shadow in the profile photos; the back photo is evenly lit
        # there, so the profiles fade out behind the ear
        weight *= np.clip(((parts["template_z"][f3] * tbary).sum(-1) - BACK_FADE[1])
                          / (BACK_FADE[0] - BACK_FADE[1]), 0, 1)

    ear_t = covered & (parts["ear_side"][tf] != 0)
    if front:
        weight[ear_t] = 0               # seen edge-on from the front: the side photos do the ears
    else:
        weight *= 1 - parts["face_tex"]   # no second set of brows, lips... from a less exact fit

    # Eyes: the fitted lids never sit exactly where hers do, so without this her irises get
    # painted onto the lids. Only a photo with her eyes outlined (the front one) does the eyes.
    ball_t = covered & eyeball[tf]
    if eye_polys is None:
        weight[ball_t | (covered & eye_area[tf])] = 0
    else:
        import cv2
        open_eye = np.zeros((HI_RES, HI_RES), np.uint8)
        cv2.fillPoly(open_eye, [np.round(p).astype(np.int32) for p in eye_polys], 1)
        open_eye = open_eye.astype(bool)
        # skin never takes what's inside her eye outline...
        weight[covered & ~eyeball[tf] & ndimage.binary_dilation(open_eye, iterations=3)[iy, ix]] = 0
        # ...and the eyeballs take only that. They sit behind the fitted lids, so they are
        # depth-tested against each other only.
        zeye, _ = rasterize(px, z, faces[eyeball], HI_RES, HI_RES)
        ok = (ball_t & in_img & (tz >= zeye[iy, ix] - 0.004)
              & ndimage.binary_erosion(open_eye, iterations=2)[iy, ix])
        weight[ball_t] = np.where(ok, np.maximum(facing, 0.05), 0)[ball_t]
    colour = bilinear(hi, tp[..., 0].ravel(), tp[..., 1].ravel()).reshape(TEX_SIZE, TEX_SIZE, 3)

    seen = weight > 0
    tex = fill_unseen(np.where(seen[..., None], colour, 0), seen, covered)
    write_obj(out / f"{name}.obj", v, faces, vt, ft, tex)

    # preview: the fitted mesh shaded over the crop
    shade = np.zeros((HI_RES, HI_RES), np.float32)
    hit = fid >= 0
    shade[hit] = 0.25 + 0.75 * np.clip(fn[fid[hit], 2], 0, 1)
    overlay = hi.copy()
    overlay[hit] = 0.35 * hi[hit] + 0.65 * shade[hit, None]
    try:
        Image.fromarray((np.concatenate([hi, overlay], 1) * 255).astype(np.uint8)).save(out / f"{name}_fit.png")
    except OSError as e:            # e.g. open in an image viewer - only a preview, so carry on
        print(f"  couldn't save {name}_fit.png ({e.strerror}) - is it open somewhere?")
    print(f"[{name}] {seen.sum() / covered.sum():.0%} of the head seen -> {name}.obj")
    return colour, weight


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    images = ROOT / "images" / "character_design"
    ap.add_argument("--images", nargs="+", default=[str(images / f"{n}.jpeg")
                                                     for n in ("bald_front", "bald", "bald_back")])
    ap.add_argument("--assets", default=str(HERE / "assets"))
    ap.add_argument("--model", default="EMOCA_v2_lr_mse_20")
    ap.add_argument("--out", default=str(HERE / "output"))
    # Face detectors miss full side views, so those get a hand-placed square: name=centre_x,centre_y,size
    ap.add_argument("--crop", nargs="*", default=["bald=1450,780,1450", "bald2=1150,880,1250",
                                                  "dream_profile=1500,880,1350"],
                    help="manual crop per image, e.g. dream_profile=1500,880,1350")
    ap.add_argument("--back", nargs="*", default=["bald_back"],
                    help="images that show the back of the head (fitted by silhouette, not by face)")
    # One well-fitted profile, flipped, covers the other side better than a second, worse-fitting profile
    ap.add_argument("--mirror", nargs="*", default=["bald"],
                    help="side views to also use flipped, for the other side of the head")
    ap.add_argument("--eye-texture", default=str(images / "eye.jpeg"),
                    help="front-on eyeball picture wrapped onto the eyeballs ('' to use the photos)")
    args = ap.parse_args()
    manual = {}
    for item in args.crop:
        stem, nums = item.split("=")
        manual[stem] = [float(x) for x in nums.split(",")]
    photos = []
    for path in args.images:
        name = Path(path).stem
        img = np.array(Image.open(path).convert("RGB"))
        photos.append((name, img))
        if name in args.mirror:
            photos.append((name + "_mirror", img[:, ::-1].copy()))
            if name in manual:
                cx, cy, s = manual[name]
                manual[name + "_mirror"] = [img.shape[1] - 1 - cx, cy, s]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assets, out = Path(args.assets), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    e_flame, e_exp, n, jaw, glob = load_model(assets, args.model, device)
    flame = load_flame(assets, n, device)
    faces = flame.faces_tensor.cpu().numpy()
    vt, ft = load_uv_template(assets)
    print(f"FLAME: {flame.v_template.shape[0]} vertices, {len(faces)} faces, {len(vt)} uv coords")

    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, face_detector="sfd")

    print("preparing the UV layout...")
    tfid, tbary = uv_raster(vt, ft, TEX_SIZE)
    covered = tfid >= 0
    uv = (np.where(covered, tfid, 0), tbary, covered)
    regions = head_parts(assets)

    def crops(img, src):
        hi, _ = crop(img, src, HI_RES)
        # where the crop runs past the photo's edge it is black padding, not her
        inside, _ = crop(np.full(img.shape[:2], 255.0), src, HI_RES)
        inside = inside > 0.99
        # on a black studio background, also skip the background wherever the mesh pokes past her outline
        mask = studio_mask(img)
        if mask is not None:
            fg, _ = crop(mask * 255.0, src, HI_RES)
            inside &= ndimage.binary_erosion(fg > 0.5, iterations=4)
        return hi, inside

    views, backs = [], []
    for name, img in photos:
        if name in args.back:
            backs.append((name, img))      # needs the face shape, so after the face views
            continue
        if name in manual:
            src = square(*manual[name])
            box = f"manual crop {manual[name][2]:.0f}px"
        else:
            det = detect_crop(fa, img)
            if det is None:
                print(f"[{name}] no face found - skipped (try --crop {name}=x,y,size, or --back {name})")
                continue
            src, kpt = det
            box = f"face {np.ptp(kpt[:, 0]):.0f}x{np.ptp(kpt[:, 1]):.0f}px"
        print(f"[{name}] {box}")
        small, _ = crop(img, src, CROP)
        centre = (src[1] + src[2]) / 2
        src_tex = centre + TEX_CROP * (src - centre)
        hi, inside = crops(img, src_tex)
        eye_polys = None
        if name not in manual:
            # her eye outlines (FAN landmarks 36-41, 42-47) in the texture crop
            tform = estimate_transform("similarity", src_tex,
                                       np.array([[0, 0], [0, HI_RES - 1], [HI_RES - 1, 0]]))
            eye_polys = [tform(kpt[36:42, :2]), tform(kpt[42:48, :2])]

        with torch.no_grad():
            x = torch.from_numpy(small.transpose(2, 0, 1))[None].to(device)
            code = e_flame(x)
            exp_code = e_exp(x)
            parts = torch.split(code, [n["n_shape"], n["n_tex"], n["n_exp"], n["n_pose"], n["n_cam"], n["n_light"]], 1)
            shape, _, exp, pose, cam, _ = parts
            exp_new = exp_code[:, :n["n_exp"]]
            extra = exp_code[:, n["n_exp"]:]
            if jaw and glob:
                pose = extra
            elif glob:
                pose = torch.cat([extra, pose[:, 3:]], 1)
            elif jaw:
                pose = torch.cat([pose[:, :3], extra], 1)
            exp = exp_new
            verts, _, _ = flame(shape, exp, pose)

        v = verts[0].cpu().numpy()
        c = cam[0].cpu().numpy()
        mask = studio_mask(img)
        if name in manual and mask is not None:
            # EMOCA's angle for a full profile is only roughly right, which puts the ears in the
            # wrong place on the texture: fine-tune head angle and camera against her silhouette
            target = crop(mask * 255.0, src, 256)[0] > 0.5

            def verts_of(rot, shape=shape, exp=exp, jaw_pose=pose[:, 3:]):
                r = torch.tensor(rot, dtype=pose.dtype, device=device)[None]
                with torch.no_grad():
                    return flame(shape, exp, torch.cat([r, jaw_pose], 1))[0][0].cpu().numpy()

            x0 = np.concatenate([pose[0, :3].cpu().numpy(), c])
            x, before, after = fit_silhouette(target, verts_of, x0, [0.05, 0.05, 0.05, 0.05 * c[0], 0.02, 0.02],
                                              faces, keep_frac=0.35)
            print(f"  silhouette match {before:.0%} -> {after:.0%}")
            if after > before:
                v, c = verts_of(x[:3]), x[3:]
                pose[0, :3] = torch.tensor(x[:3], dtype=pose.dtype, device=device)
                cam = torch.tensor(c, dtype=cam.dtype, device=device)[None]
        # cameras from here on are in the wider texture crop (which holds the whole head)
        c_tex = np.array([c[0] / TEX_CROP, c[1], c[2]])
        views.append(dict(name=name, kind="side" if name in manual else "front", img=img, mask=mask, src=src_tex,
                          kpt=None if name in manual else kpt, shape=shape, exp=exp, pose=pose, cam=c_tex,
                          v=v, hi=hi, inside=inside, eye_polys=eye_polys))

    if not views:
        sys.exit("No faces found in any image.")

    shape = torch.stack([w["shape"] for w in views]).mean(0)
    zero_exp = torch.zeros(1, n["n_exp"], device=device)
    for name, img in backs:
        print(f"[{name}] back view")
        mask = studio_mask(img)
        if mask is None:
            sys.exit("back views need a plain black background to find her outline")

        def verts_of(rot):
            with torch.no_grad():
                r = torch.tensor(rot, dtype=torch.float32, device=device)[None]
                return flame(shape, zero_exp, torch.cat([r, torch.zeros_like(r)], 1))[0][0].cpu().numpy()

        rot0 = np.array([0, np.pi, 0])                # turned round 180 degrees
        src, c0 = back_start(mask, verts_of(rot0))
        target = crop(mask * 255.0, src, 256)[0] > 0.5
        x, _, after = fit_silhouette(target, verts_of, np.concatenate([rot0, c0]),
                                     [0.05, 0.05, 0.05, 0.05 * c0[0], 0.02, 0.02], faces, keep_frac=0.45)
        print(f"  silhouette match {after:.0%}")
        hi, inside = crops(img, src)
        pose = torch.cat([torch.tensor(x[:3], dtype=torch.float32, device=device)[None],
                          torch.zeros(1, 3, device=device)], 1)
        views.append(dict(name=name, kind="back", img=img, mask=mask, src=src, kpt=None, shape=shape,
                          exp=zero_exp, pose=pose, cam=x[3:], v=verts_of(x[:3]), hi=hi, inside=inside,
                          eye_polys=None))

    # EMOCA guessed each photo on its own. Now fit one shape to all of them together (face
    # landmarks, her ears, her outline in every photo), so they agree - and her ears end up where
    # hers are. Needs the black studio background for the outlines.
    fix = np.zeros((len(flame.v_template), 3), np.float32)       # per-vertex detail offsets
    if all(w["mask"] is not None for w in views):
        print("fitting one head to all the photos...")
        shape, fix, fitted = multiview_fit.fit(flame, views, faces, flame.v_template.cpu().numpy(),
                                               regions["eyeball"], device)
        for w, f in zip(views, fitted):
            w["v"] = f["v"]
            w["pose"], w["exp"] = f["pose"], f["exp"]
            w["cam"] = f["cam"]
    for w in views:
        np.savez(out / f"{w['name']}_flame.npz", shape=shape.cpu().numpy(), exp=w["exp"].cpu().numpy(),
                 pose=w["pose"].cpu().numpy(), cam=np.asarray(w["cam"])[None])
        w["colour"], w["weight"] = bake_view(w["name"], w["hi"], w["inside"], w["v"], w["cam"], faces, vt, ft,
                                             uv, out, dict(regions, side_view=w["kind"] == "side"),
                                             w["eye_polys"])

    # The photos are lit differently. Bring each one's colours to the blend of those before it
    # (front first) where they overlap - smoothly, so skin tone matches across the seams.
    acc = views[0]["colour"] * views[0]["weight"][..., None]
    wacc = views[0]["weight"].copy()
    for w in views[1:]:
        if w["kind"] == "back":     # the back photo overlaps the sides only where they are in shadow
            acc += w["colour"] * w["weight"][..., None]
            wacc += w["weight"]
            continue
        ref = acc / np.maximum(wacc, 1e-9)[..., None]
        w["colour"] = match_colour(w["colour"], w["weight"], ref, wacc, uv[2])
        acc += w["colour"] * w["weight"][..., None]
        wacc += w["weight"]

    # combined neutral head: mean shape, no expression, no pose, texture blended by how well each view saw it
    covered = uv[2]
    with torch.no_grad():
        v, _, _ = flame(shape, zero_exp, torch.zeros(1, 6, device=device))
    v = v[0].cpu().numpy() + fix
    wsum = sum(w["weight"] for w in views)
    # which photo textured what: red the front, green the sides, blue the back
    share = np.zeros(wsum.shape + (3,), np.float32)
    for w in views:
        share[..., {"front": 0, "side": 1, "back": 2}[w["kind"]]] += w["weight"]
    share /= np.maximum(wsum, 1e-9)[..., None]
    Image.fromarray((share * 255).astype(np.uint8)).save(out / "texture_sources.png")
    tex = sum(w["colour"] * w["weight"][..., None] for w in views) / np.maximum(wsum, 1e-9)[..., None]
    seen = wsum > 0
    tex = even_tone(tex, seen, covered, regions["face_tex"], covered & (regions["ear_side"][uv[0]] != 0)
                    | covered & regions["eyeball"][uv[0]])
    seen_frac = seen.sum() / covered.sum()
    # what her lids hide of the eyeballs gets the white of her eyes, not a smear of skin
    ball_t = covered & regions["eyeball"][uv[0]]
    got = ball_t & seen
    sclera = np.array([0.85, 0.82, 0.80])
    if got.sum() > 50:
        lum = tex[got].mean(1)
        sclera = tex[got][lum >= np.percentile(lum, 80)].mean(0)
    tex[ball_t & ~seen] = sclera
    if args.eye_texture and Path(args.eye_texture).exists():
        eye_img = np.asarray(Image.open(args.eye_texture).convert("RGB"), np.float32) / 255
        front = next((w for w in views if w["kind"] == "front"), None)
        if front is not None:
            irises = [measure_iris(front["hi"], poly) for poly in front["eye_polys"]]
            # her two irises are the same size: trust the smaller reading (more of it is lash-free)
            r = min(i[1] for i in irises)
            irises = [(c, r) for c, _ in irises]
            eye_img = her_eye_picture(eye_img, front["hi"], front["eye_polys"], irises)
        # FLAME's eyeballs sit a little to the outside of her eye openings, so looking straight
        # ahead they'd peer inward. Slide each one (sideways and up/down only) to sit right
        # behind the middle of its opening: the centre of her eyelid landmarks (36-41, 42-47).
        idx = flame.full_lmk_faces_idx.reshape(-1).cpu().numpy()
        bary = flame.full_lmk_bary_coords.reshape(-1, 3).cpu().numpy()
        lmk3d = (v[faces[idx]] * bary[..., None]).sum(1)
        ball_v = np.unique(faces[regions["eyeball"]])
        for opening in (lmk3d[36:42].mean(0), lmk3d[42:48].mean(0)):
            mine = ball_v[(v[ball_v, 0] > 0) == (opening[0] > 0)]
            shift = opening - v[mine].mean(0)
            shift[2] = 0
            v[mine] += shift
            fix[mine] += shift
        aims = iris_r = None
        if front is not None:
            aims, iris_r = [], []
            for poly, iris, k in zip(front["eye_polys"], irises, (36, 42)):
                lids = lmk3d[k:k + 6]
                aims.append(iris_aim(poly, lids, iris))
                # her iris's size against her eye's width, on this head's eye
                iris_r.append(iris[1] / np.linalg.norm(poly[3] - poly[0]) * np.linalg.norm(lids[3] - lids[0]))
            print("iris: " + ", ".join(f"{r * 2000:.1f} mm across" for r in iris_r))
        tex[ball_t] = wrap_eyeball(eye_img, v, faces, regions["eyeball"], uv, ball_t, aims, iris_r)
    seen |= ball_t
    tex = fill_unseen(np.where(seen[..., None], tex, 0), seen, covered)
    write_obj(out / "dream_head.obj", v, faces, vt, ft, tex)
    np.savez(out / "dream_head_flame.npz", shape=shape.cpu().numpy(), fix=fix)
    print(f"combined: {seen_frac:.0%} of the head seen -> dream_head.obj")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
