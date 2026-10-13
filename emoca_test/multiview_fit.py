"""
One FLAME shape for all the photos at once.

EMOCA guesses a face shape from each photo on its own, and averaging those can't make the
photos agree: her ears ended up in the wrong place in every view. Here a single shared shape,
plus each photo's head angle, jaw, expression and camera, is optimised so that the head matches
every photo together:

  - the front photo's 68 face landmarks (FAN)
  - her ear: 20 landmarks per visible ear in the side photos (ear_landmarks.py)
  - her outline in every photo (front, sides and back): the mesh's silhouette edge is pulled onto
    the photo's, and the photo's onto the mesh's, using distance maps (no renderer needed)

with mild priors keeping the shape and expressions human. EMOCA's guesses are the start. The
front photo carries her likeness, so it weighs most and the shape stays close to its reading.

FLAME's 100 shape sliders can't reach every detail of her (her eyes are narrower than any
nearby FLAME face), so a last pass lets every vertex move a few millimetres on its own, kept
smooth, to land on her landmarks, ears and outline. Those offsets go into the rig too.
"""

import numpy as np
import torch
from scipy import ndimage

import ear_landmarks

FIT_RES = 512               # resolution of the outline maps
NECK_Y = -0.10              # FLAME template height below which the mesh is neck (cut off; not her outline)
EYE_SETBACK = (0.0, 0.0, -0.0015)  # metres: eyeballs moved back behind her lids
DETAIL_SMOOTH = 30.0        # detail pass: how strongly neighbouring vertices must move together
DETAIL_SIZE = 0.3           # ...and how strongly offsets are kept small
LMK_WEIGHT = 25             # face landmarks against outlines: her features matter more than her outline
FRONT_WEIGHT = 10           # the front photo counts this much more than each other photo
REACH = 0.1                 # outline points further apart than this (crop is 2 wide) are different
                            # features (her neck, shoulders), not a misfit, and don't pull


def _norm(points, src):
    """Image pixels -> the view crop's normalised coords (-1..1), like DECA's camera."""
    side = src[2, 0] - src[0, 0]
    return (points - src[0]) / side * 2 - 1


def head_crop(mask):
    """A square around her whole head (the ear detector needs the head around the ear)."""
    rows = np.nonzero(mask.any(1))[0]
    top = rows[0]
    upper = mask[top:top + int(0.5 * (rows[-1] - top))]
    xs = np.nonzero(upper.any(0))[0]
    width = xs[-1] - xs[0]
    return (xs[0] + xs[-1]) / 2, top + 0.5 * width, width


def find_ear(model, img, mask, device):
    """20 ear landmarks in image pixels and their confidences: the head-sized crop (of a few
    tried) the detector is surest about."""
    from emoca_to_obj import crop, square
    cx, cy, w = head_crop(mask)
    best = None
    for k in (0.9, 1.0, 1.15):
        src = square(cx, cy, k * w)
        c, _ = crop(img, src, 512)
        pts, conf = ear_landmarks.detect(model, c, device)
        if best is None or conf.mean() > best[1].mean():
            side = src[2, 0] - src[0, 0]
            best = (src[0] + pts / 511 * side, conf)
    return best


class Outline:
    """Distance maps for one photo's outline, in the view crop's normalised coords."""

    def __init__(self, mask, src, device):
        from emoca_to_obj import crop
        m = crop(mask * 255.0, src, FIT_RES)[0] > 0.5
        valid = crop(np.full(mask.shape, 255.0), src, FIT_RES)[0] > 0.99
        valid[[0, -1], :] = valid[:, [0, -1]] = False
        valid = ndimage.binary_erosion(valid, iterations=3)
        # where the head runs off the photo or the crop, that edge isn't hers
        edge = m ^ ndimage.binary_erosion(m, border_value=1)
        edge &= valid
        dist = ndimage.distance_transform_edt(~edge) / (FIT_RES / 2)
        self.dist = torch.tensor(dist, dtype=torch.float32, device=device)[None, None]
        self.valid = torch.tensor(valid, dtype=torch.float32, device=device)[None, None]
        ys, xs = np.nonzero(edge)
        self.edge_pts = torch.tensor(np.stack([xs, ys], 1) / (FIT_RES / 2) - 1, dtype=torch.float32, device=device)

    def sample(self, img, p):
        """Bilinear lookup of a map at normalised points p [N, 2]."""
        g = p.view(1, 1, -1, 2)
        return torch.nn.functional.grid_sample(img, g, align_corners=False, padding_mode="border").view(-1)


def _rodrigues(r):
    """Axis-angle [B, 3] -> rotation matrices [B, 3, 3]."""
    angle = r.norm(dim=1, keepdim=True).clamp(min=1e-8)
    k = r / angle
    K = torch.zeros(len(r), 3, 3, device=r.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(1, 2)
    a = angle[..., None]
    return torch.eye(3, device=r.device) + torch.sin(a) * K + (1 - torch.cos(a)) * K @ K


def _laplacian(faces, device):
    """Uniform mesh Laplacian (each vertex minus the mean of its neighbours), sparse [V, V]."""
    import scipy.sparse as sp
    n = faces.max() + 1
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adj = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    adj.data[:] = 1
    deg = np.asarray(adj.sum(1)).ravel()
    lap = (sp.eye(n) - sp.diags(1 / deg) @ adj).tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.stack([lap.row, lap.col])), torch.tensor(lap.data, dtype=torch.float32), (n, n),
                                   device=device).coalesce()


def _edges(faces):
    """Mesh edges shared by exactly two faces, and those faces."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    f = np.tile(np.arange(len(faces)), 3)
    e.sort(1)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, f = e[order], f[order]
    same = (e[1:] == e[:-1]).all(1)
    i = np.nonzero(same)[0]
    return e[i], np.stack([f[i], f[i + 1]], 1)


def fit(flame, views, faces, template, eyeball_faces, device, iters=(150, 500, 400)):
    """views: list of dicts with img, mask, src (the crop the camera is in), kind ('front'|'side'|'back'),
    shape, exp, pose (torch [1, ...] from EMOCA / the silhouette fit), cam (numpy s, tx, ty),
    and kpt (FAN 68 landmarks in image pixels) for the front. Returns the shared shape, the detail
    offsets (per vertex, in FLAME's unposed space) and, per view, the fitted vertices (offsets
    included), camera and pose/exp."""
    B = len(views)
    ftorch = torch.tensor(faces, device=device)
    edges, edge_faces = _edges(faces)
    edges, edge_faces = torch.tensor(edges, device=device), torch.tensor(edge_faces, device=device)
    # vertices whose outline counts: not neck (cut off), not eyeballs (inside the lids)
    eye_v = np.zeros(len(template), bool)
    eye_v[np.unique(faces[eyeball_faces])] = True
    usable = torch.tensor((template[:, 1] > NECK_Y) & ~eye_v, device=device)

    model = ear_landmarks.load(device)
    targets = []
    for w in views:
        t = dict(outline=Outline(w["mask"], w["src"], device))
        if w["kind"] == "front":
            t["lmk"] = torch.tensor(_norm(w["kpt"][:, :2], w["src"]), dtype=torch.float32, device=device)
        if w["kind"] == "side" and model is not None:
            pts, conf = find_ear(model, w["img"], w["mask"], device)
            t["ear"] = torch.tensor(_norm(pts, w["src"]), dtype=torch.float32, device=device)
            t["ear_conf"] = torch.tensor(conf, dtype=torch.float32, device=device)
        targets.append(t)

    # Her likeness comes from the front photo: start from its face shape and stay close to it.
    # The other photos still place her ears and the back of her skull, but only nudge the face.
    front = [w for w in views if w["kind"] == "front"]
    shape0 = (front[0] if front else views[0])["shape"].detach().clone()
    shape = shape0.clone().requires_grad_(True)
    rot = torch.cat([w["pose"][:, :3] for w in views]).clone().requires_grad_(True)
    jaw = torch.cat([w["pose"][:, 3:] for w in views]).clone().requires_grad_(True)
    exp = torch.cat([w["exp"] for w in views]).clone().requires_grad_(True)
    cam = torch.tensor(np.stack([w["cam"] for w in views]), dtype=torch.float32, device=device).requires_grad_(True)
    is_back = torch.tensor([w["kind"] == "back" for w in views], device=device)

    # detail offsets, in FLAME's own (unposed) space. Each eyeball moves as one piece, by the
    # average offset of the lids around it, so the iris stays centred in her eye opening.
    offsets = torch.zeros(len(template), 3, device=device, requires_grad=True)
    movable = torch.tensor(~eye_v, dtype=torch.float32, device=device)[:, None]
    eyes = []
    for side in (template[:, 0] > 0, template[:, 0] <= 0):
        ball = eye_v & side
        centre = template[ball].mean(0)
        lids = ~eye_v & (np.linalg.norm(template - centre, axis=1) < 0.02)
        eyes.append((torch.tensor(np.nonzero(ball)[0], device=device),
                     torch.tensor(np.nonzero(lids)[0], device=device)))

    def detail():
        off = offsets * movable
        for ball, lids in eyes:
            # ...set back a little, so lids pulled in to her eye shape still cover it all round
            off = off.index_put((ball,), off[lids].mean(0) + off.new_tensor(EYE_SETBACK))
        return off
    lmk_faces = ftorch[flame.full_lmk_faces_idx.reshape(-1)]                  # [68, 3]
    lmk_bary = flame.full_lmk_bary_coords.reshape(-1, 3)[..., None]          # [68, 3, 1]
    lap = _laplacian(faces, device)

    def forward():
        e = torch.where(is_back[:, None], torch.zeros_like(exp), exp)
        j = torch.where(is_back[:, None], torch.zeros_like(jaw), jaw)
        v, lmk, _ = flame(shape.expand(B, -1), e, torch.cat([rot, j], 1))
        # offsets turn with the head (the face is rigid with it, apart from the jaw's small swing)
        off = detail() @ _rodrigues(rot).transpose(1, 2)                         # [B, V, 3]
        lmk = lmk + (off[:, lmk_faces] * lmk_bary).sum(2)
        return v + off, lmk

    def project(x, b):
        s, tx, ty = cam[b, 0], cam[b, 1], cam[b, 2]
        return torch.stack([(x[..., 0] + tx) * s, -(x[..., 1] + ty) * s], -1)

    # which ear each side photo shows: the one nearer the camera at the start
    with torch.no_grad():
        v0, _ = forward()
    for b, t in enumerate(targets):
        if "ear" not in t:
            continue
        left = v0[b, ear_landmarks.LEFT_EAR, 2].mean() > v0[b, ear_landmarks.RIGHT_EAR, 2].mean()
        t["ear_idx"] = ear_landmarks.LEFT_EAR if left else ear_landmarks.RIGHT_EAR
        # drop detections far from where the mesh's ear starts out (strays onto the face)
        start = project(v0[b, t["ear_idx"]], b)
        dist = (start - t["ear"]).norm(dim=1)
        keep = (t["ear_conf"] > 0.12) & (dist < 3 * dist.median())
        t["ear_w"] = keep.float() * t["ear_conf"]
        print(f"  {views[b]['name']}: {'left' if left else 'right'} ear, {int(keep.sum())}/20 ear points used")

    def outline_loss(v, b, t):
        with torch.no_grad():
            tri = v[b][ftorch]
            nz = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)[:, 2]
            rim = (nz[edge_faces[:, 0]] > 0) != (nz[edge_faces[:, 1]] > 0)
            idx = torch.unique(edges[rim].flatten())
            idx = idx[usable[idx]]
        p = project(v[b, idx], b)
        ok = t["outline"].sample(t["outline"].valid, p) > 0.5
        p = p[ok]
        if len(p) < 10:
            return v.sum() * 0
        to_photo = t["outline"].sample(t["outline"].dist, p).clamp(max=REACH).pow(2).mean()
        # and the photo's outline onto the mesh's, above the mesh's neck cut
        with torch.no_grad():
            pts = t["outline"].edge_pts
            pts = pts[pts[:, 1] < p[:, 1].max()]
            if len(pts) > 600:
                pts = pts[torch.randperm(len(pts), device=pts.device)[:600]]
        d = torch.cdist(pts, p).min(1)[0] if len(pts) else p.new_zeros(0)
        to_mesh = d[d < REACH].pow(2).mean() if (d < REACH).any() else 0
        return to_photo + to_mesh

    exp0, jaw0 = exp.detach().clone(), jaw.detach().clone()

    def losses(v, lmk):
        out = {"outline": 0, "lmk": 0, "ear": 0}
        for b, t in enumerate(targets):
            out["outline"] = out["outline"] + outline_loss(v, b, t) * (FRONT_WEIGHT if "lmk" in t else 1)
            if "lmk" in t:
                d = (project(lmk[b, :, :2], b) - t["lmk"]).pow(2).sum(1)
                wts = torch.ones(68, device=device)
                wts[:17] = 0.75                                     # jaw line: FAN is less sure there,
                wts[6:11] = 3.0                                     # but her chin is sharper than FLAME's
                wts[36:48] = 3.0                                    # and her lids set how open her eyes look
                out["lmk"] = out["lmk"] + FRONT_WEIGHT * (d * wts).sum() / wts.sum()
            if "ear_idx" in t:
                d = (project(v[b, t["ear_idx"]], b) - t["ear"]).pow(2).sum(1)
                out["ear"] = out["ear"] + (d * t["ear_w"]).sum() / t["ear_w"].sum().clamp(min=1e-6)
        # shape stays human; jaw and expression stay near EMOCA's reading unless the photos insist
        m = offsets * movable
        out["detail"] = DETAIL_SMOOTH * torch.sparse.mm(lap, m).pow(2).sum() + DETAIL_SIZE * m.pow(2).sum()
        out["prior"] = (2e-6 * shape.pow(2).sum() + 8e-6 * (shape - shape0).pow(2).sum()
                        + 2e-5 * (exp - exp0).pow(2).sum()
                        + 0.3 * (jaw - jaw0).pow(2).sum())
        return out

    def run(params, n, label):
        opt = torch.optim.Adam(params, lr=3e-3)
        for i in range(n):
            v, lmk = forward()
            parts = losses(v, lmk)
            loss = (parts["outline"] + LMK_WEIGHT * parts["lmk"] + 2 * parts["ear"] + parts["prior"]
                    + parts["detail"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i in (0, n - 1):
                print(f"  {label} {i:3d}: " + ", ".join(f"{k} {float(x):.5f}" for k, x in parts.items()))

    run([rot, cam], iters[0], "rigid")
    run([shape, rot, jaw, exp, cam], iters[1], "full ")
    run([offsets, exp, cam], iters[2], "detail")

    with torch.no_grad():
        v, _ = forward()
        fix = detail().cpu().numpy()
    print(f"  detail offsets: mean {np.linalg.norm(fix, axis=1).mean() * 1000:.1f} mm, "
          f"max {np.linalg.norm(fix, axis=1).max() * 1000:.1f} mm")
    return shape.detach(), fix, [dict(v=v[b].cpu().numpy(), cam=cam[b].detach().cpu().numpy(),
                                 pose=torch.cat([rot, jaw], 1)[b:b + 1].detach(),
                                 exp=exp[b:b + 1].detach()) for b in range(B)]
