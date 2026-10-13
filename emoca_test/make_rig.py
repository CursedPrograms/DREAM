"""
EMOCA test: DREAM's FLAME head -> rigged head for Blender (.blend) and game engines (.fbx).

emoca_to_obj.py only writes static OBJs. FLAME itself is rigged, so this rebuilds her head
from the FLAME model with that rig intact:
  - expression blendshapes as shape keys (exp_00..exp_49, EMOCA's 50), plus
    photo_expression: the expression she has in the front photo
  - an armature with FLAME's joints (root, neck, jaw, eye.L, eye.R), skin-weighted
  - the textured material from dream_head.png, and a glossy one for the eyeballs

Run with DREAM's venv311 (Blender 4.x must be installed; it runs in the background):
    emoca_test\\make_rig.bat
    venv311\\Scripts\\python.exe emoca_test\\make_rig.py
    venv311\\Scripts\\python.exe emoca_test\\make_rig.py --fit --images images\\dream.jpg
--fit first reruns emoca_to_obj.py (any --images / --crop go to it), then builds the rig.

FLAME / EMOCA files are under a non-commercial licence: the outputs are too.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# FLAME joint order (kintree_table): 0 global, 1 neck, 2 jaw, 3/4 eyes
JOINT_NAMES = ["root", "neck", "jaw", "eye", "eye"]


def find_blender(hint):
    if hint:
        return hint
    found = shutil.which("blender")
    if found:
        return found
    exes = sorted(glob.glob(r"C:\Program Files\Blender Foundation\*\blender.exe"))
    if exes:
        return exes[-1]
    sys.exit("Blender not found - install Blender 4.x or pass --blender path\\to\\blender.exe")


def to_blender(p):
    """FLAME is Y-up with the face looking down +Z; Blender is Z-up with the front view looking down +Y."""
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def build_rig_data(assets, head_npz, photo_npz, n_exp):
    from emoca_to_obj import load_pickle, load_uv_template

    m = load_pickle(assets / "FLAME" / "geometry" / "generic_model.pkl")
    shapedirs = np.asarray(m["shapedirs"], np.float64)
    shape = np.load(head_npz)["shape"][0]
    n_shape = len(shape)

    # her neutral face: template + shape blendshapes (FLAME keeps 300 shape, then 100 expression)
    v = np.asarray(m["v_template"]) + shapedirs[:, :, :n_shape] @ shape
    head = np.load(head_npz)
    if "fix" in head:           # the fit's per-vertex detail offsets (what FLAME's sliders can't reach)
        v = v + head["fix"]
    exp_dirs = shapedirs[:, :, 300:300 + n_exp]                      # (V, 3, n_exp)
    jreg = m["J_regressor"]
    joints = np.asarray(jreg @ v)                                      # (5, 3), from her shape
    parents = np.asarray(m["kintree_table"])[0].astype(np.int64)
    parents[0] = -1

    names = []
    for i, base in enumerate(JOINT_NAMES):
        names.append(base if base != "eye" else ("eye.L" if joints[i, 0] > 0 else "eye.R"))

    photo = np.zeros(3 * len(v))
    if photo_npz and Path(photo_npz).exists():
        exp = np.load(photo_npz)["exp"][0][:n_exp]
        photo = (exp_dirs @ exp)
        photo = to_blender(photo)

    vt, ft = load_uv_template(assets)
    faces = np.asarray(m["f"]).astype(np.int64)
    eyeball = np.asarray(m["weights"])[:, 3:].max(1) > 0.5          # skinned to the eye joints
    return dict(
        verts=to_blender(v).astype(np.float32),
        faces=faces,
        eye_faces=eyeball[faces].all(1),
        exp_deltas=to_blender(exp_dirs.transpose(2, 0, 1)).astype(np.float32),   # (n_exp, V, 3)
        photo_delta=np.asarray(photo, np.float32).reshape(len(v), 3),
        joints=to_blender(joints).astype(np.float32),
        parents=parents,
        joint_names=np.array(names),
        weights=np.asarray(m["weights"], np.float32),
        vt=vt, ft=ft,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true", help="rerun emoca_to_obj.py first (extra args go to it)")
    ap.add_argument("--assets", default=str(HERE / "assets"))
    ap.add_argument("--head", default=str(OUT / "dream_head_flame.npz"), help="face shape to rig")
    ap.add_argument("--photo", default=str(OUT / "bald_front_flame.npz"), help="source of the photo_expression key")
    ap.add_argument("--texture", default=str(OUT / "dream_head.png"))
    ap.add_argument("--out", default=str(OUT / "dream_rig"), help="writes <out>.blend and <out>.fbx")
    ap.add_argument("--n-exp", type=int, default=50)
    ap.add_argument("--neck", type=float, default=5.0,
                    help="cm of neck kept below the chin (FLAME's runs ~10 cm, almost to the shoulders)")
    ap.add_argument("--blender", help="path to blender.exe (found automatically otherwise)")
    args, fit_args = ap.parse_known_args()

    if args.fit:
        subprocess.run([sys.executable, str(HERE / "emoca_to_obj.py"), *fit_args], check=True)
    elif fit_args:
        sys.exit(f"unknown arguments {fit_args} (they only apply with --fit)")

    for p in (args.head, args.texture):
        if not Path(p).exists():
            sys.exit(f"missing {p} - run with --fit (or emoca_to_obj.py) first")

    blender = find_blender(args.blender)
    sys.argv = sys.argv[:1]     # emoca_to_obj reads nothing from argv on import, but keep it clean
    data = build_rig_data(Path(args.assets), args.head, args.photo, args.n_exp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_path = out.with_name(out.name + "_data.npz")
    # her chin is FLAME landmark 8; Blender trims the neck this far below it
    emb = np.load(Path(args.assets) / "FLAME" / "geometry" / "landmark_embedding.npy",
                  allow_pickle=True, encoding="latin1").item()
    tri = data["faces"][np.asarray(emb["full_lmk_faces_idx"]).reshape(-1)[8]]
    chin = (data["verts"][tri] * np.asarray(emb["full_lmk_bary_coords"]).reshape(-1, 3)[8][:, None]).sum(0)
    data["neck_cut"] = np.float32(chin[2] - args.neck / 100)
    np.savez(data_path, **data)
    print(f"rig data: {len(data['verts'])} vertices, {len(data['exp_deltas'])} expressions, "
          f"joints {', '.join(data['joint_names'])}")

    print(f"building in Blender: {blender}")
    subprocess.run([blender, "-b", "--factory-startup", "--python", str(HERE / "blender_build_rig.py"), "--",
                    str(data_path), str(Path(args.texture).resolve()),
                    str(out.with_suffix(".blend")), str(out.with_suffix(".fbx"))], check=True)
    os.unlink(data_path)

    # the four preview renders, side by side
    from PIL import Image
    shots = [out.with_name(f"{out.name}_view{i}.png") for i in range(4)]
    strip = Image.new("RGB", (1600, 400), "white")
    for i, shot in enumerate(shots):
        im = Image.open(shot).convert("RGB")
        w = im.width
        strip.paste(im.crop((w // 10, w // 10, w - w // 10, w - w // 10)).resize((400, 400)), (i * 400, 0))
        im.close()
        os.unlink(shot)
    views = out.with_name(out.name + "_views.png")
    try:
        strip.save(views, optimize=True)
    except OSError:             # e.g. open in an image viewer - save next to it instead
        views = out.with_name(out.name + "_views_new.png")
        strip.save(views, optimize=True)
        print(f"couldn't overwrite {out.name}_views.png (open somewhere?) - saved {views.name}")
    print(f"done: {out.with_suffix('.blend')}\n      {out.with_suffix('.fbx')}")


if __name__ == "__main__":
    main()
