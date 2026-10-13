# EMOCA head: photos to a rigged 3D head

This turns a few photos of a person into a textured, rigged 3D head. You get a Blender `.blend`
and an `.fbx` with FLAME's jaw, neck and eye bones and 50 expression shape keys. It was built for
DREAM, but works for any new character.

It's a separate test program, not part of DREAM itself. EMOCA, FLAME and the ear landmarker are
under a **non-commercial research licence**, and so is everything this makes. `assets/` stays
out of git.

## Setup (once)

1. Run `download_weights.bat` from the repo root and answer **Y** at step 5 (EMOCA). That puts
   EMOCA, FLAME and the ear landmarker in `emoca_test\assets\`.
2. Install Blender 4.x (it's found under `C:\Program Files\Blender Foundation\`; otherwise pass
   `--blender path\to\blender.exe`).
3. Everything runs with DREAM's `venv311`. A CUDA GPU helps, but the CPU works too.

## The photos

The fit relies on a few things about the photos:

| Photo | What it needs to be |
|---|---|
| **Front** (required) | Straight on, eyes open, neutral face, even light. This photo matters most: it carries the likeness and textures the whole face and eyes. |
| **Profile** (one side) | A true side view with the ear visible. Face detectors miss full profiles, so it needs a manual crop (below). It's also flipped to cover the other side, so one good profile is enough. |
| **Back** | The back of the head, square on. It's fitted to the outline only. |
| Eye close-up (optional) | One eye, front on, sharp. It's wrapped onto the eyeballs. |

- **Bald, or hair pulled tight**: FLAME is a bald head, and hair gets painted onto the skull.
- **A black studio background** in every photo. The outlines come from it, and without it the
  multi-view fit (shared shape, ears, outlines) is skipped and the result is much worse.
- The same lighting in all of them, if you can. The colours are matched across photos, but
  shadows on one side still show up as dark patches (see `TODO.md`).
- High resolution (DREAM's are about 3000 px). Put them in `images\character_design\`.

## Making a new character

Say the photos are `images\ava_front.jpeg`, `images\ava_side.jpeg`, `images\ava_back.jpeg` and
`images\ava_eye.jpeg`.

**1. Find the crop for the profile.** Open the profile in an image viewer and read off the centre
of the head (x, y in pixels) and a square size that holds the whole head with a little room.
DREAM's profile is `bald=1450,780,1450`: centred at (1450, 780), 1450 px across.

**2. Fit the head and bake the texture** (takes about 3 minutes):

```bat
venv311\Scripts\python.exe emoca_test\emoca_to_obj.py ^
    --images images\ava_front.jpeg images\ava_side.jpeg images\ava_back.jpeg ^
    --crop ava_side=1450,780,1450 ^
    --mirror ava_side ^
    --back ava_back ^
    --eye-texture images\ava_eye.jpeg ^
    --out emoca_test\output_ava
```

How each photo is treated is set by the options, using the file name without its extension:
`--crop` marks side views, `--back` marks back views, and anything else counts as a front view.
`--mirror` also uses a side view flipped, for the other side of the head. Pass `--eye-texture ""`
to take the eyes from the front photo instead.

**3. Check the fit** in `emoca_test\output_ava\`:
- `*_fit.png`: each photo next to the fitted mesh laid over it. The outline, ears, eyes and mouth
  should line up. If a profile is off, fix its `--crop` and run step 2 again.
- `texture_sources.png`: which photo textured what (red front, green sides, blue back, black
  nothing).
- `dream_head.obj` / `.png`: the combined head. It keeps that name for every character, so give
  each character its own `--out` folder.

**4. Build the rig** (about 30 seconds):

```bat
venv311\Scripts\python.exe emoca_test\make_rig.py ^
    --head    emoca_test\output_ava\dream_head_flame.npz ^
    --texture emoca_test\output_ava\dream_head.png ^
    --photo   emoca_test\output_ava\ava_front_flame.npz ^
    --out     emoca_test\output_ava\ava_rig
```

That writes `ava_rig.blend`, `ava_rig.fbx` and `ava_rig_views.png` (front, 3/4, side and back
renders). `--neck 5` sets how many cm of neck are kept below the chin.

> Run the two steps separately for a new character. `make_rig.py --fit` passes extra arguments on
> to `emoca_to_obj.py`, but it keeps `--out` and `--assets` for itself. That only works with the
> default (DREAM) output folder.

## Rebuilding DREAM

```bat
emoca_test\make_rig.bat --fit
```

The defaults are DREAM's photos (`bald_front`, `bald`, `bald_back`, `eye.jpeg`) and
`emoca_test\output\`. Without `--fit`, it only rebuilds the rig from the last fit.

## What's in the rig

- **Shape keys:** `exp_00` to `exp_49` are FLAME's expression space. They're statistical blends,
  not named poses, so they're hard to animate by hand. `photo_expression` is the expression in the
  front photo.
- **Bones:** `root`, `neck`, `jaw`, `eye.L`, `eye.R`, skin-weighted from FLAME.
- **Materials:** the baked skin texture, plus a glossy one for the eyeballs.

## Files

| File | What it does |
|---|---|
| `emoca_to_obj.py` | Reads each photo with EMOCA, fits the head, bakes and blends the texture, then writes OBJs |
| `multiview_fit.py` | Fits one FLAME shape to all the photos (landmarks, ears, outlines), plus per-vertex detail |
| `ear_landmarks.py` | 20 ear points per side photo, used to place the ears |
| `make_rig.py` | Builds the rig data from FLAME and runs Blender on it |
| `blender_build_rig.py` | Runs inside Blender: the mesh, shape keys, armature, materials, `.fbx` export and preview renders |
| `make_rig.bat` | Shortcut for `make_rig.py` with DREAM's venv |
| `TODO.md` | Known blemishes and open work |
