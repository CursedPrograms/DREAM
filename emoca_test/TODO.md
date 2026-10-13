# EMOCA head: still to do

Rebuild everything with `emoca_test\make_rig.bat --fit` (about 3 minutes). Check the result in
`output\dream_rig_views.png` and the per-photo fits in `output\*_fit.png`.

## Texture
- **Dark band behind the ears, up to the crown.** No photo sees that strip well: the profiles
  are in shadow there (they now fade out behind the ear, `BACK_FADE`), and the back photo darkens
  at its edges. Tried keeping only the back photo's square-on part (`facing < 0.65`): worse, hard
  dark edges on the back. Next to try: match the back photo's colours to the profiles' only in a
  narrow band where both see the skin well, or use `bald2` (lit from behind) just for that strip.
  Measured: where the profiles and the back photo overlap, the profiles are about half as bright,
  so it's the profiles' shadow, not the back photo. `even_tone` (lifts dips in the scalp's broad
  tone, `TONE_DIP`) softened it a little, but it's still visible in the side view. The band
  also follows the UV seam between the head and the back-of-head islands.
  Tried 2026-10-12, none helped: matching the profiles to the back photo as well as the front (the
  overlap is too narrow, and the back photo is dark at its edge too: 0.74 in the middle, 0.40 at
  the band, against the profile's 0.17); `TONE_DIP` 161; `bald2` (lit from behind) as a profile that
  textures only behind the ear - brighter there (0.25-0.38), but it changed the fitted shape and the
  band at the crown looked harder. Likely needs real de-lighting (divide out the shading).
- **The fit isn't repeatable.** EMOCA's outputs differ by ~1e-6 between identical runs (GPU float
  noise, not fixed by `torch.use_deterministic_algorithms` or deterministic cuDNN), and the
  multi-view fit magnifies that to ~3 mm on the head (shape coefficients by ~0.2-0.4). So two
  renders can differ in shape with no change at all: render twice before believing a shape change.
  Either make the fit better conditioned (stronger priors) or run EMOCA on the CPU.
- **Dark smudge low on the cheek** in the 3/4 and side views, where the front photo hands over
  to the profile (edge of `face_tex`). Possibly the mirrored profile's flipped lighting.
- **Faint arc on the upper forehead** where the front photo fades into the profile.

## Shape
- Previews use a 50 mm lens close up, which widens the skull; compare with a longer lens before
  judging the skull shape.

## Eyes
- Her right eye can still show a little white below the iris; check the lower lid against the
  photo with the eyes close-up.

## Rig
- Named ARKit-style shapes (jawOpen, eyeBlink...) built from FLAME's expression space, for
  lip-sync. FLAME's 50 `exp_*` keys are statistical blends, hard to animate by hand.

## Notes
- `assets/` (EMOCA, FLAME, the ear landmarker) is research-use only and stays out of git.
