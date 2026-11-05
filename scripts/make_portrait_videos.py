"""make_portrait_videos.py - portrait (upright phone) versions of DREAM's clips.

Each landscape clip in videos/ gets a 720x1280 twin in videos/portrait/ with the
same name. Her face fills the phone's width: a square from the middle of the
1920x1080 frame, faded at the top and bottom into the black she stands in.
Her page plays the twin while the screen is taller than it is wide, and goes
back to the landscape clip when the phone is turned (app.py, static/js/dream.js).

    venv311\\Scripts\\python.exe scripts\\make_portrait_videos.py          # new or changed clips
    venv311\\Scripts\\python.exe scripts\\make_portrait_videos.py --all    # every clip again

Needs ffmpeg on PATH. A clip with no twin plays its landscape version, cropped.
"""
import os
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS = os.path.join(ROOT, "videos")
PORTRAIT = os.path.join(VIDEOS, "portrait")

W, H = 720, 1280          # the portrait frame
SIDE = 1080               # the square taken from the middle of the landscape frame
TOP = (H - W) // 2        # where that square sits once scaled to W x W
FADE = 150                # pixels of fade at its top and bottom edges


def fade_overlay(path):
    """Black above and below the square, easing to clear over FADE pixels inside it."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    px = img.load()
    bottom = TOP + W
    for y in range(H):
        if y < TOP or y >= bottom:
            a = 255
        else:
            d = min(y - TOP, bottom - 1 - y)
            a = 0 if d >= FADE else 255 - int(255 * (d / FADE) ** 1.5)
        for x in range(W):
            px[x, y] = (0, 0, 0, a)
    img.save(path)


def convert(src, dst, overlay):
    vf = (f"[0:v]crop={SIDE}:{SIDE},scale={W}:{W},pad={W}:{H}:0:{TOP}:black[v];"
          f"[v][1:v]overlay=0:0,format=yuv420p")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", overlay,
                    "-filter_complex", vf, "-map", "0:a?", "-c:v", "libx264", "-crf", "26",
                    "-preset", "medium", "-movflags", "+faststart", "-c:a", "copy", dst],
                   check=True)


def main():
    redo = "--all" in sys.argv
    os.makedirs(PORTRAIT, exist_ok=True)
    clips = sorted(f for f in os.listdir(VIDEOS) if f.endswith(".mp4"))
    with tempfile.TemporaryDirectory() as tmp:
        overlay = os.path.join(tmp, "fade.png")
        fade_overlay(overlay)
        made = 0
        for name in clips:
            src, dst = os.path.join(VIDEOS, name), os.path.join(PORTRAIT, name)
            if not redo and os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                continue
            print(f"[portrait] {name}")
            convert(src, dst, overlay)
            made += 1
    print(f"[portrait] {made} made, {len(clips) - made} already up to date -> videos/portrait/")


if __name__ == "__main__":
    main()
