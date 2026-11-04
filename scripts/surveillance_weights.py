"""
surveillance_weights.py - the models her camera's detector uses (surveillance_core.py), into
resources/weights/. download_weights.bat runs it; safe to run again (what's there is skipped).

    yolov8n.pt                 people and objects (Ultralytics YOLOv8 nano)
    age_googlenet.onnx         age   } ONNX model zoo, on the faces it finds
    gender_googlenet.onnx      gender}
    emotion-ferplus-8.onnx     expression
    face_detection_yunet_2023mar.onnx   faces (copied from models/faces when it's there)

Copied from ModelBackup/surveillance (config.json, e.g. F:/AIModels) when they're there; downloaded
and then backed up there otherwise.
"""
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "resources" / "weights"
FILES = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
    "age_googlenet.onnx": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/age_googlenet.onnx",
    "gender_googlenet.onnx": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/age_gender/models/gender_googlenet.onnx",
    "emotion-ferplus-8.onnx": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
}


def _backup():
    try:
        d = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["Config"]["DREAM"].get("ModelBackup", "")
    except (OSError, ValueError, KeyError):
        d = ""
    return Path(d) / "surveillance" if d else None


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    backup = _backup()
    ok = True
    yunet = ROOT / "models" / "faces" / "face_detection_yunet_2023mar.onnx"
    if yunet.exists() and not (DEST / yunet.name).exists():
        shutil.copy2(yunet, DEST / yunet.name)
    for name, url in FILES.items():
        out = DEST / name
        if out.exists() and out.stat().st_size > 1000:
            print(f"  {name}: already there")
            continue
        if backup and (backup / name).exists():
            shutil.copy2(backup / name, out)
            print(f"  {name}: copied from {backup}")
            continue
        try:
            print(f"  {name}: downloading ...")
            req = urllib.request.Request(url, headers={"User-Agent": "DREAM"})
            with urllib.request.urlopen(req, timeout=120) as r, open(out.with_suffix(".part"), "wb") as f:
                shutil.copyfileobj(r, f)
            out.with_suffix(".part").replace(out)
            if backup:
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out, backup / name)
        except Exception as e:
            ok = False
            print(f"  {name}: FAILED ({e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
