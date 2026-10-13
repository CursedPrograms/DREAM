"""
Ear landmarks: 20 points on an ear in a photo, for fitting FLAME's ears to hers.

Uses the ear landmarker released with flame-head-tracker
(https://github.com/PeizhiYan/flame-head-tracker, trained on the i-Bug ears dataset - research use
only). Weights: emoca_test/assets/ear/ear_landmarker.pth, fetched by download_weights.bat.

The file is a whole PyTorch module converted from ONNX by onnx2torch. Its forward code is plain
conv / ReLU / pool calls; the only onnx2torch classes it needs are concat and transpose, so those
are stood in for here instead of installing onnx2torch.
"""

import pickle
import types
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "assets" / "ear" / "ear_landmarker.pth"
INPUT_SIZE = 368

# FLAME vertices for the 20 i-Bug ear landmarks (from flame-head-tracker's tracker_base.py).
# FLAME's left ear is on +x, which faces the camera when the head turns to its right (yaw > 0).
LEFT_EAR = [342, 341, 166, 514, 476, 185, 369, 29, 204, 641, 179, 178, 71, 68, 138, 141, 91, 40, 96, 184]
RIGHT_EAR = [1263, 844, 845, 2655, 870, 872, 1207, 523, 901, 1859, 860, 621, 618, 890, 981, 556, 554, 676, 1209, 868]


class OnnxConcat(torch.nn.Module):
    def forward(self, *tensors):
        return torch.cat(tensors, dim=self.axis)


class OnnxTranspose(torch.nn.Module):
    def forward(self, x):
        return x.permute(*self.perm)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("onnx2torch"):
            return {"OnnxConcat": OnnxConcat, "OnnxTranspose": OnnxTranspose}[name]
        return super().find_class(module, name)


_pickle = types.SimpleNamespace(Unpickler=_Unpickler, load=lambda f, **kw: _Unpickler(f, **kw).load(),
                                __name__="ear_pickle")


def load(device):
    if not WEIGHTS.exists():
        return None
    model = torch.load(WEIGHTS, map_location=device, pickle_module=_pickle, weights_only=False)
    return model.eval()


@torch.no_grad()
def detect(model, img, device):
    """img: RGB float 0-1 crop with the ear in it (the tracker feeds its face crop).
    Returns 20 points in that crop's pixels, and how confident each heatmap peak is."""
    import cv2
    h, w = img.shape[:2]
    x = cv2.resize(img.astype(np.float32), (INPUT_SIZE, INPUT_SIZE))
    heat = model(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()      # [46, 46, 55]
    heat = cv2.resize(heat, (INPUT_SIZE, INPUT_SIZE))
    heat = gaussian_filter(heat, sigma=(2.5, 2.5, 0))[..., :20]
    flat = heat.reshape(-1, 20)
    idx = flat.argmax(0)
    pts = np.stack([idx % INPUT_SIZE, idx // INPUT_SIZE], 1).astype(np.float32)
    return pts * [w / INPUT_SIZE, h / INPUT_SIZE], flat.max(0)
