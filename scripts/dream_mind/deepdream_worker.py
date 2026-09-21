"""
deepdream_worker.py - TensorFlow DeepDream for one image, run as its own process.

    python deepdream_worker.py <in.jpg> <out.jpg> '<preset json>' <max_side>

The DeepDream code from the old scripts (deep_dream.py, deep_dream_batch.py,
nightmare_dreamer.py) with their hard-coded paths and settings turned into
arguments. It runs in a separate process so it can be killed when she wakes and
so a TensorFlow that fails to load can't take DREAM down with it.
"""

import json
import sys
from types import ModuleType

# TensorFlow's old code imports pkg_resources; the old scripts stub it out for Python 3.12.
_pkg = ModuleType("pkg_resources")
try:
    from packaging.version import parse as _parse
    _pkg.parse_version = _parse
except ImportError:
    _pkg.parse_version = lambda v: [int(x) for x in v.split(".") if x.isdigit()]
sys.modules.setdefault("pkg_resources", _pkg)

import numpy as np
import tensorflow as tf
from PIL import Image


def main(src, dst, preset, max_side):
    img = Image.open(src).convert("RGB")
    img.thumbnail((max_side, max_side))

    base = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
    model = tf.keras.Model(inputs=base.input, outputs=[base.get_layer(n).output for n in preset["layers"]])

    def loss_of(x):
        acts = model(tf.expand_dims(x, 0))
        if not isinstance(acts, (list, tuple)):
            acts = [acts]
        return tf.reduce_sum([tf.math.reduce_mean(a) for a in acts])

    @tf.function
    def step(x, size):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = loss_of(x)
        g = tape.gradient(loss, x)
        g /= tf.math.reduce_std(g) + 1e-8
        return tf.clip_by_value(x + g * size, -1, 1)

    x = tf.convert_to_tensor(tf.keras.applications.inception_v3.preprocess_input(np.array(img)))
    original = tf.shape(x)[:2]
    for octave in range(preset["octaves"]):
        new_size = tf.cast(tf.cast(original, tf.float32) / (preset["scale"] ** octave), tf.int32)
        x = tf.image.resize(x, new_size)
        for _ in range(preset["steps"]):
            x = step(x, preset["step_size"])
    x = tf.image.resize(x, original)
    out = tf.cast(255 * (x + 1.0) / 2.0, tf.uint8).numpy()
    Image.fromarray(out).save(dst, quality=90)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4]))
