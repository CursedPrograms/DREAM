#!/usr/bin/env python3

import os
import sys
import glob
from types import ModuleType
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

# --- PYTHON 3.12 COMPATIBILITY FIX ---
pkg_mock = ModuleType("pkg_resources")
try:
    from packaging.version import parse as parse_version
    pkg_mock.parse_version = parse_version
except ImportError:
    pkg_mock.parse_version = lambda v: [int(x) for x in v.split('.') if x.isdigit()]
sys.modules["pkg_resources"] = pkg_mock

# --- CONFIGURATION ---
# Target folder with your kaleidoscopic frames
INPUT_SESSION = r'D:\cc\Friday\output\dreams\dream_20260412_191749_interpolate'
# DeepDream aggressive layers
DREAM_LAYERS = ['mixed7', 'mixed8'] 
# Performance / Intensity
NUM_OCTAVES = 4        # Higher = more detailed/fractal
OCTAVE_SCALE = 2.4     # Scaling factor between octaves
STEPS_PER_OCTAVE = 20  # Total iterations per scale
STEP_SIZE = 0.8      # Gradient step (higher = more fried)
JITTER_MAX = 64        # Random roll to prevent static artifacts

# --- MODEL SETUP ---
print("👁️  Loading the Inception Nightmare Engine...")
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
layers = [base_model.get_layer(name).output for name in DREAM_LAYERS]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    # Target the mean of activations to "amplify" what the model sees
    return tf.reduce_sum([tf.math.reduce_mean(act) for act in layer_activations])

@tf.function
def deepdream_step(img, model, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    
    gradients = tape.gradient(loss, img)
    # Normalize gradients so every pixel gets a push
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    img = img + gradients * step_size
    img = tf.clip_by_value(img, -1, 1)
    return loss, img

def random_roll(img, maxroll):
    shift = np.random.randint(-maxroll, maxroll + 1, size=2)
    return tf.roll(img, shift=shift, axis=[0, 1]), shift

def render_deep_dream(img):
    img = tf.keras.applications.inception_v3.preprocess_input(np.array(img))
    img = tf.convert_to_tensor(img)

    original_shape = tf.shape(img)[:2]

    for octave in range(NUM_OCTAVES):
        if octave > 0:
            # Upscale image based on original size and scale factor
            new_size = tf.cast(tf.cast(original_shape, tf.float32) / (OCTAVE_SCALE**(NUM_OCTAVES-1-octave)), tf.int32)
            img = tf.image.resize(img, new_size)

        for _ in range(STEPS_PER_OCTAVE):
            # Jittering helps the DeepDream patterns spread organically
            img, shift = random_roll(img, JITTER_MAX)
            _, img = deepdream_step(img, dream_model, STEP_SIZE)
            img = tf.roll(img, shift=-shift, axis=[0, 1])

    # Final resize back to target
    img = tf.image.resize(img, original_shape)
    # Post-process back to uint8 [0, 255]
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

def main():
    input_dir = Path(INPUT_SESSION)
    output_dir = input_dir.parent / (input_dir.name + "_DEEP_DREAM_NIGHTMARE")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(glob.glob(str(input_dir / "frame_*.png")))
    if not frame_paths:
        print(f"❌ No frames found in {input_dir}")
        return

    print(f"🚀 Processing {len(frame_paths)} frames at {NUM_OCTAVES} octaves...")
    
    for f_path in frame_paths:
        img_name = os.path.basename(f_path)
        raw_img = Image.open(f_path).convert('RGB')
        
        # The magic happens here
        result_tensor = render_deep_dream(raw_img)
        
        final_img = Image.fromarray(result_tensor.numpy())
        final_img.save(output_dir / img_name)
        print(f"✅ Hallucinated {img_name}")

    print(f"\n✨ DONE. Find the heavy deep dream output in:\n{output_dir}")

if __name__ == "__main__":
    main()