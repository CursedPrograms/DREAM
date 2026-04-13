#!/usr/bin/env python3

import os
import sys
import glob
from types import ModuleType
from pathlib import Path

# --- FIXES ---
pkg_mock = ModuleType("pkg_resources")
sys.modules["pkg_resources"] = pkg_mock
import numpy as np
import tensorflow as tf
from PIL import Image

# --- THE PATTERN ENGINE ---
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# Layer choice is EVERYTHING. 
# mixed2/3 = Geometrics/Fractals
# mixed4/5 = Eyes/Birds/Dogs
# mixed8/9 = Complex architectures
dream_layers = ['mixed3', 'mixed4', 'mixed5'] 
layers = [base_model.get_layer(name).output for name in dream_layers]
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    losses = [tf.math.reduce_mean(act) for act in layer_activations]
    return tf.reduce_sum(losses)

@tf.function
def deepdream_step(img, model, step_size):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    gradients = tape.gradient(loss, img)
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    img = img + gradients * step_size
    img = tf.clip_by_value(img, -1, 1)
    return loss, img

def run_deep_dream_with_octaves(img, steps_per_octave=20, step_size=0.01, octave_scale=1.3, num_octaves=3):
    img = tf.keras.applications.inception_v3.preprocess_input(np.array(img))
    img = tf.convert_to_tensor(img)

    original_shape = tf.shape(img)[:2]

    for i in range(num_octaves):
        # Scale the image down for the current octave
        new_size = tf.cast(tf.cast(original_shape, tf.float32) / (octave_scale**i), tf.int32)
        img = tf.image.resize(img, new_size)

        for _ in range(steps_per_octave):
            _, img = deepdream_step(img, dream_model, step_size)
        
        print(f"  Finished Octave {i}...", end="\r")

    # Resize back to original size
    img = tf.image.resize(img, original_shape)
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

def process_session_hyper_trippy(session_path):
    input_dir = Path(session_path)
    output_dir = input_dir.parent / (input_dir.name + "_ULTRA_TRIPPY")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(glob.glob(str(input_dir / "frame_*.png")))
    
    for f_path in frame_paths:
        img_name = os.path.basename(f_path)
        raw_img = Image.open(f_path).convert('RGB')
        
        # Apply the multi-scale deep dream
        dream_result = run_deep_dream_with_octaves(raw_img, num_octaves=4, steps_per_octave=15)
        
        final_img = Image.fromarray(dream_result.numpy())
        final_img.save(output_dir / img_name)
        print(f"✔ Hallucinated {img_name}")

if __name__ == "__main__":
    target = r'D:\cc\Friday\output\dreams\dream_20260412_191749_interpolate'
    process_session_hyper_trippy(target)