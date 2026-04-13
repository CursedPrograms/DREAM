#!/usr/bin/env python3

# patch_deepface_final.py
import os
import sys

try:
    import deepface
    fbdeepface_path = os.path.join(os.path.dirname(deepface.__file__), "basemodels", "FbDeepFace.py")
except ImportError:
    print("DeepFace not found in this environment.")
    sys.exit(1)

with open(fbdeepface_path, "r") as f:
    code = f.read()

# Replace any multi-import that includes LocallyConnected2D
import_line = "from tensorflow.keras.layers import ("
if import_line in code:
    lines = code.splitlines()
    new_lines = []
    for line in lines:
        if "from tensorflow.keras.layers import (" in line:
            # Remove LocallyConnected2D from this import
            line = line.replace("LocallyConnected2D,", "")
        new_lines.append(line)
    code = "\n".join(new_lines)
    # Add a direct import from keras.layers at the top
    code = "from keras.layers import LocallyConnected2D\n" + code

with open(fbdeepface_path, "w") as f:
    f.write(code)

print(f"FbDeepFace.py fully patched for Python 3.12 + TF 2.21 + Keras 3.x at: {fbdeepface_path}")