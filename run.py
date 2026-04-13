#!/usr/bin/env python3

import os
import sys

print("Switching to dream.py...")

os.execv(sys.executable, [sys.executable, "scripts/dream.py"])
