#!/usr/bin/env python3
 
import os
import sys

print("Switching to options.py...")

os.execv(sys.executable, [sys.executable, "scripts/options.py"])
