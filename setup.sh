#!/bin/bash
# ultralytics pulls in opencv-python (GUI) as dependency
# Force replace with headless version after all installs
pip uninstall -y opencv-python 2>/dev/null || true
pip install --force-reinstall opencv-python-headless --quiet
