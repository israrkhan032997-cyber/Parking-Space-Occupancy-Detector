#!/bin/bash
# Force replace opencv-python with headless version
# ultralytics pulls in opencv-python which needs libGL
# This script runs before app startup on Streamlit Cloud
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip install opencv-python-headless --quiet
