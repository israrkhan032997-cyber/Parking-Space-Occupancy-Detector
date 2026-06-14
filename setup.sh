#!/bin/bash
# ultralytics installs opencv-python (GUI) which needs libgthread-2.0.so.0
# Force replace with headless version AFTER all packages are installed
pip uninstall -y opencv-python 2>/dev/null || true
pip install --force-reinstall opencv-python-headless --quiet
echo "OpenCV headless reinstall done"
python -c "import cv2; print('cv2 OK:', cv2.__version__)"
