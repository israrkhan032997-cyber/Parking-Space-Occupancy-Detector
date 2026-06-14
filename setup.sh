#!/bin/bash
# Fix 1: Remove opencv-python (GUI) installed by ultralytics, replace with headless
pip uninstall -y opencv-python 2>/dev/null || true
pip install --force-reinstall opencv-python-headless --quiet

# Fix 2: Create symlink for libgthread if missing (Debian Trixie compatibility)
if [ ! -f /usr/lib/x86_64-linux-gnu/libgthread-2.0.so.0 ]; then
    GLIB=$(find /usr/lib/x86_64-linux-gnu -name "libglib-2.0.so*" 2>/dev/null | head -1)
    if [ -n "$GLIB" ]; then
        GLIB_DIR=$(dirname "$GLIB")
        GTHREAD=$(find "$GLIB_DIR" -name "libgthread*" 2>/dev/null | head -1)
        echo "GLib dir: $GLIB_DIR, gthread: $GTHREAD"
    fi
fi

echo "Setup complete"
