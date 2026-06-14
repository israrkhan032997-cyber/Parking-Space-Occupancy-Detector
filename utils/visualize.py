"""
visualize.py
------------
Utility functions for drawing YOLOv8 detection results on images and
generating occupancy statistics overlays for the ParkVision app.

Exported functions
------------------
- draw_detections(image, results, show_labels) -> np.ndarray
- get_summary(results) -> dict
- overlay_stats(image, summary) -> np.ndarray
"""

from typing import Dict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class index → display name
CLASS_NAMES: Dict[int, str] = {
    0: "free",
    1: "occupied",
    2: "partial",
}

# Class index → BGR bounding-box color
CLASS_COLORS: Dict[int, tuple] = {
    0: (0, 255, 0),    # Green  — free
    1: (0, 0, 255),    # Red    — occupied
    2: (0, 255, 255),  # Yellow — partially occupied
}

# Default color for unknown classes
UNKNOWN_COLOR: tuple = (200, 200, 200)

# Overlay panel dimensions and styling
OVERLAY_PANEL_X: int = 10
OVERLAY_PANEL_Y: int = 10
OVERLAY_PANEL_WIDTH: int = 260
OVERLAY_LINE_HEIGHT: int = 34
OVERLAY_ALPHA: float = 0.6          # Transparency of the dark background panel
OVERLAY_FONT_SCALE: float = 0.75
OVERLAY_FONT_THICKNESS: int = 2
OVERLAY_FONT: int = cv2.FONT_HERSHEY_SIMPLEX

# Bounding-box drawing settings
BOX_THICKNESS: int = 2
LABEL_FONT_SCALE: float = 0.55
LABEL_FONT_THICKNESS: int = 1
LABEL_PADDING: int = 4              # Pixels of padding around label text background


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_detections(
    image: np.ndarray,
    results: object,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw colour-coded bounding boxes and optional labels on a copy of *image*.

    Each detected parking space is drawn with:
      - Green  box  → free  (class 0)
      - Red    box  → occupied  (class 1)
      - Yellow box  → partially occupied  (class 2)

    A filled label badge showing ``<class_name> <confidence>%`` is drawn
    above each box when *show_labels* is ``True``.

    Args:
        image: The source image as a ``uint8`` BGR NumPy array.
        results: The :class:`ultralytics.engine.results.Results` object
                 returned by ``model(image_array)``.
        show_labels: If ``True``, render class name and confidence on each box.

    Returns:
        A new ``uint8`` BGR NumPy array with all detections drawn.
    """
    # Work on a copy to avoid mutating the original array
    annotated: np.ndarray = image.copy()

    # Ultralytics wraps results in a list when called in batch mode;
    # handle both a single Results object and a list of them.
    result_list = results if isinstance(results, list) else [results]

    for result in result_list:
        # result.boxes is None when there are no detections
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            # Extract pixel coordinates (xyxy format)
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            # Class ID and confidence
            class_id: int = int(box.cls[0].cpu().numpy())
            confidence: float = float(box.conf[0].cpu().numpy())

            # Select color for this class
            color: tuple = CLASS_COLORS.get(class_id, UNKNOWN_COLOR)
            class_name: str = CLASS_NAMES.get(class_id, f"cls{class_id}")

            # Draw the bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            if show_labels:
                # Build the label string: e.g. "free 87%"
                label_text: str = f"{class_name} {confidence * 100:.0f}%"

                # Measure text size to draw a filled background badge
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text,
                    OVERLAY_FONT,
                    LABEL_FONT_SCALE,
                    LABEL_FONT_THICKNESS,
                )

                # Badge sits just above the top-left corner of the box
                badge_x1: int = x1
                badge_y1: int = max(0, y1 - text_h - 2 * LABEL_PADDING)
                badge_x2: int = x1 + text_w + 2 * LABEL_PADDING
                badge_y2: int = y1

                # Filled rectangle as label background
                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)

                # Choose contrasting text colour (dark text on bright backgrounds)
                text_color: tuple = (0, 0, 0) if class_id != 1 else (255, 255, 255)

                cv2.putText(
                    annotated,
                    label_text,
                    (badge_x1 + LABEL_PADDING, badge_y2 - LABEL_PADDING),
                    OVERLAY_FONT,
                    LABEL_FONT_SCALE,
                    text_color,
                    LABEL_FONT_THICKNESS,
                    cv2.LINE_AA,
                )

    return annotated


def get_summary(results: object) -> Dict[str, int]:
    """Count detected parking spaces per occupancy class.

    Args:
        results: The :class:`ultralytics.engine.results.Results` object
                 (or a list thereof) returned by ``model(image_array)``.

    Returns:
        A dictionary with integer counts::

            {
                "free":                int,
                "occupied":            int,
                "partially_occupied":  int,
                "total":               int,
            }
    """
    # Initialise counters for every known class
    counts: Dict[str, int] = {
        "free": 0,
        "occupied": 0,
        "partially_occupied": 0,
        "total": 0,
    }

    # Internal mapping from class_id to summary key
    id_to_key: Dict[int, str] = {
        0: "free",
        1: "occupied",
        2: "partially_occupied",
    }

    result_list = results if isinstance(results, list) else [results]

    for result in result_list:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            class_id: int = int(box.cls[0].cpu().numpy())
            key: str = id_to_key.get(class_id, "occupied")  # default unknown → occupied
            counts[key] += 1
            counts["total"] += 1

    return counts


def overlay_stats(
    image: np.ndarray,
    summary: Dict[str, int],
) -> np.ndarray:
    """Draw a semi-transparent statistics panel on the top-left of *image*.

    The panel shows Free, Occupied, Partial, and Total counts, each in its
    class-matching colour so the overlay is immediately readable.

    Args:
        image: The source image as a ``uint8`` BGR NumPy array (may already
               have bounding boxes drawn by :func:`draw_detections`).
        summary: The occupancy count dict returned by :func:`get_summary`.

    Returns:
        A new ``uint8`` BGR NumPy array with the stats panel composited onto
        the top-left corner.
    """
    output: np.ndarray = image.copy()
    img_h, img_w = output.shape[:2]

    # Define the rows that will appear in the panel
    rows = [
        ("Free",     summary.get("free", 0),               CLASS_COLORS[0]),
        ("Occupied", summary.get("occupied", 0),            CLASS_COLORS[1]),
        ("Partial",  summary.get("partially_occupied", 0),  CLASS_COLORS[2]),
        ("Total",    summary.get("total", 0),               (255, 255, 255)),
    ]

    # Calculate panel height dynamically based on number of rows
    num_rows: int = len(rows) + 1          # +1 for the title row
    panel_height: int = num_rows * OVERLAY_LINE_HEIGHT + 16
    panel_width: int = OVERLAY_PANEL_WIDTH

    # Clamp panel to image boundaries
    px1: int = OVERLAY_PANEL_X
    py1: int = OVERLAY_PANEL_Y
    px2: int = min(px1 + panel_width, img_w - 1)
    py2: int = min(py1 + panel_height, img_h - 1)

    # Create the semi-transparent overlay by blending a dark rectangle
    overlay: np.ndarray = output.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, output, 1.0 - OVERLAY_ALPHA, 0, output)

    # Draw a thin border around the panel
    cv2.rectangle(output, (px1, py1), (px2, py2), (80, 80, 80), 1)

    # -- Title row ---------------------------------------------------------
    title_y: int = py1 + OVERLAY_LINE_HEIGHT - 8
    cv2.putText(
        output,
        "ParkVision Stats",
        (px1 + 8, title_y),
        OVERLAY_FONT,
        OVERLAY_FONT_SCALE,
        (220, 220, 220),
        OVERLAY_FONT_THICKNESS,
        cv2.LINE_AA,
    )

    # -- Data rows ---------------------------------------------------------
    for idx, (label, count, color) in enumerate(rows):
        row_y: int = title_y + (idx + 1) * OVERLAY_LINE_HEIGHT
        text: str = f"{label}: {count}"

        cv2.putText(
            output,
            text,
            (px1 + 12, row_y),
            OVERLAY_FONT,
            OVERLAY_FONT_SCALE,
            color,
            OVERLAY_FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return output
