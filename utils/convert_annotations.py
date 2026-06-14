"""
convert_annotations.py
-----------------------
Parses annotations.xml and parking.csv from the parking lot dataset,
converts polygon/bounding-box annotations to YOLO format, splits the
dataset 80/20 into train/val sets, copies images, and writes label files.

Label mapping
-------------
  free_parking_space           -> 0  (free)
  not_free_parking_space       -> 1  (occupied)
  partially_free_parking_space -> 2  (partially_occupied)
"""

import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR: Path = Path(__file__).resolve().parent.parent / "dataset"
IMAGES_SRC_DIR: Path = DATASET_DIR / "images"
ANNOTATIONS_XML: Path = DATASET_DIR / "annotations.xml"
PARKING_CSV: Path = DATASET_DIR / "parking.csv"

IMAGES_TRAIN_DIR: Path = DATASET_DIR / "images" / "train"
IMAGES_VAL_DIR: Path = DATASET_DIR / "images" / "val"
LABELS_TRAIN_DIR: Path = DATASET_DIR / "labels" / "train"
LABELS_VAL_DIR: Path = DATASET_DIR / "labels" / "val"

TRAIN_RATIO: float = 0.8
RANDOM_SEED: int = 42

# Map XML label names to YOLO class IDs
LABEL_MAP: Dict[str, int] = {
    "free_parking_space": 0,
    "not_free_parking_space": 1,
    "partially_free_parking_space": 2,
}

CLASS_NAMES: Dict[int, str] = {
    0: "free",
    1: "occupied",
    2: "partially_occupied",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def polygon_to_bbox(points_str: str) -> Tuple[float, float, float, float]:
    """Convert a CVAT polygon points string to an axis-aligned bounding box.

    Args:
        points_str: Semicolon-separated ``x,y`` coordinate pairs, e.g.
                    ``"10.0,20.0;30.0,25.0;28.0,50.0;8.0,48.0"``.

    Returns:
        A tuple ``(xmin, ymin, xmax, ymax)`` in pixel coordinates.
    """
    pairs = points_str.strip().split(";")
    xs: List[float] = []
    ys: List[float] = []
    for pair in pairs:
        x_str, y_str = pair.split(",")
        xs.append(float(x_str))
        ys.append(float(y_str))
    return min(xs), min(ys), max(xs), max(ys)


def to_yolo_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """Convert absolute pixel bounding box to normalised YOLO format.

    Args:
        xmin: Left edge of the bounding box in pixels.
        ymin: Top edge of the bounding box in pixels.
        xmax: Right edge of the bounding box in pixels.
        ymax: Bottom edge of the bounding box in pixels.
        img_width: Full image width in pixels.
        img_height: Full image height in pixels.

    Returns:
        ``(cx, cy, w, h)`` all normalised to ``[0.0, 1.0]``.
    """
    cx = (xmin + xmax) / (2.0 * img_width)
    cy = (ymin + ymax) / (2.0 * img_height)
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height

    # Clamp values to valid range to handle edge annotations
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def ensure_dirs() -> None:
    """Create all required output directories if they do not already exist."""
    for directory in (
        IMAGES_TRAIN_DIR,
        IMAGES_VAL_DIR,
        LABELS_TRAIN_DIR,
        LABELS_VAL_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_xml_annotations(
    xml_path: Path,
) -> Dict[str, Dict]:
    """Parse a CVAT-style annotations.xml file.

    Extracts image metadata and polygon annotations, converting each polygon
    to an axis-aligned bounding box with its YOLO class ID.

    Args:
        xml_path: Path to the ``annotations.xml`` file.

    Returns:
        A dictionary keyed by the bare image filename (e.g. ``"0.png"``)
        whose values are dicts with keys:
          - ``width``  (int)
          - ``height`` (int)
          - ``annotations`` (list of ``[class_id, cx, cy, w, h]``)
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    image_data: Dict[str, Dict] = {}

    for image_elem in root.findall("image"):
        # Extract image attributes
        raw_name: str = image_elem.get("name", "")
        img_width: int = int(image_elem.get("width", 0))
        img_height: int = int(image_elem.get("height", 0))

        if img_width == 0 or img_height == 0:
            print(f"  [WARN] Skipping '{raw_name}' — zero dimensions in XML.")
            continue

        # Use only the filename part (strip any leading path prefix)
        img_filename: str = Path(raw_name).name

        annotations: List[List[float]] = []

        for polygon in image_elem.findall("polygon"):
            label: str = polygon.get("label", "")
            if label not in LABEL_MAP:
                # Unknown label — skip silently
                continue
            class_id: int = LABEL_MAP[label]

            points_str: Optional[str] = polygon.get("points")
            if not points_str:
                continue

            try:
                xmin, ymin, xmax, ymax = polygon_to_bbox(points_str)
                cx, cy, w, h = to_yolo_bbox(
                    xmin, ymin, xmax, ymax, img_width, img_height
                )
                annotations.append([class_id, cx, cy, w, h])
            except (ValueError, ZeroDivisionError) as exc:
                print(f"  [WARN] Bad polygon in '{img_filename}': {exc}")
                continue

        image_data[img_filename] = {
            "width": img_width,
            "height": img_height,
            "annotations": annotations,
        }

    return image_data


# ---------------------------------------------------------------------------
# Dataset split and file writing
# ---------------------------------------------------------------------------

def split_dataset(
    image_files: List[str],
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str]]:
    """Randomly split a list of image filenames into train and val sets.

    Args:
        image_files: List of image filenames to split.
        train_ratio: Fraction of images to allocate to the training set.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(train_files, val_files)``.
    """
    shuffled = image_files.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    split_idx: int = max(1, int(len(shuffled) * train_ratio))
    train_files = shuffled[:split_idx]
    val_files = shuffled[split_idx:]

    # Guarantee at least one file in val when dataset is very small
    if len(val_files) == 0 and len(train_files) > 1:
        val_files = [train_files.pop()]

    return train_files, val_files


def write_label_file(
    label_path: Path,
    annotations: List[List[float]],
) -> None:
    """Write a single YOLO-format label ``.txt`` file.

    Each line has the format: ``class_id cx cy w h``
    with floating-point values rounded to 6 decimal places.

    Args:
        label_path: Full path for the output ``.txt`` file.
        annotations: List of ``[class_id, cx, cy, w, h]`` records.
    """
    lines: List[str] = []
    for ann in annotations:
        class_id = int(ann[0])
        cx, cy, w, h = ann[1], ann[2], ann[3], ann[4]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path.write_text("\n".join(lines), encoding="utf-8")


def copy_image_and_label(
    img_filename: str,
    image_data: Dict[str, Dict],
    images_src: Path,
    images_dst: Path,
    labels_dst: Path,
) -> bool:
    """Copy an image file and write its corresponding label file.

    Args:
        img_filename: Bare filename of the image (e.g. ``"0.png"``).
        image_data: Parsed annotation dict from :func:`parse_xml_annotations`.
        images_src: Source directory containing original images.
        images_dst: Destination directory for copied images.
        labels_dst: Destination directory for written label files.

    Returns:
        ``True`` if the image was successfully copied, ``False`` otherwise.
    """
    src_path: Path = images_src / img_filename
    if not src_path.exists():
        print(f"  [WARN] Image not found, skipping: {src_path}")
        return False

    # Copy the image
    shutil.copy2(str(src_path), str(images_dst / img_filename))

    # Write the label file (stem replaces extension with .txt)
    stem: str = Path(img_filename).stem
    label_path: Path = labels_dst / f"{stem}.txt"
    annotations = image_data.get(img_filename, {}).get("annotations", [])
    write_label_file(label_path, annotations)

    return True


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(
    image_data: Dict[str, Dict],
    train_files: List[str],
    val_files: List[str],
) -> None:
    """Print a formatted summary of the converted dataset to stdout.

    Args:
        image_data: Parsed annotation data for all images.
        train_files: List of image filenames assigned to training.
        val_files: List of image filenames assigned to validation.
    """
    # Aggregate class counts across all annotations
    class_counts: Dict[int, int] = {0: 0, 1: 0, 2: 0}
    total_annotations: int = 0

    for data in image_data.values():
        for ann in data["annotations"]:
            cid = int(ann[0])
            class_counts[cid] = class_counts.get(cid, 0) + 1
            total_annotations += 1

    print("\n" + "=" * 55)
    print("  DATASET CONVERSION SUMMARY")
    print("=" * 55)
    print(f"  Total images processed : {len(image_data)}")
    print(f"  Total annotations      : {total_annotations}")
    print(f"  Train images           : {len(train_files)}")
    print(f"  Val images             : {len(val_files)}")
    print("-" * 55)
    for cid, name in CLASS_NAMES.items():
        print(f"  Class {cid} ({name:<22}) : {class_counts.get(cid, 0):>5} boxes")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def convert_dataset(
    xml_path: Path = ANNOTATIONS_XML,
    csv_path: Path = PARKING_CSV,
    images_src: Path = IMAGES_SRC_DIR,
) -> None:
    """Orchestrate the full dataset conversion pipeline.

    Steps performed:
      1. Parse ``annotations.xml`` for bounding boxes and labels.
      2. Optionally read ``parking.csv`` to confirm available image files.
      3. Split images 80/20 into train/val.
      4. Copy images and write YOLO ``.txt`` label files.
      5. Print a conversion summary.

    Args:
        xml_path: Path to the CVAT annotations XML file.
        csv_path: Path to the parking occupancy CSV file.
        images_src: Directory containing source image files.
    """
    print("\n[INFO] Starting dataset conversion …")

    # -- Step 1: Ensure output directories exist ---------------------------
    try:
        ensure_dirs()
        print("[INFO] Output directories ready.")
    except OSError as exc:
        print(f"[ERROR] Could not create output directories: {exc}")
        raise

    # -- Step 2: Parse XML annotations -------------------------------------
    image_data: Dict[str, Dict] = {}
    if xml_path.exists():
        try:
            print(f"[INFO] Parsing XML: {xml_path}")
            image_data = parse_xml_annotations(xml_path)
            print(f"[INFO] Found {len(image_data)} annotated images in XML.")
        except ET.ParseError as exc:
            print(f"[ERROR] Failed to parse XML: {exc}")
            raise
    else:
        print(f"[WARN] annotations.xml not found at {xml_path}. Skipping XML parsing.")

    # -- Step 3: Cross-reference with CSV (optional) -----------------------
    csv_image_files: List[str] = []
    if csv_path.exists():
        try:
            print(f"[INFO] Reading CSV: {csv_path}")
            df = pd.read_csv(str(csv_path))
            # Extract bare filenames from the 'image' column
            csv_image_files = [Path(p).name for p in df["image"].tolist()]
            print(f"[INFO] CSV references {len(csv_image_files)} image files.")
        except Exception as exc:
            print(f"[WARN] Could not read CSV ({exc}). Proceeding without it.")
    else:
        print(f"[WARN] parking.csv not found at {csv_path}. Skipping CSV cross-reference.")

    # -- Step 4: Determine the final image list ----------------------------
    # Priority: images that appear in both XML and CSV (if CSV is available),
    # otherwise use whatever was parsed from the XML.
    if csv_image_files and image_data:
        # Keep only images that the XML has annotations for
        combined = [f for f in csv_image_files if f in image_data]
        # Also add XML images not listed in CSV (extra safety net)
        xml_only = [f for f in image_data if f not in csv_image_files]
        all_image_files: List[str] = combined + xml_only
        print(f"[INFO] Combined dataset: {len(combined)} matched, "
              f"{len(xml_only)} XML-only images.")
    elif image_data:
        all_image_files = list(image_data.keys())
    elif csv_image_files:
        # CSV-only mode: no annotation data available
        all_image_files = csv_image_files
        print("[WARN] Running in CSV-only mode — label files will be empty.")
    else:
        print("[ERROR] No usable data found. Check dataset paths.")
        return

    if not all_image_files:
        print("[ERROR] Image list is empty after filtering. Aborting.")
        return

    # -- Step 5: Split into train / val ------------------------------------
    train_files, val_files = split_dataset(all_image_files)
    print(f"[INFO] Split → train: {len(train_files)}, val: {len(val_files)}")

    # -- Step 6: Copy images and write labels ------------------------------
    train_copied = 0
    val_copied = 0

    print("[INFO] Copying train images and writing labels …")
    for filename in train_files:
        try:
            success = copy_image_and_label(
                filename, image_data, images_src, IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR
            )
            if success:
                train_copied += 1
        except Exception as exc:
            print(f"  [ERROR] Failed to process '{filename}': {exc}")

    print("[INFO] Copying val images and writing labels …")
    for filename in val_files:
        try:
            success = copy_image_and_label(
                filename, image_data, images_src, IMAGES_VAL_DIR, LABELS_VAL_DIR
            )
            if success:
                val_copied += 1
        except Exception as exc:
            print(f"  [ERROR] Failed to process '{filename}': {exc}")

    print(f"[INFO] Copied — train: {train_copied}, val: {val_copied}")

    # -- Step 7: Print summary ---------------------------------------------
    print_summary(image_data, train_files, val_files)
    print("[INFO] Dataset conversion complete.")


if __name__ == "__main__":
    convert_dataset()
