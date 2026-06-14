"""
train.py
--------
Fine-tunes a YOLOv8n model on the parking space occupancy dataset and
saves the best checkpoint to models/best.pt.

Usage
-----
  python train.py

Prerequisites
-------------
  1. Run ``python utils/convert_annotations.py`` first to populate
     dataset/images/train, dataset/images/val, dataset/labels/train,
     and dataset/labels/val.
  2. Ensure ``data.yaml`` is present in the project root.
  3. Ensure ``ultralytics`` is installed (``pip install ultralytics``).
"""

import shutil
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH: str = "yolov8n.pt"       # Pretrained YOLOv8-nano weights (auto-downloaded)
DATA_YAML: str = "data.yaml"          # YOLO dataset configuration file
EPOCHS: int = 40                      # Number of training epochs
IMG_SIZE: int = 640                   # Input image size (pixels)
BATCH: int = 16                       # Batch size (reduce to 8 if GPU VRAM is limited)
SAVE_DIR: str = "models"              # Directory to save the final best.pt
PROJECT_DIR: str = "runs/train"       # Ultralytics training output directory
RUN_NAME: str = "parking_yolov8"      # Sub-folder name within PROJECT_DIR


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> YOLO:
    """Load a pretrained or custom YOLOv8 model.

    Args:
        model_path: Path to a ``.pt`` weights file, or an Ultralytics model
                    shorthand such as ``"yolov8n.pt"`` (auto-downloaded).

    Returns:
        An initialised :class:`ultralytics.YOLO` instance.

    Raises:
        FileNotFoundError: If a local path is given but the file does not exist.
        RuntimeError: If the model cannot be loaded.
    """
    print(f"[INFO] Loading base model: {model_path}")
    try:
        model = YOLO(model_path)
        print("[INFO] Model loaded successfully.")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_path}': {exc}") from exc


def train_model(model: YOLO) -> Optional[object]:
    """Fine-tune the YOLO model on the parking dataset.

    Trains for ``EPOCHS`` epochs using the configuration in ``DATA_YAML``.
    Ultralytics automatically saves ``best.pt`` and ``last.pt`` inside
    ``runs/train/<RUN_NAME>/weights/``.

    Args:
        model: A loaded :class:`ultralytics.YOLO` instance.

    Returns:
        The training results object returned by ``model.train()``, or
        ``None`` if training failed.

    Raises:
        FileNotFoundError: If ``DATA_YAML`` does not exist.
        RuntimeError: If training raises an unexpected error.
    """
    # Verify the data config exists before starting a potentially long run
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"data.yaml not found at '{DATA_YAML}'. "
            "Make sure you are running train.py from the project root."
        )

    print(f"[INFO] Starting training — epochs={EPOCHS}, imgsz={IMG_SIZE}, batch={BATCH}")
    print(f"[INFO] Dataset config : {DATA_YAML}")
    print(f"[INFO] Results will be saved to: {PROJECT_DIR}/{RUN_NAME}/")

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            project=PROJECT_DIR,
            name=RUN_NAME,
            exist_ok=True,       # Overwrite previous run with the same name
            verbose=True,        # Show per-epoch metrics in the console
            patience=15,         # Early-stopping patience (epochs without improvement)
            save=True,           # Save best and last checkpoints
            plots=True,          # Generate training curve plots
        )
        print("[INFO] Training complete.")
        return results
    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc


def export_best_model(run_name: str = RUN_NAME) -> Path:
    """Copy the Ultralytics best.pt checkpoint to the models/ directory.

    Ultralytics saves the best weights to:
      ``runs/train/<run_name>/weights/best.pt``

    This function copies that file to ``models/best.pt`` so the Streamlit
    app can find it at a fixed, predictable path.

    Args:
        run_name: The training run sub-folder name (matches ``RUN_NAME``).

    Returns:
        The :class:`pathlib.Path` to the copied ``models/best.pt``.

    Raises:
        FileNotFoundError: If the Ultralytics best.pt was not produced.
        OSError: If the file cannot be copied.
    """
    # Source path produced by Ultralytics
    src: Path = Path(PROJECT_DIR) / run_name / "weights" / "best.pt"

    if not src.exists():
        raise FileNotFoundError(
            f"Expected best.pt at '{src}' but it was not found. "
            "Training may not have completed successfully."
        )

    # Ensure destination directory exists
    dest_dir: Path = Path(SAVE_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest: Path = dest_dir / "best.pt"
    shutil.copy2(str(src), str(dest))
    print(f"[INFO] Best model saved to: {dest}")
    return dest


def print_metrics(results: object) -> None:
    """Print key training metrics from the results object.

    Extracts and displays mAP50, mAP50-95, precision, and recall from the
    Ultralytics results object. Gracefully handles missing attributes so
    the script does not crash on older Ultralytics versions.

    Args:
        results: The object returned by :meth:`ultralytics.YOLO.train`.
    """
    print("\n" + "=" * 50)
    print("  TRAINING METRICS (final epoch)")
    print("=" * 50)

    try:
        # Ultralytics stores validation metrics in results.results_dict
        metrics: dict = results.results_dict  # type: ignore[union-attr]
        map50: float = metrics.get("metrics/mAP50(B)", float("nan"))
        map50_95: float = metrics.get("metrics/mAP50-95(B)", float("nan"))
        precision: float = metrics.get("metrics/precision(B)", float("nan"))
        recall: float = metrics.get("metrics/recall(B)", float("nan"))

        print(f"  mAP@0.50       : {map50:.4f}")
        print(f"  mAP@0.50:0.95  : {map50_95:.4f}")
        print(f"  Precision      : {precision:.4f}")
        print(f"  Recall         : {recall:.4f}")
    except AttributeError:
        # Fall back if the results object structure differs
        print("  [WARN] Could not extract detailed metrics from results object.")
        print(f"  Results: {results}")

    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full training pipeline end-to-end.

    Steps:
      1. Load the pretrained YOLOv8n model.
      2. Fine-tune it on the parking dataset.
      3. Copy the best checkpoint to ``models/best.pt``.
      4. Print final validation metrics.
    """
    try:
        # Step 1 — Load base model
        model: YOLO = load_model(MODEL_PATH)

        # Step 2 — Fine-tune on parking data
        results = train_model(model)

        # Step 3 — Export best checkpoint to a fixed location
        export_best_model(run_name=RUN_NAME)

        # Step 4 — Display metrics
        if results is not None:
            print_metrics(results)

        print("[INFO] All done. You can now run:  streamlit run app.py")

    except FileNotFoundError as exc:
        print(f"\n[ERROR] File not found: {exc}")
        print("  → Run 'python utils/convert_annotations.py' first.")
    except RuntimeError as exc:
        print(f"\n[ERROR] Runtime error during training: {exc}")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")


if __name__ == "__main__":
    main()
