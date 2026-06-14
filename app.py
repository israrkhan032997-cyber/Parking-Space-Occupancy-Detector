"""
app.py
------
ParkVision — Smart Parking Occupancy Detector
Streamlit web application that accepts a parking lot image, runs YOLOv8
inference, and displays colour-coded occupancy bounding boxes alongside
summary metric cards.

Usage
-----
  streamlit run app.py
"""

import io
import os
from pathlib import Path
from typing import Optional

# Force headless OpenCV before any import — required on Streamlit Cloud
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["DISPLAY"] = ""

# Use PIL for all image I/O — avoids libGL dependency entirely
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from utils.visualize import draw_detections, get_summary, overlay_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH: Path = Path("models") / "best.pt"
APP_TITLE: str = "🅿️ ParkVision — Smart Parking Occupancy Detector"
APP_SUBTITLE: str = (
    "Upload a parking lot image to automatically detect and classify each "
    "parking space as **free**, **occupied**, or **partially occupied**."
)
ACCEPTED_IMAGE_TYPES: list = ["jpg", "jpeg", "png"]
DEFAULT_CONFIDENCE: float = 0.50
DEFAULT_IOU: float = 0.45
CONFIDENCE_STEP: float = 0.05
RESULT_FILENAME: str = "parking_result.png"

# CSS for dark metric cards with rounded corners and subtle shadows
CUSTOM_CSS: str = """
<style>
    /* ---- Global page tweaks ---- */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ---- Metric card container ---- */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
    }

    /* ---- Metric label ---- */
    div[data-testid="metric-container"] label {
        color: #a0a0c0 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ---- Metric value ---- */
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f0f0ff !important;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        line-height: 1.15;
    }

    /* ---- Sidebar styling ---- */
    section[data-testid="stSidebar"] {
        background: #12121e;
        border-right: 1px solid #2a2a3e;
    }
    section[data-testid="stSidebar"] .stSlider label {
        color: #c0c0d8 !important;
    }

    /* ---- Info / placeholder box ---- */
    .upload-placeholder {
        border: 2px dashed #3a3a5c;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        color: #7070a0;
        font-size: 1rem;
    }

    /* ---- Section divider ---- */
    hr.section-divider {
        border: none;
        border-top: 1px solid #2a2a3e;
        margin: 1.5rem 0;
    }
</style>
"""


# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ParkVision",
    layout="wide",
    page_icon="🅿️",
    initial_sidebar_state="expanded",
)

# Inject custom styles
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached so it is only loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading YOLOv8 model …")
def load_model(model_path: Path) -> YOLO:
    """Load the trained YOLOv8 model from *model_path*.

    Uses ``@st.cache_resource`` so the model is loaded once and shared
    across all Streamlit re-runs within the same session.

    Args:
        model_path: Absolute or project-relative path to the ``best.pt`` file.

    Returns:
        An initialised :class:`ultralytics.YOLO` model ready for inference.

    Raises:
        FileNotFoundError: Raised (and caught by the caller) when the weights
                           file does not exist.
        RuntimeError: Raised if the model cannot be initialised.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Run train.py first to generate the model."
        )
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a BGR NumPy array for OpenCV/YOLO.

    Args:
        pil_image: A :class:`PIL.Image.Image` in any mode.

    Returns:
        A ``uint8`` BGR NumPy array.
    """
    # Convert to RGB first, then flip channel order to BGR
    rgb_array: np.ndarray = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    bgr_array: np.ndarray = rgb_array[:, :, ::-1].copy()  # RGB → BGR without cv2
    return bgr_array


def bgr_to_bytes(bgr_image: np.ndarray, ext: str = ".png") -> bytes:
    """Encode a BGR NumPy array to raw image bytes using PIL.

    Args:
        bgr_image: A ``uint8`` BGR NumPy array.
        ext: File extension indicating the target format (e.g. ``".png"``).

    Returns:
        The encoded image as a :class:`bytes` object.
    """
    # Flip BGR → RGB for PIL
    rgb_array: np.ndarray = bgr_image[:, :, ::-1].copy()
    pil_img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    fmt = ext.lstrip(".").upper()
    fmt = "JPEG" if fmt == "JPG" else fmt
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def run_inference(
    model: YOLO,
    bgr_image: np.ndarray,
    confidence: float,
    iou: float,
) -> object:
    """Run YOLOv8 inference on a BGR image array.

    Args:
        model: The loaded :class:`ultralytics.YOLO` model.
        bgr_image: Input image as a ``uint8`` BGR NumPy array.
        confidence: Minimum detection confidence threshold (0–1).
        iou: Non-maximum suppression IoU threshold (0–1).

    Returns:
        The raw Ultralytics ``Results`` object from ``model()``.

    Raises:
        RuntimeError: If inference fails for any reason.
    """
    try:
        results = model(bgr_image, conf=confidence, iou=iou, verbose=False)
        return results
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(model_path: Path) -> tuple:
    """Render the sidebar controls and return the selected parameter values.

    Args:
        model_path: Path used to determine model status display.

    Returns:
        A tuple ``(confidence, iou, show_labels)`` with the current widget
        values selected by the user.
    """
    with st.sidebar:
        st.title("⚙️ Detection Settings")
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Confidence threshold slider
        confidence: float = st.slider(
            label="Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_CONFIDENCE,
            step=CONFIDENCE_STEP,
            help="Only detections with confidence ≥ this value will be shown.",
        )

        # IoU threshold slider
        iou: float = st.slider(
            label="IoU Threshold (NMS)",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_IOU,
            step=CONFIDENCE_STEP,
            help="Non-maximum suppression overlap threshold.",
        )

        # Show / hide label badges on boxes
        show_labels: bool = st.checkbox(
            label="Show Bounding Box Labels",
            value=True,
            help="Toggle class name and confidence text on each box.",
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("🤖 Model Status")

        # Display model path and load status
        st.caption(f"Path: `{model_path}`")
        if model_path.exists():
            st.success("✅ Model loaded", icon="✅")
        else:
            st.error("❌ Model not found", icon="❌")
            st.caption("Run `python train.py` to train and save the model.")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.subheader("🎨 Legend")
        st.markdown("🟢 **Free** — empty parking space")
        st.markdown("🔴 **Occupied** — car present")
        st.markdown("🟡 **Partial** — partially occupied")

    return confidence, iou, show_labels


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ParkVision Streamlit application.

    Orchestrates model loading, sidebar rendering, file upload handling,
    inference execution, result display, and the download button.
    """
    # -- Header ------------------------------------------------------------
    st.title(APP_TITLE)
    st.markdown(APP_SUBTITLE)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # -- Sidebar -----------------------------------------------------------
    confidence, iou, show_labels = render_sidebar(MODEL_PATH)

    # -- Load model (halt app if model is missing) -------------------------
    try:
        model: YOLO = load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        st.error(
            f"⚠️ **Model not found.**\n\n{exc}\n\n"
            "Run `python train.py` first to generate the model.",
            icon="🚨",
        )
        st.stop()
    except RuntimeError as exc:
        st.error(f"⚠️ **Model loading failed:** {exc}", icon="🚨")
        st.stop()

    # -- File uploader -----------------------------------------------------
    uploaded_file: Optional[object] = st.file_uploader(
        label="📂 Upload Parking Lot Image",
        type=ACCEPTED_IMAGE_TYPES,
        help="Supported formats: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        # Show a friendly placeholder when no file has been uploaded yet
        st.markdown(
            """
            <div class="upload-placeholder">
                <h3>📷 No image uploaded yet</h3>
                <p>Use the uploader above to select a parking lot image.<br>
                Supported formats: <strong>JPG, JPEG, PNG</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # -- Process uploaded image --------------------------------------------
    try:
        # Decode the uploaded file as a PIL image
        pil_image: Image.Image = Image.open(uploaded_file)

        # Convert to BGR NumPy array for YOLO / OpenCV
        bgr_image: np.ndarray = pil_to_bgr(pil_image)

    except Exception as exc:
        st.error(f"⚠️ **Failed to read the uploaded image:** {exc}", icon="🚨")
        return

    # -- Run inference -----------------------------------------------------
    try:
        with st.spinner("🔍 Running detection …"):
            results = run_inference(model, bgr_image, confidence, iou)
    except RuntimeError as exc:
        st.error(f"⚠️ **Inference error:** {exc}", icon="🚨")
        return

    # -- Build annotated image and summary ---------------------------------
    try:
        annotated_bgr: np.ndarray = draw_detections(bgr_image, results, show_labels)
        annotated_bgr = overlay_stats(
            annotated_bgr,
            get_summary(results),
        )
        summary: dict = get_summary(results)
    except Exception as exc:
        st.error(f"⚠️ **Visualisation error:** {exc}", icon="🚨")
        return

    # -- Side-by-side image display ----------------------------------------
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_orig, col_result = st.columns(2, gap="medium")

    with col_orig:
        st.subheader("🖼️ Original Image")
        st.image(pil_image, use_container_width=True, caption="Original Image")

    with col_result:
        st.subheader("🔍 Detection Result")
        # Convert annotated BGR back to RGB for Streamlit display (no cv2 needed)
        annotated_rgb: np.ndarray = annotated_bgr[:, :, ::-1].copy()
        st.image(annotated_rgb, use_container_width=True, caption="Detection Result")

    # -- Metrics row -------------------------------------------------------
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.subheader("📊 Occupancy Summary")

    m1, m2, m3, m4 = st.columns(4, gap="medium")

    with m1:
        st.metric(label="🟢 Free Spaces", value=summary.get("free", 0))
    with m2:
        st.metric(label="🔴 Occupied", value=summary.get("occupied", 0))
    with m3:
        st.metric(label="🟡 Partial", value=summary.get("partially_occupied", 0))
    with m4:
        st.metric(label="📊 Total Spaces", value=summary.get("total", 0))

    # -- Download button ---------------------------------------------------
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    try:
        result_bytes: bytes = bgr_to_bytes(annotated_bgr, ext=".png")
        st.download_button(
            label="⬇️ Download Result Image",
            data=result_bytes,
            file_name=RESULT_FILENAME,
            mime="image/png",
            help="Save the annotated detection result as a PNG file.",
            use_container_width=False,
        )
    except RuntimeError as exc:
        st.warning(f"Could not prepare download: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
