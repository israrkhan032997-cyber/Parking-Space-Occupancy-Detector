# 🅿️ ParkVision — Smart Parking Occupancy Detector

---

## Problem Statement

Finding a free parking space in a busy lot is time-consuming and frustrating.
Manual monitoring is labour-intensive and does not scale. ParkVision solves
this by using a fine-tuned YOLOv8 computer-vision model to automatically
detect every parking space in an overhead lot image and classify it as
**free**, **occupied**, or **partially occupied** — in real time.

---

## Real-World Use Case

| Environment | Benefit |
|---|---|
| Shopping malls | Guide drivers to open spaces instantly |
| Office buildings | Monitor utilisation for facilities planning |
| Airports | Real-time capacity dashboards |
| Smart-city systems | Feed occupancy data to navigation apps |
| Security teams | Detect unauthorised long-term parking |

---

## Features

- 🔍 **Automatic space detection** — finds every parking bay in the image
- 🟢🔴🟡 **3-class classification** — free / occupied / partially occupied
- 📊 **Live metric cards** — Free, Occupied, Partial, and Total counts
- 🖼️ **Side-by-side view** — original vs. annotated result
- ⬇️ **One-click download** — save the annotated image as PNG
- ⚙️ **Adjustable thresholds** — confidence & IoU sliders in the sidebar
- 🏷️ **Toggleable labels** — show or hide class name badges on boxes
- 🎨 **Professional dark UI** — clean Streamlit interface with custom CSS
- 🚀 **YOLOv8n backbone** — fast, lightweight, production-ready

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core programming language |
| YOLOv8 (Ultralytics) | Object detection & classification model |
| OpenCV | Image loading, drawing, colour conversion |
| NumPy | Array operations |
| Streamlit | Interactive web application framework |
| Pandas | CSV parsing & data manipulation |
| Pillow | Image I/O in the Streamlit app |
| PyYAML | YOLO dataset configuration (data.yaml) |
| pathlib | Cross-platform file path handling |

---

## Folder Structure

```
Parking-Space-Occupancy-Detector/
├── app.py                         # Streamlit web application
├── train.py                       # YOLOv8 fine-tuning script
├── data.yaml                      # YOLO dataset configuration
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── dataset/
│   ├── annotations.xml            # CVAT polygon annotations (source)
│   ├── parking.csv                # Image-to-mask mapping (source)
│   ├── images/
│   │   ├── 0.png … 32.png         # Raw parking lot images (source)
│   │   ├── train/                 # Train images (populated by convert_annotations.py)
│   │   └── val/                   # Val images   (populated by convert_annotations.py)
│   ├── labels/
│   │   ├── train/                 # YOLO .txt labels for train set
│   │   └── val/                   # YOLO .txt labels for val set
│   ├── boxes/                     # Mask images (source reference)
│   └── raw/                       # Raw data staging area
│
├── models/
│   └── best.pt                    # Trained YOLOv8 weights (produced by train.py)
│
└── utils/
    ├── __init__.py
    ├── convert_annotations.py     # XML → YOLO format converter
    └── visualize.py               # Detection drawing & stats overlay
```

---

## Setup Instructions

### Step 1 — Clone or download the project

```bash
git clone https://github.com/your-username/Parking-Space-Occupancy-Detector.git
cd Parking-Space-Occupancy-Detector
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment to keep dependencies isolated.
> ```bash
> python -m venv .venv
> .venv\Scripts\activate   # Windows
> source .venv/bin/activate # macOS / Linux
> pip install -r requirements.txt
> ```

### Step 3 — Place dataset files in the dataset/ folder

Ensure these files are present (they ship with this repo):

```
dataset/annotations.xml
dataset/parking.csv
dataset/images/0.png … 32.png
dataset/boxes/0.png  … 32.png
```

### Step 4 — Convert annotations to YOLO format

```bash
python utils/convert_annotations.py
```

This script will:
- Parse `annotations.xml` and `parking.csv`
- Convert polygon annotations → YOLO bounding boxes
- Split images 80 / 20 into train and val sets
- Populate `dataset/images/train`, `dataset/images/val`
- Populate `dataset/labels/train`, `dataset/labels/val`
- Print a conversion summary

### Step 5 — Train the model

```bash
python train.py
```

This script will:
- Download `yolov8n.pt` pretrained weights automatically (first run)
- Fine-tune on the parking dataset for 40 epochs
- Save the best checkpoint to `models/best.pt`
- Print mAP, precision, and recall after training

Training typically takes 5–15 minutes on a GPU, or 30–60 minutes on CPU.

### Step 6 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Input / Output

**Input:**  
A single parking lot image in `JPG`, `JPEG`, or `PNG` format uploaded via
the web interface.

**Output:**
- Annotated image with colour-coded bounding boxes drawn on each parking space
- Class label badge + confidence score on every box
- Semi-transparent statistics overlay (top-left corner of result image)
- Summary metric cards: Free · Occupied · Partial · Total
- Downloadable annotated image (`parking_result.png`)

---

## System Workflow

```
User Opens App
      ↓
Upload Parking Lot Image
      ↓
Pre-process Image (BGR conversion)
      ↓
Run YOLOv8 Inference (conf + IoU thresholds from sidebar)
      ↓
Detect & Classify Parking Spaces
      ↓
Draw Colour-Coded Bounding Boxes
      ↓
Compute Occupancy Summary
      ↓
Display Results + Metric Cards
      ↓
Download Annotated Image
```

---

## Application Flow

```
Load Dataset (annotations.xml + parking.csv)
      ↓
Convert Annotations to YOLO Format (convert_annotations.py)
      ↓
Split 80/20 → Train / Val Sets
      ↓
Load Pretrained YOLOv8n Model
      ↓
Fine-tune on Parking Dataset (train.py)
      ↓
Save Best Weights → models/best.pt
      ↓
Load Model in Streamlit App (cached)
      ↓
Detect Parking Spaces on Uploaded Image
      ↓
Display Annotated Result + Statistics
```

---

## Class Labels

| ID | Class Name | Box Color | Description |
|---|---|---|---|
| 0 | `free` | 🟢 Green `(0, 255, 0)` | Parking space is empty |
| 1 | `occupied` | 🔴 Red `(0, 0, 255)` | Parking space has a vehicle |
| 2 | `partially_occupied` | 🟡 Yellow `(0, 255, 255)` | Parking space is partially blocked |

---

## Deliverables

| File | Description |
|---|---|
| `app.py` | Working Streamlit web application |
| `train.py` | YOLOv8 fine-tuning script |
| `utils/convert_annotations.py` | Dataset conversion utility |
| `utils/visualize.py` | Detection drawing & stats overlay |
| `data.yaml` | YOLO dataset configuration |
| `requirements.txt` | Python dependency list |
| `models/best.pt` | Trained YOLOv8 model (produced after training) |

---

## Success Criteria

- ✅ Parking spaces detected correctly on an uploaded image
- ✅ Each space classified as `free`, `occupied`, or `partially_occupied`
- ✅ Colour-coded bounding boxes displayed (Green / Red / Yellow)
- ✅ Metric cards showing correct Free / Occupied / Partial / Total counts
- ✅ Download button produces a valid annotated PNG
- ✅ Professional clean UI with dark theme and custom CSS
- ✅ Model trained, validated, and saved to `models/best.pt`
- ✅ Full error handling — app never crashes silently

---

*Built with ❤️ using Python · YOLOv8 · OpenCV · Streamlit*

## Parking Space Occupancy Detector App

https://parking-space-occupancy-detector-2wqxvj9p8dxhx3gentxk3p.streamlit.app/
🅿️ Smart Parking Occupancy Detector — Upload a parking lot image to detect free, occupied, and partially occupied spaces using YOLOv8 AI model.

