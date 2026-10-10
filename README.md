# VisDrone Object Detection System

> **Course:** Computer Vision (AI447) — Spring 2025-2026  
> **University:** Jordan University of Science and Technology

---

## 1. Project Overview

A deep learning-based object detection and tracking system trained on the **VisDrone dataset**. Detects 10 object classes from drone imagery using three architectures, served through a FastAPI web interface.

| | |
|---|---|
| **Dataset** | VisDrone (drone aerial imagery) |
| **Classes** | pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor |
| **Models** | YOLOv8m · Faster R-CNN · SSD MobileNetV3 |
| **Bonus** | DeepSORT multi-object tracking on video |
| **Interface** | FastAPI web app — runs locally |

---

## 2. Requirements

**System:**
- Python 3.9 or higher
- Windows 10/11 (or Linux/macOS)
- GPU recommended (NVIDIA + CUDA) — CPU works but slower
- At least 4 GB RAM (8 GB recommended)

**Libraries:**
```
fastapi
uvicorn[standard]
python-multipart
torch
torchvision
ultralytics
opencv-python
Pillow
numpy
matplotlib
deep-sort-realtime
```

---

## 3. Project Structure

```
CV project\
├── app.py                        ← Main FastAPI application
├── requirements.txt              ← Python dependencies
├── outputs\                      ← Processed videos saved here
└── visdrone_models\              ← Located at C:\Users\USER\Desktop\visdrone_models
    ├── yolov8m_best.pt
    ├── fasterrcnn_best.pth
    ├── ssd_mobilenet_best.pth
    └── config.json
```

---

## 4. Installation & Setup

**Step 1 — Open terminal in VS Code**
```bash
cd "C:\Users\USER\CV project"
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install fastapi uvicorn[standard] python-multipart torch torchvision ultralytics opencv-python Pillow numpy matplotlib deep-sort-realtime
```

**Step 3 — Verify model files exist**
```
C:\Users\USER\Desktop\visdrone_models\yolov8m_best.pt
C:\Users\USER\Desktop\visdrone_models\fasterrcnn_best.pth
C:\Users\USER\Desktop\visdrone_models\ssd_mobilenet_best.pth
C:\Users\USER\Desktop\visdrone_models\config.json
```

---

## 5. How to Run

```bash
uvicorn app:api --reload --port 8000
```

Wait for:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Then open your browser at:
```
http://localhost:8000
```

To stop the server press `Ctrl + C`.

---

## 6. How to Use the Interface

### Image Detection Tab

1. **Upload an image** — drag and drop or click to browse (JPG / PNG / WEBP)
2. **Select a model** — choose from the dropdown (see Section 7)
3. **Set confidence threshold** — drag the slider (default: 0.35)
4. **Click RUN DETECTION**
5. **View results** — annotated image with bounding boxes, labels, confidence scores, and inference time

### Video Tracking Tab

1. **Upload a video** — MP4 / AVI / MOV
2. **Processing** — progress bar shows status while frames are processed
3. **Output** — video with bounding boxes and persistent track IDs per object
4. **Stats** — total frames, unique track IDs, total inference time

### Input & Output Formats

| | Image Detection | Video Tracking |
|---|---|---|
| **Input** | JPG / PNG / WEBP | MP4 / AVI / MOV |
| **Output** | Annotated image in browser | Processed video saved to `outputs\` |
| **Extra info** | Detection list + confidence scores | Track IDs + frame stats |

---

## 7. Models Description

| Model | Speed | Accuracy | Best Used For |
|---|---|---|---|
| **YOLOv8m** | Fastest | High | Real-time detection, large images |
| **Faster R-CNN** | Moderate | Highest | High-accuracy scenarios, small objects |
| **SSD MobileNetV3** | Fast | Moderate | Low-resource environments |

All models detect these 10 classes:  
`pedestrian` · `people` · `bicycle` · `car` · `van` · `truck` · `tricycle` · `awning-tricycle` · `bus` · `motor`

---

## 8. Limitations & Notes

- Without a GPU, inference is slower — Faster R-CNN may take 5–10 seconds per image on CPU
- Models are trained on drone footage; performance may be lower on ground-level images
- Very large images (4K+) may slow down processing
- Video tracking processes the full video before displaying results; long videos take more time
- Processed videos in `outputs\` are not deleted automatically

---

## 9. Common Issues

| Problem | Solution |
|---|---|
| Model not found error | Check all 4 files exist in `C:\Users\USER\Desktop\visdrone_models\` |
| Port 8000 already in use | Run `uvicorn app:api --port 8001` and open `localhost:8001` |
| pip install fails | Run `pip install --upgrade pip` then retry |
| No detections showing | Lower the confidence threshold (try 0.15 or 0.20) |
| Video processing very slow | Use a shorter clip or switch to YOLOv8m |

---

*Computer Vision Project — AI447 | Spring 2025-2026 | Jordan University of Science and Technology*
