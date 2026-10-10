"""
VisDrone Object Detection — Local FastAPI App
=============================================
Run locally on VS Code after downloading weights from Colab.

Setup:
    pip install fastapi uvicorn[standard] python-multipart
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics opencv-python Pillow deep-sort-realtime

Run:
    uvicorn app:api --reload --port 8000 --reload-exclude "outputs/*"

Then open:  http://localhost:8000
"""

import io, json, time, base64, functools, threading, uuid, shutil
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

import cv2
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────
MODELS_DIR   = Path(r"C:\Users\USER\Desktop\visdrone_models\visdrone_models")
CONFIG_PATH  = MODELS_DIR / "config.json"
OUTPUTS_DIR  = Path(__file__).parent / "outputs"
UPLOADS_DIR  = OUTPUTS_DIR / "_uploads"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Load class names from config.json
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

CLASSES     = CFG["classes"]
NUM_CLASSES = CFG["num_classes"]
WEIGHTS     = CFG["weights"]

print(f"Device : {DEVICE}")
print(f"Classes: {CLASSES}")

# ── Model loaders ─────────────────────────────────────────────────
def load_yolo():
    path = MODELS_DIR / WEIGHTS["yolo"]
    print(f"Loading YOLOv8m from {path} ...")
    return YOLO(str(path))

def load_frcnn():
    path = MODELS_DIR / WEIGHTS["frcnn"]
    print(f"Loading Faster R-CNN from {path} ...")
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_ssd():
    path = MODELS_DIR / WEIGHTS["ssd"]
    print(f"Loading SSD MobileNetV3 from {path} ...")
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    in_channels = []
    for c in model.head.classification_head.module_list:
        if hasattr(c, "in_channels"):
            in_channels.append(c.in_channels)
        else:
            for layer in c.modules():
                if isinstance(layer, nn.Conv2d):
                    in_channels.append(layer.in_channels)
                    break
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, NUM_CLASSES,
        norm_layer=functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.03),
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# Load all models at startup
print("\n⏳ Loading models (this may take ~30s first time)...")
YOLO_MODEL  = load_yolo()
FRCNN_MODEL = load_frcnn()
SSD_MODEL   = load_ssd()
print("✅ All models loaded!\n")

# ── Inference helpers ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CMAP = plt.colormaps.get_cmap("tab10").resampled(len(CLASSES))

def _color(cls_name):
    idx = CLASSES.index(cls_name) if cls_name in CLASSES else 0
    r, g, b, _ = CMAP(idx)
    return int(r * 255), int(g * 255), int(b * 255)

def run_yolo(img_np, conf):
    results = YOLO_MODEL.predict(img_np, conf=conf, verbose=False)
    boxes, scores, labels = [], [], []
    for r in results:
        for box in r.boxes:
            boxes.append([float(x) for x in box.xyxy[0].tolist()])
            scores.append(float(box.conf[0]))
            labels.append(CLASSES[int(box.cls[0])])
    return boxes, scores, labels

def run_torchvision(model, img_np, conf):
    tensor = TF.to_tensor(Image.fromarray(img_np)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)[0]
    mask   = out["scores"] >= conf
    boxes  = out["boxes"][mask].cpu().numpy().tolist()
    scores = out["scores"][mask].cpu().numpy().tolist()
    labels = [
        CLASSES[int(l) - 1] if 1 <= int(l) <= len(CLASSES) else "unknown"
        for l in out["labels"][mask].cpu().numpy()
    ]
    return boxes, scores, labels

def annotate(img_np, boxes, scores, labels):
    img = img_np.copy()
    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        color = _color(label)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (int(x1), int(y1) - th - 4),
                      (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(img, text, (int(x1), int(y1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def img_to_b64(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

# ── Video tracking job system ─────────────────────────────────────
JOBS = {}
JOBS_LOCK = threading.Lock()
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def _make_writer(path: Path, fps: float, size):
    """Try H.264 (avc1) first for browser compat; fall back to mp4v."""
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        w = cv2.VideoWriter(str(path), fourcc, fps, size)
        if w.isOpened():
            return w, fourcc_str
        w.release()
    raise RuntimeError("Could not open VideoWriter with avc1 or mp4v")

def process_video_job(job_id: str, input_path: Path, output_path: Path, conf: float):
    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "processing"

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_path.name}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer, codec = _make_writer(output_path, fps, (width, height))

        tracker = DeepSort(max_age=30, n_init=2)
        track_ids = set()
        t0 = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = YOLO_MODEL.predict(frame, conf=conf, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    score   = float(box.conf[0])
                    cls_idx = int(box.cls[0])
                    cls_name = CLASSES[cls_idx] if 0 <= cls_idx < len(CLASSES) else "obj"
                    detections.append(([x1, y1, x2 - x1, y2 - y1], score, cls_name))

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                track_ids.add(tid)
                x1, y1, x2, y2 = [int(v) for v in track.to_ltrb()]
                cls_name = track.get_det_class() or "obj"
                color = _color(cls_name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {cls_name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 5 == 0 or frame_idx == total:
                with JOBS_LOCK:
                    JOBS[job_id]["frame"] = frame_idx
                    JOBS[job_id]["total_frames"] = total
                    JOBS[job_id]["progress"] = (frame_idx / total) if total else 0.0
                    JOBS[job_id]["unique_tracks"] = len(track_ids)

        cap.release()
        writer.release()

        elapsed = round(time.time() - t0, 1)
        with JOBS_LOCK:
            JOBS[job_id].update({
                "status": "done",
                "frame": frame_idx,
                "total_frames": frame_idx,
                "progress": 1.0,
                "unique_tracks": len(track_ids),
                "elapsed_s": elapsed,
                "fps_processed": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
                "codec": codec,
                "filename": output_path.name,
                "video_url": f"/outputs/{output_path.name}",
            })

        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass

    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)

# ── FastAPI ───────────────────────────────────────────────────────
api = FastAPI(title="VisDrone Detection API", version="1.1")

# ── UI ────────────────────────────────────────────────────────────
UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>VisDrone — Object Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:#0a0c10;--surface:#111620;--card:#161c28;--border:#1e2a3a;
    --accent:#00e5ff;--accent2:#ff6b35;--text:#e0eaf8;--muted:#5a6a80;
    --green:#39ff87;--radius:8px;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;
       min-height:100vh;display:flex;flex-direction:column;align-items:center}

  header{width:100%;background:linear-gradient(90deg,#0a0c10,#0d1520 50%,#0a0c10);
         border-bottom:1px solid var(--border);padding:18px 40px;
         display:flex;align-items:center;gap:18px}
  .logo-box{width:44px;height:44px;background:var(--accent);border-radius:6px;
             display:grid;place-items:center;font-size:22px}
  .header-text h1{font-size:22px;font-weight:700;letter-spacing:2px;
                  color:var(--accent);text-transform:uppercase}
  .header-text p{font-size:12px;color:var(--muted);font-family:'JetBrains Mono',monospace;letter-spacing:1px}
  .status-dot{margin-left:auto;display:flex;align-items:center;gap:8px;
              font-size:12px;color:var(--green);font-family:'JetBrains Mono',monospace}
  .status-dot::before{content:'';width:8px;height:8px;border-radius:50%;
                      background:var(--green);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

  .tab-bar{width:100%;max-width:1100px;padding:24px 20px 0;display:flex;gap:4px;border-bottom:1px solid var(--border)}
  .tab-btn{background:transparent;color:var(--muted);border:none;
           padding:14px 22px;font-family:'Rajdhani',sans-serif;font-size:15px;
           font-weight:600;letter-spacing:1.5px;text-transform:uppercase;
           cursor:pointer;border-bottom:2px solid transparent;
           transition:color .2s,border-color .2s}
  .tab-btn:hover{color:var(--text)}
  .tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}

  main{width:100%;max-width:1100px;padding:24px 20px 32px;
       display:grid;grid-template-columns:340px 1fr;gap:24px}
  .tab-panel{display:none}
  .tab-panel.active{display:contents}

  .card{background:var(--card);border:1px solid var(--border);
        border-radius:var(--radius);padding:24px}
  .card-title{font-size:11px;letter-spacing:2px;text-transform:uppercase;
              color:var(--accent);font-family:'JetBrains Mono',monospace;
              margin-bottom:20px;display:flex;align-items:center;gap:8px}
  .card-title::before{content:'';display:block;width:3px;height:14px;
                      background:var(--accent);border-radius:2px}

  .upload-zone{border:2px dashed var(--border);border-radius:var(--radius);
               padding:32px 16px;text-align:center;cursor:pointer;
               transition:border-color .2s,background .2s;
               position:relative;margin-bottom:20px}
  .upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);
                                            background:rgba(0,229,255,.04)}
  .upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
  .upload-icon{font-size:36px;margin-bottom:8px}
  .upload-hint{font-size:13px;color:var(--muted)}
  .preview-wrap{margin-bottom:20px;display:none}
  .preview-wrap img,.preview-wrap video{width:100%;border-radius:var(--radius);border:1px solid var(--border)}

  label.ctrl-label{display:block;font-size:11px;letter-spacing:1.5px;
                   text-transform:uppercase;color:var(--muted);margin-bottom:6px}
  select,input[type=range]{width:100%;background:var(--surface);
    border:1px solid var(--border);color:var(--text);border-radius:var(--radius);
    padding:10px 12px;font-family:'Rajdhani',sans-serif;font-size:15px;
    margin-bottom:16px;appearance:none;cursor:pointer}
  select:focus{outline:1px solid var(--accent)}
  .range-row{display:flex;justify-content:space-between;font-size:12px;
             color:var(--muted);margin-top:-12px;margin-bottom:16px}
  .conf-val{color:var(--accent);font-weight:700}

  .run-btn{width:100%;padding:14px;background:var(--accent);color:#000;
           border:none;border-radius:var(--radius);font-family:'Rajdhani',sans-serif;
           font-size:16px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
           cursor:pointer;transition:opacity .2s,transform .1s}
  .run-btn:hover{opacity:.85}
  .run-btn:active{transform:scale(.98)}
  .run-btn:disabled{opacity:.4;cursor:not-allowed}

  .result-panel{display:flex;flex-direction:column;gap:20px}
  .result-img-wrap{background:var(--surface);border:1px solid var(--border);
                   border-radius:var(--radius);overflow:hidden;min-height:300px;
                   display:grid;place-items:center}
  .result-img-wrap img,.result-img-wrap video{width:100%;display:block}
  .placeholder-txt{color:var(--muted);font-family:'JetBrains Mono',monospace;
                   font-size:13px;text-align:center;padding:24px}

  .stats-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
  .stats-row.four{grid-template-columns:repeat(4,1fr)}
  .stat-box{background:var(--surface);border:1px solid var(--border);
            border-radius:var(--radius);padding:14px;text-align:center}
  .stat-val{font-size:24px;font-weight:700;color:var(--accent);
            font-family:'JetBrains Mono',monospace}
  .stat-lbl{font-size:11px;color:var(--muted);letter-spacing:1px}

  .det-list{list-style:none;display:flex;flex-direction:column;gap:6px;
            max-height:220px;overflow-y:auto}
  .det-item{background:var(--surface);border:1px solid var(--border);
            border-radius:6px;padding:8px 12px;display:flex;align-items:center;
            justify-content:space-between;font-size:14px}
  .det-cls{font-weight:600;color:var(--text)}
  .det-conf{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--accent2)}
  .dot{width:8px;height:8px;border-radius:50%;margin-right:8px;flex-shrink:0}

  .spinner-wrap{display:none;flex-direction:column;align-items:center;gap:12px;padding:40px}
  .spinner{width:40px;height:40px;border:3px solid var(--border);
           border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  .progress-wrap{display:none;flex-direction:column;gap:10px;padding:30px;width:100%;max-width:480px}
  .progress-bar{width:100%;height:14px;background:var(--surface);
                border:1px solid var(--border);border-radius:7px;overflow:hidden}
  .progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));
                 width:0%;transition:width .3s}
  .progress-info{display:flex;justify-content:space-between;font-size:12px;
                 color:var(--muted);font-family:'JetBrains Mono',monospace}
  .progress-pct{color:var(--accent);font-weight:700}

  .error-box{background:rgba(255,50,50,.08);border:1px solid rgba(255,80,80,.3);
             border-radius:var(--radius);padding:14px 18px;color:#ff6b6b;
             font-size:14px;display:none}

  @media(max-width:780px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div class="logo-box">🎯</div>
  <div class="header-text">
    <h1>VisDrone Detection</h1>
    <p>AI447 — Computer Vision · Running Locally</p>
  </div>
  <div class="status-dot">LOCAL · ONLINE</div>
</header>

<nav class="tab-bar">
  <button class="tab-btn active" data-tab="image">🖼 Image Detection</button>
  <button class="tab-btn" data-tab="video">🎬 Video Tracking</button>
</nav>

<main>
  <!-- ===== IMAGE TAB ===== -->
  <div class="tab-panel active" id="panel-image">
    <aside>
      <div class="card">
        <div class="card-title">Input</div>
        <div class="upload-zone" id="img-drop">
          <input type="file" id="img-input" accept="image/*"/>
          <div class="upload-icon">📁</div>
          <div style="font-size:15px;font-weight:600;margin-bottom:4px">Drop image here</div>
          <div class="upload-hint">or click to browse · JPG / PNG / WEBP</div>
        </div>
        <div class="preview-wrap" id="img-preview-wrap"><img id="img-preview" src="" alt=""/></div>

        <label class="ctrl-label">Model</label>
        <select id="model-sel">
          <option value="yolo">YOLOv8m  (fastest)</option>
          <option value="frcnn">Faster R-CNN  (balanced)</option>
          <option value="ssd">SSD MobileNetV3  (lightweight)</option>
        </select>

        <label class="ctrl-label">Confidence threshold</label>
        <input type="range" id="img-conf" min="5" max="95" step="5" value="35"/>
        <div class="range-row">
          <span>0.05</span>
          <span>Conf: <span class="conf-val" id="img-conf-val">0.35</span></span>
          <span>0.95</span>
        </div>

        <button class="run-btn" id="img-run" disabled>▶ RUN DETECTION</button>
        <div class="error-box" id="img-error"></div>
      </div>
    </aside>

    <section class="result-panel">
      <div class="card" style="padding:0;overflow:hidden">
        <div class="result-img-wrap" id="img-result">
          <div class="placeholder-txt">
            Upload an image and click<br/>
            <strong style="color:var(--accent)">RUN DETECTION</strong><br/>
            to see results here
          </div>
          <div class="spinner-wrap" id="img-spinner">
            <div class="spinner"></div>
            <span style="color:var(--muted);font-size:13px;font-family:'JetBrains Mono',monospace">Running inference…</span>
          </div>
        </div>
      </div>

      <div class="stats-row" id="img-stats" style="display:none">
        <div class="stat-box"><div class="stat-val" id="stat-dets">—</div><div class="stat-lbl">DETECTIONS</div></div>
        <div class="stat-box"><div class="stat-val" id="stat-ms">—</div><div class="stat-lbl">INFERENCE (ms)</div></div>
        <div class="stat-box"><div class="stat-val" id="stat-model">—</div><div class="stat-lbl">MODEL</div></div>
      </div>

      <div class="card" id="det-card" style="display:none">
        <div class="card-title">Detections</div>
        <ul class="det-list" id="det-list"></ul>
      </div>
    </section>
  </div>

  <!-- ===== VIDEO TAB ===== -->
  <div class="tab-panel" id="panel-video">
    <aside>
      <div class="card">
        <div class="card-title">Input · YOLOv8 + DeepSORT</div>
        <div class="upload-zone" id="vid-drop">
          <input type="file" id="vid-input" accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/webm"/>
          <div class="upload-icon">🎞</div>
          <div style="font-size:15px;font-weight:600;margin-bottom:4px">Drop video here</div>
          <div class="upload-hint">or click to browse · MP4 / AVI / MOV</div>
        </div>
        <div class="preview-wrap" id="vid-preview-wrap"><video id="vid-preview" controls muted></video></div>

        <label class="ctrl-label">Detector</label>
        <select disabled><option>YOLOv8m + DeepSORT</option></select>

        <label class="ctrl-label">Confidence threshold</label>
        <input type="range" id="vid-conf" min="5" max="95" step="5" value="35"/>
        <div class="range-row">
          <span>0.05</span>
          <span>Conf: <span class="conf-val" id="vid-conf-val">0.35</span></span>
          <span>0.95</span>
        </div>

        <button class="run-btn" id="vid-run" disabled>▶ RUN TRACKING</button>
        <div class="error-box" id="vid-error"></div>
      </div>
    </aside>

    <section class="result-panel">
      <div class="card" style="padding:0;overflow:hidden">
        <div class="result-img-wrap" id="vid-result">
          <div class="placeholder-txt">
            Upload a video and click<br/>
            <strong style="color:var(--accent)">RUN TRACKING</strong><br/>
            to process with YOLOv8 + DeepSORT
          </div>
          <div class="progress-wrap" id="vid-progress">
            <div style="text-align:center;font-size:14px;color:var(--text);font-family:'JetBrains Mono',monospace">
              Processing video… <span class="progress-pct" id="vid-pct">0%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" id="vid-fill"></div></div>
            <div class="progress-info">
              <span id="vid-frames">frame 0 / 0</span>
              <span id="vid-tracks">tracks: 0</span>
            </div>
          </div>
        </div>
      </div>

      <div class="stats-row four" id="vid-stats" style="display:none">
        <div class="stat-box"><div class="stat-val" id="stat-frames">—</div><div class="stat-lbl">FRAMES</div></div>
        <div class="stat-box"><div class="stat-val" id="stat-ids">—</div><div class="stat-lbl">UNIQUE IDs</div></div>
        <div class="stat-box"><div class="stat-val" id="stat-elapsed">—</div><div class="stat-lbl">ELAPSED (s)</div></div>
        <div class="stat-box"><div class="stat-val" id="stat-fps">—</div><div class="stat-lbl">PROC FPS</div></div>
      </div>
    </section>
  </div>
</main>

<script>
const CLASSES=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor'];
const COLORS=['#e74c3c','#e67e22','#f1c40f','#2ecc71','#1abc9c','#3498db','#9b59b6','#e91e63','#00bcd4','#8bc34a'];

// ── Tab switching ──
document.querySelectorAll('.tab-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-'+btn.dataset.tab).classList.add('active');
  });
});

// ── Generic file-drop wiring ──
function wireDrop(zoneId, inputId, onFile){
  const zone=document.getElementById(zoneId), inp=document.getElementById(inputId);
  inp.addEventListener('change',()=>onFile(inp.files[0]));
  zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('drag-over')});
  zone.addEventListener('dragleave',()=>zone.classList.remove('drag-over'));
  zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('drag-over');onFile(e.dataTransfer.files[0])});
}

// ── IMAGE DETECTION ──
const imgConf=document.getElementById('img-conf'), imgConfVal=document.getElementById('img-conf-val');
imgConf.addEventListener('input',()=>imgConfVal.textContent=(imgConf.value/100).toFixed(2));
let imgFile=null;
const imgPreview=document.getElementById('img-preview'),
      imgPreviewWrap=document.getElementById('img-preview-wrap'),
      imgRun=document.getElementById('img-run'),
      imgErr=document.getElementById('img-error'),
      imgResult=document.getElementById('img-result'),
      imgSpinner=document.getElementById('img-spinner'),
      imgStats=document.getElementById('img-stats'),
      detCard=document.getElementById('det-card'),
      detList=document.getElementById('det-list');

wireDrop('img-drop','img-input',f=>{
  if(!f)return;
  imgFile=f;
  imgPreview.src=URL.createObjectURL(f);
  imgPreviewWrap.style.display='block';
  imgRun.disabled=false;
  imgErr.style.display='none';
});

imgRun.addEventListener('click',async()=>{
  if(!imgFile)return;
  const ph=imgResult.querySelector('.placeholder-txt'); if(ph)ph.style.display='none';
  imgResult.querySelectorAll('img').forEach(e=>e.remove());
  imgSpinner.style.display='flex';
  imgStats.style.display='none';
  detCard.style.display='none';
  imgErr.style.display='none';
  imgRun.disabled=true;

  const form=new FormData();
  form.append('file',imgFile);
  form.append('model',document.getElementById('model-sel').value);
  form.append('conf',(imgConf.value/100).toFixed(2));

  try{
    const res=await fetch('/detect',{method:'POST',body:form});
    const data=await res.json();
    imgSpinner.style.display='none';
    if(data.error){imgErr.textContent='⚠ '+data.error;imgErr.style.display='block';imgRun.disabled=false;return}

    const el=document.createElement('img');
    el.src='data:image/jpeg;base64,'+data.image_b64;
    imgResult.appendChild(el);

    imgStats.style.display='grid';
    document.getElementById('stat-dets').textContent=data.count;
    document.getElementById('stat-ms').textContent=data.inference_ms;
    const sel=document.getElementById('model-sel');
    document.getElementById('stat-model').textContent=sel.options[sel.selectedIndex].text.split(' ')[0];

    detList.innerHTML='';
    if(!data.detections.length){
      detList.innerHTML='<li class="det-item" style="color:var(--muted)">No detections above threshold.</li>';
    } else {
      data.detections.forEach(d=>{
        const idx=CLASSES.indexOf(d.label);
        const col=idx>=0?COLORS[idx]:'#aaa';
        const li=document.createElement('li');
        li.className='det-item';
        li.innerHTML=`<div style="display:flex;align-items:center"><div class="dot" style="background:${col}"></div><span class="det-cls">${d.label}</span></div><span class="det-conf">${d.score}</span>`;
        detList.appendChild(li);
      });
    }
    detCard.style.display='block';
  }catch(err){
    imgSpinner.style.display='none';
    imgErr.textContent='⚠ Request failed: '+err.message;
    imgErr.style.display='block';
  }
  imgRun.disabled=false;
});

// ── VIDEO TRACKING ──
const vidConf=document.getElementById('vid-conf'), vidConfVal=document.getElementById('vid-conf-val');
vidConf.addEventListener('input',()=>vidConfVal.textContent=(vidConf.value/100).toFixed(2));
let vidFile=null, pollTimer=null;
const vidPreview=document.getElementById('vid-preview'),
      vidPreviewWrap=document.getElementById('vid-preview-wrap'),
      vidRun=document.getElementById('vid-run'),
      vidErr=document.getElementById('vid-error'),
      vidResult=document.getElementById('vid-result'),
      vidProgress=document.getElementById('vid-progress'),
      vidFill=document.getElementById('vid-fill'),
      vidPct=document.getElementById('vid-pct'),
      vidFrames=document.getElementById('vid-frames'),
      vidTracks=document.getElementById('vid-tracks'),
      vidStats=document.getElementById('vid-stats');

wireDrop('vid-drop','vid-input',f=>{
  if(!f)return;
  vidFile=f;
  vidPreview.src=URL.createObjectURL(f);
  vidPreviewWrap.style.display='block';
  vidRun.disabled=false;
  vidErr.style.display='none';
});

function clearVidResult(){
  const ph=vidResult.querySelector('.placeholder-txt'); if(ph)ph.style.display='none';
  vidResult.querySelectorAll('video').forEach(v=>v.remove());
}

vidRun.addEventListener('click',async()=>{
  if(!vidFile)return;
  clearVidResult();
  vidProgress.style.display='flex';
  vidFill.style.width='0%';
  vidPct.textContent='0%';
  vidFrames.textContent='frame 0 / 0';
  vidTracks.textContent='tracks: 0';
  vidStats.style.display='none';
  vidErr.style.display='none';
  vidRun.disabled=true;

  const form=new FormData();
  form.append('file',vidFile);
  form.append('conf',(vidConf.value/100).toFixed(2));

  try{
    const res=await fetch('/track',{method:'POST',body:form});
    const data=await res.json();
    if(!res.ok || data.error){throw new Error(data.error||('HTTP '+res.status))}
    pollJob(data.job_id);
  }catch(err){
    vidProgress.style.display='none';
    vidErr.textContent='⚠ '+err.message;
    vidErr.style.display='block';
    vidRun.disabled=false;
  }
});

function pollJob(jobId){
  if(pollTimer)clearInterval(pollTimer);
  pollTimer=setInterval(async()=>{
    try{
      const r=await fetch('/track/status/'+jobId);
      const j=await r.json();
      if(j.error && j.status!=='processing' && j.status!=='queued'){
        clearInterval(pollTimer);
        vidProgress.style.display='none';
        vidErr.textContent='⚠ '+j.error;
        vidErr.style.display='block';
        vidRun.disabled=false;
        return;
      }
      const pct=Math.round((j.progress||0)*100);
      vidFill.style.width=pct+'%';
      vidPct.textContent=pct+'%';
      vidFrames.textContent='frame '+(j.frame||0)+' / '+(j.total_frames||0);
      vidTracks.textContent='tracks: '+(j.unique_tracks||0);

      if(j.status==='done'){
        clearInterval(pollTimer);
        vidProgress.style.display='none';
        const v=document.createElement('video');
        v.src=j.video_url;
        v.controls=true;
        v.autoplay=true;
        v.muted=true;
        vidResult.appendChild(v);
        vidStats.style.display='grid';
        document.getElementById('stat-frames').textContent=j.total_frames;
        document.getElementById('stat-ids').textContent=j.unique_tracks;
        document.getElementById('stat-elapsed').textContent=j.elapsed_s;
        document.getElementById('stat-fps').textContent=j.fps_processed;
        vidRun.disabled=false;
      }
    }catch(err){
      clearInterval(pollTimer);
      vidProgress.style.display='none';
      vidErr.textContent='⚠ Polling failed: '+err.message;
      vidErr.style.display='block';
      vidRun.disabled=false;
    }
  },800);
}
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────
@api.get("/", response_class=HTMLResponse)
async def index():
    return UI

@api.post("/detect")
async def detect(
    file:  UploadFile = File(...),
    model: str        = Form("yolo"),
    conf:  float      = Form(0.35),
):
    try:
        raw    = await file.read()
        img_np = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        t0     = time.time()

        if model == "yolo":
            boxes, scores, labels = run_yolo(img_np, conf)
        elif model == "frcnn":
            boxes, scores, labels = run_torchvision(FRCNN_MODEL, img_np, conf)
        else:
            boxes, scores, labels = run_torchvision(SSD_MODEL, img_np, conf)

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        annotated  = annotate(img_np, boxes, scores, labels)

        return JSONResponse({
            "count":        len(boxes),
            "inference_ms": elapsed_ms,
            "image_b64":    img_to_b64(annotated),
            "detections":   [{"label": l, "score": f"{s:.3f}"} for l, s in zip(labels, scores)],
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@api.post("/track")
async def track(
    file: UploadFile = File(...),
    conf: float      = Form(0.35),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported video extension '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTS)}")

    job_id  = uuid.uuid4().hex[:10]
    in_path = UPLOADS_DIR / f"{job_id}{ext}"
    out_path = OUTPUTS_DIR / f"{job_id}.mp4"

    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "frame": 0,
            "total_frames": 0,
            "unique_tracks": 0,
        }

    threading.Thread(
        target=process_video_job,
        args=(job_id, in_path, out_path, conf),
        daemon=True,
    ).start()

    return {"job_id": job_id}

@api.get("/track/status/{job_id}")
async def track_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return JSONResponse({"error": "job not found", "status": "missing"}, status_code=404)
        return dict(job)

@api.get("/outputs/{filename}")
async def get_output(filename: str):
    safe = Path(filename).name
    path = OUTPUTS_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="output not found")
    return FileResponse(path, media_type="video/mp4", filename=safe)

@api.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASSES}
