"""
Waste Sorting - Live Stream with YOLO Detection + Keras Classification
YOLO finds where objects are (works on cluttered scenes, trash cans, tables)
Your Keras model classifies each detected region as compost/recycle/landfill.

Run: python live_stream.py
Open: http://localhost:5000
"""

import cv2
import numpy as np
from tensorflow import keras
from flask import Flask, Response
from ultralytics import YOLO
import threading

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH        = "models/waste_sorting_model.keras"
IMG_SIZE          = 224
CLASSES           = ['compost', 'recycle', 'landfill']

# How often to run detection + classification (every N frames)
# Lower = more responsive, higher = smoother video on slow hardware
DETECT_EVERY_N_FRAMES = 8

# YOLO confidence threshold — detections below this are ignored
YOLO_CONFIDENCE = 0.35

# Keras confidence threshold — below this we label the box "?"
KERAS_CONFIDENCE = 0.50

# Colors per category (BGR)
CATEGORY_COLORS = {
    'compost':  (34,  180, 34),
    'recycle':  (200, 140, 30),
    'landfill': (40,  40,  220),
    None:       (160, 160, 160),
}
# ───────────────────────────────────────────────────────────────────────────────

# Shared detection state between inference and stream threads
latest_detections = []   # list of (x1, y1, x2, y2, label, confidence, yolo_class)
detections_lock   = threading.Lock()

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading YOLO model...")
yolo = YOLO("yolov8n.pt")   # Downloads ~6MB on first run, then cached
print("✓ YOLO loaded!")

print("Loading Keras waste model...")
keras_model = keras.models.load_model(MODEL_PATH)
print("✓ Keras model loaded!")
# ───────────────────────────────────────────────────────────────────────────────


def classify_crop(frame, x1, y1, x2, y2):
    """Crop a detected region and run it through the Keras waste classifier."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    resized    = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    batch      = np.expand_dims(normalized, axis=0)

    preds          = keras_model.predict(batch, verbose=0)[0]
    predicted_idx  = int(np.argmax(preds))
    confidence     = float(preds[predicted_idx])

    if confidence < KERAS_CONFIDENCE:
        return None, confidence

    return CLASSES[predicted_idx], confidence


def run_detection(frame):
    """Run YOLO on the frame, then classify each detected region with Keras."""
    results = yolo(frame, conf=YOLO_CONFIDENCE, verbose=False)[0]
    new_detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        yolo_class      = results.names[int(box.cls[0])]

        # Pad the crop slightly so we don't clip the object edges
        h, w = frame.shape[:2]
        pad  = 8
        x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
        x2c, y2c = min(w, x2 + pad), min(h, y2 + pad)

        label, confidence = classify_crop(frame, x1c, y1c, x2c, y2c)
        new_detections.append((x1, y1, x2, y2, label, confidence, yolo_class))

    return new_detections


def draw_boxes(frame, detections):
    """Draw a bounding box and label for each detected item."""
    for (x1, y1, x2, y2, label, confidence, yolo_class) in detections:
        color         = CATEGORY_COLORS.get(label, CATEGORY_COLORS[None])
        display_label = label.upper() if label else "?"
        conf_text     = f" {confidence:.0%}" if label else ""
        text          = f" {display_label}{conf_text} "

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Corner accents (makes it feel more like a scanner)
        corner_len = 18
        thickness  = 4
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

        # Label background
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        thickness  = 2
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        label_y = y1 - 8 if y1 - 8 > th + 4 else y2 + th + 8
        cv2.rectangle(frame,
                      (x1, label_y - th - baseline - 4),
                      (x1 + tw, label_y + baseline),
                      color, -1)
        cv2.putText(frame, text,
                    (x1, label_y - 4),
                    font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        # Small YOLO class name below (subtle, helps with debugging)
        cv2.putText(frame, yolo_class,
                    (x1 + 4, y2 - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.9,
                    color, 1, cv2.LINE_AA)

    return frame


def generate_frames():
    camera      = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        # Run detection on every Nth frame
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            detections = run_detection(frame)
            with detections_lock:
                latest_detections[:] = detections

        # Draw the latest detections on every frame (keeps video smooth)
        with detections_lock:
            current = list(latest_detections)

        frame = draw_boxes(frame, current)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

    camera.release()


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Waste Sorter</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                background: #080808;
                width: 100vw;
                height: 100vh;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                font-family: 'Courier New', monospace;
            }

            header {
                background: #111;
                border-bottom: 1px solid #1e1e1e;
                padding: 10px 24px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-shrink: 0;
            }

            .title {
                font-size: 0.85rem;
                letter-spacing: 4px;
                color: #666;
            }

            .title span {
                color: #4caf50;
            }

            .legend {
                display: flex;
                gap: 20px;
            }

            .legend-item {
                display: flex;
                align-items: center;
                gap: 7px;
                font-size: 0.7rem;
                letter-spacing: 2px;
                color: #555;
            }

            .dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
            }

            .stream-wrap {
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }

            img {
                width: 100%;
                height: 100%;
                object-fit: contain;
            }
        </style>
    </head>
    <body>
        <header>
            <div class="title"><span>♻</span> WASTE SORTER — LIVE DETECTION</div>
            <div class="legend">
                <div class="legend-item">
                    <div class="dot" style="background:#22b422"></div>COMPOST
                </div>
                <div class="legend-item">
                    <div class="dot" style="background:#c88c1e"></div>RECYCLE
                </div>
                <div class="legend-item">
                    <div class="dot" style="background:#2828dc"></div>LANDFILL
                </div>
            </div>
        </header>
        <div class="stream-wrap">
            <img src="/video">
        </div>
    </body>
    </html>
    '''


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  WASTE SORTER — YOLO + KERAS LIVE DETECTION")
    print("=" * 55)
    print("  Browser (this machine):  http://localhost:5000")
    print("  Browser (network):       http://YOUR_IP:5000")
    print("=" * 55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)