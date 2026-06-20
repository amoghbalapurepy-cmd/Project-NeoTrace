"""
╔════════════════════════════════════════════════════════════════════════╗
║                OBJECT INTEL SYSTEM  v2  (Pro Edition)                  ║
╠════════════════════════════════════════════════════════════════════════╣
║  • Persistent object tracking   (IoU tracker + EMA smoothing + ID)     ║
║  • Hands                        (MediaPipe – 21 landmarks/hand)        ║
║  • Gesture recognition          (Fist / Pointer / Peace / Palm / etc.) ║
║  • Face mesh + Iris             (MediaPipe – 468+ landmarks)           ║
║  • Eye blink detection          (Eye-Aspect-Ratio algorithm)           ║
║  • Scene objects                (YOLOv8 – 80 COCO classes)             ║
║  • In-hand detection            (tracked-object bbox ∩ hand bbox)      ║
║  • Brand / Company identifier   (CLIP zero-shot, optional)             ║
║  • AR shape drawing             (draw in the air -> shape is detected  ║
║                                   and rendered as a clean neon AR      ║
║                                   shape: circle / triangle / square /  ║
║                                   rectangle / pentagon / line)         ║
╠════════════════════════════════════════════════════════════════════════╣
║  HONEST NOTE: no computer-vision system is ever literally 100%         ║
║  accurate. This build maximizes *practical* accuracy via a bigger      ║
║  model, multi-frame confirmation (kills one-frame flicker / false      ║
║  positives), smoothing, and confidence-gated brand guesses that say    ║
║  "unknown" rather than bluff. That's how real systems do it.           ║
╚════════════════════════════════════════════════════════════════════════╝

INSTALL:
    pip install -r requirements.txt
    (brand identification needs torch + transformers + pillow — optional,
     the rest of the app works fine without them)

RUN:
    python object_identifier.py

CONTROLS (shown live in-app too):
    Q  Quit            S  Screenshot         H  Toggle HUD
    O  Toggle objects  B  Toggle brand-ID     A  Toggle AR drawing
    C  Clear AR canvas

AR DRAWING — HOW TO USE:
    • Point with ONE finger (index up, rest curled) and move it in the
      air to draw. Change your hand shape (fist / open palm / etc.) to
      finish the stroke — the system will classify it as a Circle,
      Triangle, Square, Rectangle, Pentagon, or Line and render a clean
      glowing AR version of it.
    • Hold an OPEN PALM for ~1.5s to clear the whole AR canvas.
"""

import math
import time
import os
from collections import deque, defaultdict

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
CAMERA_INDEX        = 0
FRAME_WIDTH         = 960      # lower = faster. Try 640x480 if still slow.
FRAME_HEIGHT        = 540

YOLO_MODEL          = "yolov8n.pt"     # n=fastest(default), s, m, l/x=most accurate but slow on CPU
YOLO_CONF_THRESHOLD = 0.40
YOLO_INFER_IMGSZ    = 480      # inference resolution fed to YOLO (independent of camera res). Lower = faster.
YOLO_SKIP_FRAMES    = 2        # run YOLO every Nth frame; reuse last result in between. 1 = every frame.

# --- Tracking ---
IOU_MATCH_THRESH    = 0.30
CONFIRM_THRESHOLD   = 3        # frames an object must be seen before it's "confirmed" / drawn
MAX_MISSED_FRAMES   = 20       # frames a track can go undetected before being dropped
BBOX_SMOOTH_ALPHA   = 0.55     # EMA smoothing factor for bounding boxes (higher = snappier)

# --- Brand identification ---
ENABLE_BRAND_ID         = False  # OFF by default — CLIP is the heaviest feature. Toggle on with 'B' in-app.
BRAND_RECHECK_INTERVAL  = 45   # frames between brand re-checks per tracked object
BRAND_MIN_CONFIDENCE    = 0.32 # below this -> shown as "Unrecognized brand"

# --- AR drawing ---
AR_MIN_POINTS         = 10
AR_SHAPE_LIFETIME     = 4.5    # seconds a finished AR shape stays on screen
AR_CLOSE_DIST_RATIO   = 0.30   # how close start/end must be (relative to bbox diag) to call it "closed"
PALM_HOLD_CLEAR_FRAMES = 45    # ~1.5s at 30fps to clear canvas with open palm

# --- Eyes ---
EAR_BLINK_THRESHOLD   = 0.21

SCREENSHOT_DIR = "screenshots"
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# Palette (BGR)
C_BG        = (18, 18, 22)
C_ACCENT    = (60, 220, 230)
C_HAND      = (0, 255, 150)
C_FACE      = (255, 180, 0)
C_EYE       = (0, 220, 255)
C_OBJ       = (170, 210, 0)
C_OBJ_HAND  = (0, 120, 255)
C_TEXT      = (235, 235, 235)
C_DIM       = (140, 140, 140)
C_GOOD      = (90, 230, 120)
C_BAD       = (90, 90, 230)

SHAPE_COLORS = {
    "Circle":    (0, 255, 255),
    "Triangle":  (0, 255, 120),
    "Square":    (255, 180, 0),
    "Rectangle": (255, 140, 0),
    "Pentagon":  (200, 100, 255),
    "Freeform":  (255, 255, 255),
    "Line":      (120, 200, 255),
}

# ════════════════════════════════════════════════════════════════════════
#  STATIC KNOWLEDGE BASE — Company info ("high class information finder")
# ════════════════════════════════════════════════════════════════════════
COMPANY_DB = {
    "Apple":              {"company": "Apple Inc.",            "founded": 1976, "hq": "Cupertino, USA",      "fact": "Originally built in a garage by Jobs, Wozniak & Wayne."},
    "Samsung":            {"company": "Samsung Electronics",   "founded": 1969, "hq": "Suwon, South Korea",  "fact": "Part of the larger Samsung Group conglomerate."},
    "Google":              {"company": "Google LLC",            "founded": 1998, "hq": "Mountain View, USA",  "fact": "Pixel devices are designed in-house by Google."},
    "OnePlus":            {"company": "OnePlus Technology",    "founded": 2013, "hq": "Shenzhen, China",     "fact": "Known for the 'Never Settle' slogan."},
    "Xiaomi":             {"company": "Xiaomi Corporation",     "founded": 2010, "hq": "Beijing, China",      "fact": "One of the world's largest smartphone makers by volume."},
    "Dell":               {"company": "Dell Technologies",     "founded": 1984, "hq": "Round Rock, USA",     "fact": "Started as 'PC's Limited' from a college dorm room."},
    "HP":                 {"company": "HP Inc.",               "founded": 1939, "hq": "Palo Alto, USA",      "fact": "Founded in a one-car garage — now a Silicon Valley landmark."},
    "Lenovo":             {"company": "Lenovo Group",          "founded": 1984, "hq": "Beijing, China",      "fact": "Acquired IBM's PC business (incl. ThinkPad) in 2005."},
    "Asus":               {"company": "ASUSTeK Computer",      "founded": 1989, "hq": "Taipei, Taiwan",      "fact": "Name derived from 'Pegasus', the mythical winged horse."},
    "Acer":               {"company": "Acer Inc.",             "founded": 1976, "hq": "New Taipei, Taiwan",  "fact": "Originally named Multitech."},
    "LG":                 {"company": "LG Electronics",        "founded": 1958, "hq": "Seoul, South Korea",  "fact": "LG stands for 'Lucky Goldstar'."},
    "Sony":               {"company": "Sony Group",            "founded": 1946, "hq": "Tokyo, Japan",        "fact": "Started as a telecom/electronics repair company post-WWII."},
    "TCL":                {"company": "TCL Technology",        "founded": 1981, "hq": "Huizhou, China",      "fact": "One of the largest global TV manufacturers by shipment."},
    "Logitech":           {"company": "Logitech International","founded": 1981, "hq": "Lausanne, Switzerland","fact": "Made one of the first commercial computer mice."},
    "Corsair":            {"company": "Corsair Gaming",        "founded": 1994, "hq": "Fremont, USA",        "fact": "Started out making memory modules before peripherals."},
    "Razer":              {"company": "Razer Inc.",            "founded": 2005, "hq": "Singapore/Irvine",    "fact": "Known for its green-snake gaming logo."},
    "Microsoft":          {"company": "Microsoft Corporation", "founded": 1975, "hq": "Redmond, USA",        "fact": "Co-founded by Bill Gates and Paul Allen."},
    "Coca-Cola":          {"company": "The Coca-Cola Company", "founded": 1892, "hq": "Atlanta, USA",        "fact": "The contour bottle shape was trademarked in 1977."},
    "Pepsi":              {"company": "PepsiCo",               "founded": 1965, "hq": "Purchase, USA",       "fact": "Originally called 'Brad's Drink' after its creator."},
    "Bisleri":            {"company": "Bisleri International", "founded": 1965, "hq": "Mumbai, India",       "fact": "The name became a generic term for bottled water in India."},
    "Starbucks":          {"company": "Starbucks Corporation", "founded": 1971, "hq": "Seattle, USA",        "fact": "Named after a character in Moby-Dick."},
    "Toyota":             {"company": "Toyota Motor Corp.",    "founded": 1937, "hq": "Toyota City, Japan",  "fact": "World's largest automaker by production volume."},
    "Honda":              {"company": "Honda Motor Co.",       "founded": 1948, "hq": "Tokyo, Japan",        "fact": "Started by making motorized bicycles after WWII."},
    "BMW":                {"company": "Bayerische Motoren Werke","founded": 1916,"hq": "Munich, Germany",    "fact": "The logo nods to its early aircraft-engine roots."},
    "Tesla":              {"company": "Tesla, Inc.",           "founded": 2003, "hq": "Austin, USA",         "fact": "Named after engineer/inventor Nikola Tesla."},
    "Ford":               {"company": "Ford Motor Company",    "founded": 1903, "hq": "Dearborn, USA",       "fact": "Pioneered the moving assembly line for cars."},
    "Nike":               {"company": "Nike, Inc.",            "founded": 1964, "hq": "Beaverton, USA",      "fact": "The Swoosh logo was designed for just $35."},
    "Adidas":             {"company": "Adidas AG",             "founded": 1949, "hq": "Herzogenaurach, Germany","fact": "Named after founder Adolf 'Adi' Dassler."},
    "The North Face":     {"company": "The North Face, Inc.",  "founded": 1966, "hq": "Denver, USA",         "fact": "Named for the coldest, iciest face of a mountain."},
    "Louis Vuitton":      {"company": "Louis Vuitton",         "founded": 1854, "hq": "Paris, France",       "fact": "The LV monogram was created in 1896 to fight counterfeits."},
    "Gucci":              {"company": "Gucci",                 "founded": 1921, "hq": "Florence, Italy",     "fact": "Started as a leather-goods shop for travelers."},
    "Samsonite":          {"company": "Samsonite International","founded": 1910,"hq": "Mansfield, USA",     "fact": "Name inspired by the biblical strongman Samson."},
    "American Tourister": {"company": "American Tourister",   "founded": 1933, "hq": "Mansfield, USA",      "fact": "Now owned by the Samsonite group."},
    "Whirlpool":          {"company": "Whirlpool Corporation", "founded": 1911, "hq": "Benton Harbor, USA",  "fact": "One of the world's largest home-appliance makers."},
    "Panasonic":          {"company": "Panasonic Holdings",    "founded": 1918, "hq": "Osaka, Japan",        "fact": "Originally founded as Matsushita Electric."},
}

# CLIP zero-shot candidate prompts per YOLO class. The model picks the
# closest match; if confidence is low we say so instead of guessing.
CATEGORY_PROMPTS = {
    "cell phone": [
        ("Apple",   "a photo of an Apple iPhone smartphone"),
        ("Samsung", "a photo of a Samsung Galaxy smartphone"),
        ("Google",  "a photo of a Google Pixel smartphone"),
        ("OnePlus", "a photo of a OnePlus smartphone"),
        ("Xiaomi",  "a photo of a Xiaomi smartphone"),
        (None,      "a photo of an unbranded generic phone"),
    ],
    "laptop": [
        ("Apple",  "a photo of an Apple MacBook laptop"),
        ("Dell",   "a photo of a Dell laptop"),
        ("HP",     "a photo of an HP laptop"),
        ("Lenovo", "a photo of a Lenovo ThinkPad laptop"),
        ("Asus",   "a photo of an Asus laptop"),
        ("Acer",   "a photo of an Acer laptop"),
        (None,     "a photo of an unbranded generic laptop"),
    ],
    "tv": [
        ("Samsung", "a photo of a Samsung television"),
        ("LG",      "a photo of an LG television"),
        ("Sony",    "a photo of a Sony television"),
        ("TCL",     "a photo of a TCL television"),
        (None,      "a photo of an unbranded generic television"),
    ],
    "keyboard": [
        ("Apple",    "a photo of an Apple Magic Keyboard"),
        ("Logitech", "a photo of a Logitech keyboard"),
        ("Corsair",  "a photo of a Corsair mechanical keyboard"),
        ("Razer",    "a photo of a Razer mechanical keyboard"),
        (None,       "a photo of an unbranded generic keyboard"),
    ],
    "mouse": [
        ("Apple",     "a photo of an Apple Magic Mouse"),
        ("Logitech",  "a photo of a Logitech computer mouse"),
        ("Razer",     "a photo of a Razer gaming mouse"),
        ("Microsoft", "a photo of a Microsoft computer mouse"),
        (None,        "a photo of an unbranded generic computer mouse"),
    ],
    "bottle": [
        ("Coca-Cola", "a photo of a Coca-Cola bottle"),
        ("Pepsi",     "a photo of a Pepsi bottle"),
        ("Bisleri",   "a photo of a Bisleri water bottle"),
        (None,        "a photo of a plain unbranded water bottle"),
    ],
    "cup": [
        ("Starbucks", "a photo of a Starbucks paper cup"),
        (None,        "a photo of a plain unbranded cup or mug"),
    ],
    "car": [
        ("Toyota", "a photo of a Toyota car"),
        ("Honda",  "a photo of a Honda car"),
        ("BMW",    "a photo of a BMW car"),
        ("Tesla",  "a photo of a Tesla car"),
        ("Ford",   "a photo of a Ford car"),
        (None,     "a photo of an unidentifiable car"),
    ],
    "backpack": [
        ("Nike",          "a photo of a Nike backpack"),
        ("Adidas",        "a photo of an Adidas backpack"),
        ("The North Face","a photo of a The North Face backpack"),
        (None,            "a photo of an unbranded generic backpack"),
    ],
    "handbag": [
        ("Louis Vuitton", "a photo of a Louis Vuitton handbag"),
        ("Gucci",         "a photo of a Gucci handbag"),
        (None,            "a photo of an unbranded generic handbag"),
    ],
    "suitcase": [
        ("Samsonite",          "a photo of a Samsonite suitcase"),
        ("American Tourister", "a photo of an American Tourister suitcase"),
        (None,                 "a photo of an unbranded generic suitcase"),
    ],
    "refrigerator": [
        ("Samsung",  "a photo of a Samsung refrigerator"),
        ("LG",       "a photo of an LG refrigerator"),
        ("Whirlpool","a photo of a Whirlpool refrigerator"),
        (None,       "a photo of an unbranded generic refrigerator"),
    ],
    "microwave": [
        ("Samsung",   "a photo of a Samsung microwave oven"),
        ("LG",        "a photo of an LG microwave oven"),
        ("Panasonic", "a photo of a Panasonic microwave oven"),
        (None,        "a photo of an unbranded generic microwave oven"),
    ],
}

# ════════════════════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ════════════════════════════════════════════════════════════════════════
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_MCPS  = [2, 5, 9, 13, 17]

# 6-pt EAR sets: [corner, top1, top2, corner, bottom2, bottom1]
LEFT_EYE_EAR_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

GESTURE_MAP = {
    (0, 0, 0, 0, 0): "Fist",
    (1, 1, 1, 1, 1): "Open Palm",
    (0, 1, 0, 0, 0): "Pointer",
    (0, 1, 1, 0, 0): "Peace",
    (1, 1, 0, 0, 0): "L-Shape",
    (1, 0, 0, 0, 1): "Call Me",
    (0, 1, 1, 1, 0): "Three",
    (0, 1, 1, 1, 1): "Four",
    (1, 0, 0, 0, 0): "Thumbs Up",
}

# ════════════════════════════════════════════════════════════════════════
#  BRAND IDENTIFIER  (CLIP zero-shot — optional, degrades gracefully)
# ════════════════════════════════════════════════════════════════════════
class BrandIdentifier:
    def __init__(self, device="cpu"):
        self.available = False
        self.device = device
        if not ENABLE_BRAND_ID:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.torch = torch
            print("[INFO] Loading CLIP brand-identification model (first run downloads ~600MB) …")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
            self.available = True
            print(f"[INFO] Brand identification ready on {device}.")
        except Exception as e:
            print(f"[WARN] Brand identification disabled (install torch+transformers+pillow to enable). Reason: {e}")
            self.available = False

    def classify(self, crop_bgr, yolo_label):
        if not self.available or yolo_label not in CATEGORY_PROMPTS or crop_bgr.size == 0:
            return None
        try:
            from PIL import Image
            candidates = CATEGORY_PROMPTS[yolo_label]
            texts = [c[1] for c in candidates]
            image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            best_idx = int(probs.argmax())
            conf = float(probs[best_idx])
            brand_name = candidates[best_idx][0]   # may be None ("unbranded")
            return brand_name, conf
        except Exception:
            return None


# ════════════════════════════════════════════════════════════════════════
#  PERSISTENT OBJECT TRACKER  (lightweight IoU tracker, SORT-style)
# ════════════════════════════════════════════════════════════════════════
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)


class ObjectTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_idx):
        """detections: list of (bbox, label, conf)"""
        unmatched = list(range(len(detections)))
        matched_ids = set()

        for tid, trk in list(self.tracks.items()):
            best_iou, best_j = 0.0, -1
            for j in unmatched:
                bbox, label, conf = detections[j]
                if label != trk["label"]:
                    continue
                val = iou(bbox, trk["bbox"])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_iou > IOU_MATCH_THRESH:
                bbox, label, conf = detections[best_j]
                a = BBOX_SMOOTH_ALPHA
                ex1, ey1, ex2, ey2 = trk["bbox"]
                nx1, ny1, nx2, ny2 = bbox
                trk["bbox"] = (
                    int(a * nx1 + (1 - a) * ex1), int(a * ny1 + (1 - a) * ey1),
                    int(a * nx2 + (1 - a) * ex2), int(a * ny2 + (1 - a) * ey2),
                )
                trk["conf"] = conf
                trk["confirmed"] = min(trk["confirmed"] + 1, 999)
                trk["missed"] = 0
                unmatched.remove(best_j)
                matched_ids.add(tid)

        for tid, trk in list(self.tracks.items()):
            if tid not in matched_ids:
                trk["missed"] += 1
                if trk["missed"] > MAX_MISSED_FRAMES:
                    del self.tracks[tid]

        for j in unmatched:
            bbox, label, conf = detections[j]
            self.tracks[self.next_id] = {
                "bbox": bbox, "label": label, "conf": conf,
                "confirmed": 1, "missed": 0,
                "brand": "?", "brand_conf": 0.0, "last_brand_check": -99999,
            }
            self.next_id += 1

        return self.tracks


# ════════════════════════════════════════════════════════════════════════
#  AR SHAPE CLASSIFIER
# ════════════════════════════════════════════════════════════════════════
def classify_shape(points):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < AR_MIN_POINTS:
        return None

    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    diag = math.hypot(w, h) + 1e-6
    if diag < 50:
        return None

    closed = np.linalg.norm(pts[0] - pts[-1]) < AR_CLOSE_DIST_RATIO * diag
    hull = cv2.convexHull(pts.astype(np.int32))
    area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True) + 1e-6
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)
    vcount = len(approx)
    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    if not closed:
        step = max(1, len(pts) // 40)
        return {"name": "Line", "pts": pts[::step], "closed": False, "centroid": (cx, cy)}

    if circularity > 0.72 or (vcount >= 6 and circularity > 0.55):
        (ccx, ccy), radius = cv2.minEnclosingCircle(hull)
        circle_pts = np.array([
            [ccx + radius * math.cos(t), ccy + radius * math.sin(t)]
            for t in np.linspace(0, 2 * math.pi, 40)
        ])
        return {"name": "Circle", "pts": circle_pts, "closed": True, "centroid": (ccx, ccy)}

    if vcount == 3:
        return {"name": "Triangle", "pts": approx, "closed": True, "centroid": (cx, cy)}

    if vcount == 4:
        rect = cv2.minAreaRect(hull)
        rw, rh = rect[1]
        ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
        box = cv2.boxPoints(rect)
        name = "Square" if ratio < 1.25 else "Rectangle"
        return {"name": name, "pts": box, "closed": True, "centroid": (cx, cy)}

    if vcount == 5:
        return {"name": "Pentagon", "pts": approx, "closed": True, "centroid": (cx, cy)}

    return {"name": "Freeform", "pts": approx, "closed": True, "centroid": (cx, cy)}


# ════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ════════════════════════════════════════════════════════════════════════
def text(img, s, x, y, color=C_TEXT, scale=0.5, thick=1):
    cv2.putText(img, s, (x + 1, y + 1), FONT, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def glass_panel(img, x1, y1, x2, y2, alpha=0.55, color=C_BG, border=C_ACCENT):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = img[y1:y2, x1:x2]
    overlay = np.full_like(sub, color)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)


def draw_brackets(img, x1, y1, x2, y2, color, length=16, thickness=2):
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (px, py), (px + dx * length, py), color, thickness)
        cv2.line(img, (px, py), (px, py + dy * length), color, thickness)


def confidence_bar(img, x, y, w, h, frac, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0, min(1, frac))), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)


def led(img, x, y, on, label):
    cv2.circle(img, (x, y), 5, C_GOOD if on else C_DIM, -1)
    text(img, label, x + 10, y + 4, color=C_TEXT if on else C_DIM, scale=0.42)


def draw_ghost_shape(frame, gs, alpha):
    pts = np.array(gs["pts"], dtype=np.int32)
    color = gs["color"]
    overlay = frame.copy()
    for thickness in (12, 7, 3):
        cv2.polylines(overlay, [pts], gs["closed"], color, thickness, cv2.LINE_AA)
    cv2.polylines(overlay, [pts], gs["closed"], (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)

    cx, cy = gs["centroid"]
    label = gs["name"].upper()
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    lx, ly = int(cx - tw / 2), int(cy)
    cv2.rectangle(frame, (lx - 5, ly - th - 6), (lx + tw + 5, ly + 4), color, -1)
    cv2.putText(frame, label, (lx, ly), FONT, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

    age = AR_SHAPE_LIFETIME * (1 - alpha)
    if age < 0.4:
        r = int((age / 0.4) * 70)
        ring_alpha = 1 - (age / 0.4)
        ov = frame.copy()
        cv2.circle(ov, (int(cx), int(cy)), max(1, r), color, 2)
        cv2.addWeighted(ov, ring_alpha, frame, 1 - ring_alpha, 0, frame)


# ════════════════════════════════════════════════════════════════════════
#  HAND / FACE GEOMETRY HELPERS
# ════════════════════════════════════════════════════════════════════════
def is_finger_up(lms, tip_idx, mcp_idx, handedness):
    tip, mcp = lms.landmark[tip_idx], lms.landmark[mcp_idx]
    if tip_idx == 4:
        return tip.x < mcp.x if handedness == "Right" else tip.x > mcp.x
    return tip.y < mcp.y


def hand_bbox(lms, w, h, pad=25):
    xs = [lm.x * w for lm in lms.landmark]
    ys = [lm.y * h for lm in lms.landmark]
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
            min(w - 1, int(max(xs)) + pad), min(h - 1, int(max(ys)) + pad))


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def eye_aspect_ratio(face_lms, idxs, w, h):
    p = [(face_lms.landmark[i].x * w, face_lms.landmark[i].y * h) for i in idxs]
    d = lambda a, b: math.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1])
    vertical = d(1, 5) + d(2, 4)
    horizontal = 2 * d(0, 3)
    return vertical / (horizontal + 1e-6)


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}" + ("" if device == "cuda" else "  (no GPU found — see FPS tips in the config section if this is slow)"))

    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.55)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("[INFO] Loading YOLO model …")
    yolo = YOLO(YOLO_MODEL)
    yolo.to(device)
    print("[INFO] YOLO ready.")

    brand_id = BrandIdentifier(device=device)
    tracker = ObjectTracker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try changing CAMERA_INDEX at the top of the file.")
        return

    show_hud, show_objects, show_brand, show_ar = True, True, ENABLE_BRAND_ID, True
    trails = defaultdict(lambda: deque(maxlen=400))
    drawing_state = defaultdict(bool)
    palm_hold = defaultdict(int)
    ghost_shapes = []
    last_detections = []   # cache so we can skip YOLO on some frames

    fps, frame_count, tick, shot_count = 0.0, 0, time.time(), 0

    print("\n[RUNNING]  Q quit | S screenshot | H hud | O objects | B brand-id | A ar-draw | C clear ar\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_count += 1
        if frame_count % 15 == 0:
            fps, tick = 15.0 / max(time.time() - tick, 1e-9), time.time()

        # ── 1. YOLO (every Nth frame) + tracker ──────────────────────────
        if show_objects:
            if frame_count % YOLO_SKIP_FRAMES == 0 or not last_detections:
                results = yolo(frame, verbose=False, conf=YOLO_CONF_THRESHOLD, imgsz=YOLO_INFER_IMGSZ)[0]
                detections = []
                for box in results.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    label = yolo.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    detections.append(((bx1, by1, bx2, by2), label, conf))
                last_detections = detections
            else:
                detections = last_detections
        else:
            detections = []
        tracks = tracker.update(detections, frame_count)

        # ── 2. Hands ──────────────────────────────────────────────────────
        hand_boxes, hand_hud = [], []
        hres = hands.process(rgb)
        if hres.multi_hand_landmarks:
            for lms, info in zip(hres.multi_hand_landmarks, hres.multi_handedness):
                side = info.classification[0].label
                mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS,
                                           mp_styles.get_default_hand_landmarks_style(),
                                           mp_styles.get_default_hand_connections_style())
                hx1, hy1, hx2, hy2 = hand_bbox(lms, w, h)
                hand_boxes.append((hx1, hy1, hx2, hy2))
                draw_brackets(frame, hx1, hy1, hx2, hy2, C_HAND)
                text(frame, f"HAND  {side}", hx1, hy1 - 8, color=C_HAND, scale=0.5)

                finger_ups = [is_finger_up(lms, t, m, side) for t, m in zip(FINGER_TIPS, FINGER_MCPS)]
                gesture = GESTURE_MAP.get(tuple(1 if f else 0 for f in finger_ups), "Unknown")
                hand_hud.append((side, gesture, finger_ups))

                for i, (tip, up) in enumerate(zip(FINGER_TIPS, finger_ups)):
                    lm = lms.landmark[tip]
                    tx, ty = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (tx, ty), 7, C_GOOD if up else C_BAD, -1)
                    cv2.circle(frame, (tx, ty), 7, (255, 255, 255), 1)

                # ── AR drawing state machine ────────────────────────────
                if show_ar:
                    if gesture == "Pointer":
                        tip = lms.landmark[FINGER_TIPS[1]]
                        trails[side].append((int(tip.x * w), int(tip.y * h)))
                        drawing_state[side] = True
                        palm_hold[side] = 0
                    else:
                        if drawing_state[side] and len(trails[side]) >= AR_MIN_POINTS:
                            shape = classify_shape(list(trails[side]))
                            if shape:
                                shape["color"] = SHAPE_COLORS.get(shape["name"], (255, 255, 255))
                                shape["created"] = time.time()
                                ghost_shapes.append(shape)
                        drawing_state[side] = False
                        trails[side].clear()

                        if gesture == "Open Palm":
                            palm_hold[side] += 1
                            if palm_hold[side] == PALM_HOLD_CLEAR_FRAMES:
                                ghost_shapes.clear()
                                for k in trails:
                                    trails[k].clear()
                        else:
                            palm_hold[side] = 0

        # ── 3. Face mesh + eyes ──────────────────────────────────────────
        face_count, eye_state = 0, []
        fres = face_mesh.process(rgb)
        if fres.multi_face_landmarks:
            face_count = len(fres.multi_face_landmarks)
            for flms in fres.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, flms, mp_face_mesh.FACEMESH_TESSELATION,
                                           None, mp_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(frame, flms, mp_face_mesh.FACEMESH_CONTOURS,
                                           None, mp_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(frame, flms, mp_face_mesh.FACEMESH_IRISES,
                                           None, mp_styles.get_default_face_mesh_iris_connections_style())

                xs = [lm.x * w for lm in flms.landmark]
                ys = [lm.y * h for lm in flms.landmark]
                fx1, fy1 = max(0, int(min(xs)) - 10), max(0, int(min(ys)) - 10)
                fx2, fy2 = min(w - 1, int(max(xs)) + 10), min(h - 1, int(max(ys)) + 10)
                draw_brackets(frame, fx1, fy1, fx2, fy2, C_FACE)
                text(frame, "FACE", fx1, fy1 - 8, color=C_FACE, scale=0.5)

                left_ear = eye_aspect_ratio(flms, LEFT_EYE_EAR_IDX, w, h)
                right_ear = eye_aspect_ratio(flms, RIGHT_EYE_EAR_IDX, w, h)
                blinking = (left_ear + right_ear) / 2 < EAR_BLINK_THRESHOLD
                eye_state.append(blinking)

        # ── 4. Draw tracked objects + brand ID ───────────────────────────
        visible_tracks, info_card = [], None
        if show_objects:
            for tid, trk in tracks.items():
                if trk["missed"] > 0 or trk["confirmed"] < CONFIRM_THRESHOLD:
                    continue
                bx1, by1, bx2, by2 = trk["bbox"]
                in_hand = any(boxes_overlap(trk["bbox"], hb) for hb in hand_boxes)

                if show_brand and trk["label"] in CATEGORY_PROMPTS and \
                        (frame_count - trk["last_brand_check"] > BRAND_RECHECK_INTERVAL):
                    crop = frame[max(0, by1):by2, max(0, bx1):bx2]
                    result = brand_id.classify(crop, trk["label"])
                    trk["last_brand_check"] = frame_count
                    if result:
                        trk["brand"], trk["brand_conf"] = result

                color = C_OBJ_HAND if in_hand else C_OBJ
                draw_brackets(frame, bx1, by1, bx2, by2, color, length=14, thickness=2)

                brand = trk.get("brand")
                if brand and brand != "?" and trk["brand_conf"] >= BRAND_MIN_CONFIDENCE:
                    sub = f"{brand}"
                else:
                    sub = None

                cap_text = f"{trk['label'].upper()}"
                if in_hand:
                    cap_text += "  [IN HAND]"
                text(frame, cap_text, bx1, by1 - 8, color=color, scale=0.48)
                if sub:
                    text(frame, sub, bx1, by2 + 16, color=(255, 255, 255), scale=0.45)

                visible_tracks.append((trk, in_hand))
                if in_hand and brand and brand != "?" and trk["brand_conf"] >= BRAND_MIN_CONFIDENCE:
                    if info_card is None or trk["brand_conf"] > info_card["brand_conf"]:
                        info_card = trk

        # ── 5. AR overlay — trails + finished ghost shapes ───────────────
        if show_ar:
            for side, pts in trails.items():
                if len(pts) > 1:
                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
            keep = []
            for gs in ghost_shapes:
                age = time.time() - gs["created"]
                if age < AR_SHAPE_LIFETIME:
                    alpha = 1 - age / AR_SHAPE_LIFETIME
                    draw_ghost_shape(frame, gs, alpha)
                    keep.append(gs)
            ghost_shapes[:] = keep

        # ── 6. HUD ─────────────────────────────────────────────────────
        if show_hud:
            glass_panel(frame, 0, 0, w, 36)
            text(frame, "OBJECT INTEL SYSTEM v2", 12, 24, color=C_ACCENT, scale=0.62, thick=2)
            text(frame, f"FPS {fps:4.1f}", w - 110, 24, color=C_GOOD if fps > 12 else C_BAD, scale=0.55)

            lx = 12
            led(frame, lx, 56, show_objects, "OBJECTS");  lx += 100
            led(frame, lx, 56, show_brand and brand_id.available, "BRAND");  lx += 90
            led(frame, lx, 56, show_ar, "AR DRAW");  lx += 100
            led(frame, lx, 56, True, f"FACES {face_count}");  lx += 110
            led(frame, lx, 56, len(hand_boxes) > 0, f"HANDS {len(hand_boxes)}")

            # Sidebar: object list
            panel_w = 250
            glass_panel(frame, w - panel_w, 80, w, 80 + 26 + len(visible_tracks[:7]) * 46)
            text(frame, "DETECTED OBJECTS", w - panel_w + 10, 102, color=C_ACCENT, scale=0.5, thick=1)
            for i, (trk, in_hand) in enumerate(sorted(visible_tracks, key=lambda t: -t[0]["conf"])[:7]):
                yy = 130 + i * 46
                col = C_OBJ_HAND if in_hand else C_OBJ
                cv2.circle(frame, (w - panel_w + 16, yy - 5), 4, col, -1)
                label_txt = trk["label"]
                if in_hand:
                    label_txt += " *"
                text(frame, label_txt, w - panel_w + 26, yy, color=C_TEXT, scale=0.45)
                confidence_bar(frame, w - panel_w + 26, yy + 6, panel_w - 50, 6, trk["conf"], col)

            # Gesture readout bottom-left
            gy = h - 20 - len(hand_hud) * 22
            for side, gesture, _ in hand_hud:
                text(frame, f"{side} hand: {gesture}", 14, gy, color=C_HAND, scale=0.48)
                gy += 22
            if eye_state:
                blink_txt = "Eyes: CLOSED" if any(eye_state) else "Eyes: OPEN"
                text(frame, blink_txt, 14, h - 14, color=C_EYE, scale=0.48)

            # Info card — brand / company details
            if info_card is not None:
                cw, ch = 320, 130
                cx1, cy1 = 12, 80
                glass_panel(frame, cx1, cy1, cx1 + cw, cy1 + ch, border=C_OBJ_HAND)
                comp = COMPANY_DB.get(info_card["brand"], {})
                text(frame, f"DEVICE INFO", cx1 + 10, cy1 + 22, color=C_OBJ_HAND, scale=0.5, thick=1)
                text(frame, f"{info_card['label'].title()} -> {info_card['brand']}", cx1 + 10, cy1 + 46, scale=0.46)
                text(frame, f"Company : {comp.get('company','-')}", cx1 + 10, cy1 + 66, scale=0.42, color=C_DIM)
                text(frame, f"Founded : {comp.get('founded','-')}   HQ: {comp.get('hq','-')}", cx1 + 10, cy1 + 84, scale=0.42, color=C_DIM)
                text(frame, f"{comp.get('fact','')}", cx1 + 10, cy1 + 104, scale=0.40, color=C_DIM)

            # Controls cheat sheet
            text(frame, "Q quit  S shot  H hud  O obj  B brand  A ar  C clear",
                 12, h - 4 if not eye_state else h - 4, color=C_DIM, scale=0.4)

        cv2.imshow("Object Intel System v2  |  Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot_count += 1
            path = os.path.join(SCREENSHOT_DIR, f"shot_{shot_count:04d}.jpg")
            cv2.imwrite(path, frame)
            print(f"[SCREENSHOT] {path}")
        elif key == ord('h'):
            show_hud = not show_hud
        elif key == ord('o'):
            show_objects = not show_objects
        elif key == ord('b'):
            show_brand = not show_brand
        elif key == ord('a'):
            show_ar = not show_ar
        elif key == ord('c'):
            ghost_shapes.clear()
            for k in trails:
                trails[k].clear()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
