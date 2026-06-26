"""
╔════════════════════════════════════════════════════════════════════════╗
║          OBJECT INTEL SYSTEM  v3  —  AR MODULE + TARGET MEMORY         ║
╠════════════════════════════════════════════════════════════════════════╣
║  • Persistent object & face tracking  (IoU tracker + EMA smoothing)    ║
║  • Sci-Fi targeting sequence           Phase1 SCANNING -> Phase2 LOCKED║
║  • Floating tool dock                  mouse-click OR finger-hover     ║
║  • 4 AR tools: Eraser / Clear / Color Fill / Color Changer             ║
║  • Projection persistence toggle       Manual (stays) / Auto (fades)  ║
║  • AR shape drawing                    draw in the air -> clean shape ║
║  • Hands, face mesh, iris, blink       (MediaPipe)                    ║
║  • Brand / company identifier          (CLIP zero-shot, optional)     ║
║  • TARGET MEMORY (AI persistence)      memorize a face/object, attach ║
║                                         photos/docs/notes, auto-recall║
╠════════════════════════════════════════════════════════════════════════╣
║  HONEST NOTES:                                                         ║
║  • No CV system is 100% accurate. Recognition is confidence-gated —    ║
║    it says "no match" rather than guessing wrong.                      ║
║  • Face identity matching uses `face_recognition` if installed         ║
║    (most accurate), else falls back to CLIP image similarity, else a  ║
║    basic color-histogram descriptor (always available, least exact).  ║
║  • Storage is LOCAL (./target_memory/) so it persists between runs on ║
║    this machine. Multi-device "cloud" sync would need a real backend  ║
║    (auth + hosted DB) — out of scope for a standalone script, but the ║
║    TargetMemoryDB class below is the seam where that would plug in.   ║
╚════════════════════════════════════════════════════════════════════════╝

INSTALL:
    pip install -r requirements.txt
    (Optional, for best results: torch+transformers+pillow for brand-ID
     and CLIP-based memory matching; face_recognition for accurate face
     identity matching. Everything else works fine without them.)

RUN:
    python object_identifier.py

CONTROLS:
    Q quit | S screenshot | H hud | O objects | B brand-id
    T target-memory | A ar-draw | C clear-all-ar | P toggle persistence
    1-4 select tool (Eraser/Clear/ColorFill/ColorChanger) | [ ] cycle color
    D  add details for the currently selected target
    Mouse: click dock buttons, click a locked target to select/apply tool
    Finger: hover (Pointer gesture) over a dock button ~1s to "tap" it
"""

import math
import time
import os
import json
import uuid
import shutil
from collections import deque, defaultdict

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
WINDOW_NAME         = "Object Intel System v3 - AR Module"
CAMERA_INDEX        = 0
FRAME_WIDTH         = 960
FRAME_HEIGHT        = 540

YOLO_MODEL          = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.40
YOLO_INFER_IMGSZ    = 480
YOLO_SKIP_FRAMES    = 2

# --- Tracking ---
IOU_MATCH_THRESH    = 0.30
CONFIRM_THRESHOLD   = 5        # frames before a target snaps from SCANNING -> LOCKED
MAX_MISSED_FRAMES   = 20
BBOX_SMOOTH_ALPHA   = 0.55

# --- Brand identification (optional) ---
ENABLE_BRAND_ID         = False
BRAND_RECHECK_INTERVAL  = 45
BRAND_MIN_CONFIDENCE    = 0.32

# --- Target Memory ---
ENABLE_TARGET_MEMORY    = True
MEMORY_DIR              = "target_memory"
TARGET_RECHECK_INTERVAL = 40    # frames between recognition checks per locked target
FACE_MATCH_MAX_DIST     = 0.60  # face_recognition Euclidean distance threshold (lower = stricter)
CLIP_MATCH_MIN_SIM      = 0.85  # CLIP cosine similarity threshold
HIST_MATCH_MIN_SIM      = 0.92  # fallback histogram cosine similarity threshold

# --- AR drawing ---
AR_MIN_POINTS          = 10
AR_SHAPE_LIFETIME      = 4.5    # seconds, used only in Auto persistence mode
AR_CLOSE_DIST_RATIO    = 0.30
PALM_HOLD_CLEAR_FRAMES = 45

# --- Eyes ---
EAR_BLINK_THRESHOLD = 0.21

# --- Tool dock / interaction ---
HOVER_CONFIRM_FRAMES = 28   # ~1s @ ~28fps of finger-hover to "tap" a dock button
HOVER_DECAY          = 0.12

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

NEON_CYAN    = (255, 255, 0)
NEON_MAGENTA = (255, 0, 255)
NEON_BLUE    = (255, 144, 30)

FILL_PALETTE = [NEON_CYAN, NEON_MAGENTA, NEON_BLUE, (0, 255, 120), (0, 90, 255)]
HUE_STEPS    = [20, 50, 80, 110, 140]   # paired index-for-index with FILL_PALETTE

SHAPE_COLORS = {
    "Circle": (0, 255, 255), "Triangle": (0, 255, 120), "Square": (255, 180, 0),
    "Rectangle": (255, 140, 0), "Pentagon": (200, 100, 255),
    "Freeform": (255, 255, 255), "Line": (120, 200, 255),
}

# ════════════════════════════════════════════════════════════════════════
#  STATIC KNOWLEDGE BASE — Company info
# ════════════════════════════════════════════════════════════════════════
COMPANY_DB = {
    "Apple":   {"company": "Apple Inc.", "founded": 1976, "hq": "Cupertino, USA", "fact": "Built in a garage by Jobs, Wozniak & Wayne."},
    "Samsung": {"company": "Samsung Electronics", "founded": 1969, "hq": "Suwon, South Korea", "fact": "Part of the larger Samsung Group."},
    "Google":  {"company": "Google LLC", "founded": 1998, "hq": "Mountain View, USA", "fact": "Pixel devices are designed in-house."},
    "OnePlus": {"company": "OnePlus Technology", "founded": 2013, "hq": "Shenzhen, China", "fact": "Known for 'Never Settle'."},
    "Xiaomi":  {"company": "Xiaomi Corporation", "founded": 2010, "hq": "Beijing, China", "fact": "Among the largest phone makers by volume."},
    "Dell":    {"company": "Dell Technologies", "founded": 1984, "hq": "Round Rock, USA", "fact": "Started as 'PC's Limited' from a dorm room."},
    "HP":      {"company": "HP Inc.", "founded": 1939, "hq": "Palo Alto, USA", "fact": "Founded in a one-car garage."},
    "Lenovo":  {"company": "Lenovo Group", "founded": 1984, "hq": "Beijing, China", "fact": "Bought IBM's PC/ThinkPad business in 2005."},
    "Asus":    {"company": "ASUSTeK Computer", "founded": 1989, "hq": "Taipei, Taiwan", "fact": "Named after Pegasus."},
    "Acer":    {"company": "Acer Inc.", "founded": 1976, "hq": "New Taipei, Taiwan", "fact": "Originally named Multitech."},
    "LG":      {"company": "LG Electronics", "founded": 1958, "hq": "Seoul, South Korea", "fact": "LG = 'Lucky Goldstar'."},
    "Sony":    {"company": "Sony Group", "founded": 1946, "hq": "Tokyo, Japan", "fact": "Started as a post-WWII repair company."},
    "TCL":     {"company": "TCL Technology", "founded": 1981, "hq": "Huizhou, China", "fact": "One of the largest TV makers by shipment."},
    "Logitech":{"company": "Logitech International", "founded": 1981, "hq": "Lausanne, Switzerland", "fact": "Made one of the first commercial mice."},
    "Corsair": {"company": "Corsair Gaming", "founded": 1994, "hq": "Fremont, USA", "fact": "Started out making memory modules."},
    "Razer":   {"company": "Razer Inc.", "founded": 2005, "hq": "Singapore/Irvine", "fact": "Known for the green-snake logo."},
    "Microsoft":{"company": "Microsoft Corporation", "founded": 1975, "hq": "Redmond, USA", "fact": "Co-founded by Gates and Allen."},
    "Coca-Cola":{"company": "The Coca-Cola Company", "founded": 1892, "hq": "Atlanta, USA", "fact": "Contour bottle trademarked in 1977."},
    "Pepsi":   {"company": "PepsiCo", "founded": 1965, "hq": "Purchase, USA", "fact": "Originally called 'Brad's Drink'."},
    "Bisleri": {"company": "Bisleri International", "founded": 1965, "hq": "Mumbai, India", "fact": "Became a generic term for bottled water in India."},
    "Starbucks":{"company": "Starbucks Corporation", "founded": 1971, "hq": "Seattle, USA", "fact": "Named after a character in Moby-Dick."},
    "Toyota":  {"company": "Toyota Motor Corp.", "founded": 1937, "hq": "Toyota City, Japan", "fact": "World's largest automaker by volume."},
    "Honda":   {"company": "Honda Motor Co.", "founded": 1948, "hq": "Tokyo, Japan", "fact": "Started by making motorized bicycles."},
    "BMW":     {"company": "Bayerische Motoren Werke", "founded": 1916, "hq": "Munich, Germany", "fact": "Logo nods to its aircraft-engine roots."},
    "Tesla":   {"company": "Tesla, Inc.", "founded": 2003, "hq": "Austin, USA", "fact": "Named after Nikola Tesla."},
    "Ford":    {"company": "Ford Motor Company", "founded": 1903, "hq": "Dearborn, USA", "fact": "Pioneered the moving assembly line."},
    "Nike":    {"company": "Nike, Inc.", "founded": 1964, "hq": "Beaverton, USA", "fact": "The Swoosh cost just $35."},
    "Adidas":  {"company": "Adidas AG", "founded": 1949, "hq": "Herzogenaurach, Germany", "fact": "Named after Adolf 'Adi' Dassler."},
    "The North Face": {"company": "The North Face, Inc.", "founded": 1966, "hq": "Denver, USA", "fact": "Named for a mountain's coldest face."},
    "Louis Vuitton": {"company": "Louis Vuitton", "founded": 1854, "hq": "Paris, France", "fact": "LV monogram created in 1896 vs counterfeits."},
    "Gucci":   {"company": "Gucci", "founded": 1921, "hq": "Florence, Italy", "fact": "Started as a leather-goods shop."},
    "Samsonite": {"company": "Samsonite International", "founded": 1910, "hq": "Mansfield, USA", "fact": "Named after the biblical Samson."},
    "American Tourister": {"company": "American Tourister", "founded": 1933, "hq": "Mansfield, USA", "fact": "Now owned by Samsonite."},
    "Whirlpool": {"company": "Whirlpool Corporation", "founded": 1911, "hq": "Benton Harbor, USA", "fact": "One of the largest appliance makers."},
    "Panasonic": {"company": "Panasonic Holdings", "founded": 1918, "hq": "Osaka, Japan", "fact": "Originally founded as Matsushita Electric."},
}

CATEGORY_PROMPTS = {
    "cell phone": [("Apple", "a photo of an Apple iPhone smartphone"), ("Samsung", "a photo of a Samsung Galaxy smartphone"),
                   ("Google", "a photo of a Google Pixel smartphone"), ("OnePlus", "a photo of a OnePlus smartphone"),
                   ("Xiaomi", "a photo of a Xiaomi smartphone"), (None, "a photo of an unbranded generic phone")],
    "laptop": [("Apple", "a photo of an Apple MacBook laptop"), ("Dell", "a photo of a Dell laptop"),
               ("HP", "a photo of an HP laptop"), ("Lenovo", "a photo of a Lenovo ThinkPad laptop"),
               ("Asus", "a photo of an Asus laptop"), ("Acer", "a photo of an Acer laptop"),
               (None, "a photo of an unbranded generic laptop")],
    "tv": [("Samsung", "a photo of a Samsung television"), ("LG", "a photo of an LG television"),
           ("Sony", "a photo of a Sony television"), ("TCL", "a photo of a TCL television"),
           (None, "a photo of an unbranded generic television")],
    "keyboard": [("Apple", "a photo of an Apple Magic Keyboard"), ("Logitech", "a photo of a Logitech keyboard"),
                 ("Corsair", "a photo of a Corsair mechanical keyboard"), ("Razer", "a photo of a Razer mechanical keyboard"),
                 (None, "a photo of an unbranded generic keyboard")],
    "mouse": [("Apple", "a photo of an Apple Magic Mouse"), ("Logitech", "a photo of a Logitech computer mouse"),
              ("Razer", "a photo of a Razer gaming mouse"), ("Microsoft", "a photo of a Microsoft computer mouse"),
              (None, "a photo of an unbranded generic computer mouse")],
    "bottle": [("Coca-Cola", "a photo of a Coca-Cola bottle"), ("Pepsi", "a photo of a Pepsi bottle"),
               ("Bisleri", "a photo of a Bisleri water bottle"), (None, "a photo of a plain unbranded water bottle")],
    "cup": [("Starbucks", "a photo of a Starbucks paper cup"), (None, "a photo of a plain unbranded cup or mug")],
    "car": [("Toyota", "a photo of a Toyota car"), ("Honda", "a photo of a Honda car"), ("BMW", "a photo of a BMW car"),
            ("Tesla", "a photo of a Tesla car"), ("Ford", "a photo of a Ford car"), (None, "a photo of an unidentifiable car")],
    "backpack": [("Nike", "a photo of a Nike backpack"), ("Adidas", "a photo of an Adidas backpack"),
                 ("The North Face", "a photo of a The North Face backpack"), (None, "a photo of an unbranded generic backpack")],
    "handbag": [("Louis Vuitton", "a photo of a Louis Vuitton handbag"), ("Gucci", "a photo of a Gucci handbag"),
                (None, "a photo of an unbranded generic handbag")],
    "suitcase": [("Samsonite", "a photo of a Samsonite suitcase"), ("American Tourister", "a photo of an American Tourister suitcase"),
                 (None, "a photo of an unbranded generic suitcase")],
    "refrigerator": [("Samsung", "a photo of a Samsung refrigerator"), ("LG", "a photo of an LG refrigerator"),
                      ("Whirlpool", "a photo of a Whirlpool refrigerator"), (None, "a photo of an unbranded generic refrigerator")],
    "microwave": [("Samsung", "a photo of a Samsung microwave oven"), ("LG", "a photo of an LG microwave oven"),
                  ("Panasonic", "a photo of a Panasonic microwave oven"), (None, "a photo of an unbranded generic microwave oven")],
}

# ════════════════════════════════════════════════════════════════════════
#  LANDMARK INDICES & GESTURES
# ════════════════════════════════════════════════════════════════════════
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_MCPS  = [2, 5, 9, 13, 17]
LEFT_EYE_EAR_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]

GESTURE_MAP = {
    (0, 0, 0, 0, 0): "Fist", (1, 1, 1, 1, 1): "Open Palm", (0, 1, 0, 0, 0): "Pointer",
    (0, 1, 1, 0, 0): "Peace", (1, 1, 0, 0, 0): "L-Shape", (1, 0, 0, 0, 1): "Call Me",
    (0, 1, 1, 1, 0): "Three", (0, 1, 1, 1, 1): "Four", (1, 0, 0, 0, 0): "Thumbs Up",
}


# ════════════════════════════════════════════════════════════════════════
#  BRAND IDENTIFIER  (CLIP zero-shot — optional)
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
            print("[INFO] Loading CLIP brand-identification model …")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.eval()
            self.available = True
            print(f"[INFO] Brand identification ready on {device}.")
        except Exception as e:
            print(f"[WARN] Brand identification disabled. Reason: {e}")

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
            return candidates[best_idx][0], float(probs[best_idx])
        except Exception:
            return None


# ════════════════════════════════════════════════════════════════════════
#  EMBEDDING ENGINE  (for Target Memory matching — graceful 3-tier fallback)
# ════════════════════════════════════════════════════════════════════════
class EmbeddingEngine:
    def __init__(self, device="cpu"):
        self.device = device
        self.face_backend = None
        self.clip_backend = False

        try:
            import importlib
            self.face_backend = importlib.import_module("face_recognition")
            print("[INFO] face_recognition available — accurate face memory enabled.")
        except Exception:
            self.face_backend = None
            print("[WARN] face_recognition not installed — face memory will use a lower-accuracy "
                  "fallback. For best results: pip install face_recognition")

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.torch = torch
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            self.clip_backend = True
            print("[INFO] CLIP available — object memory matching enabled.")
        except Exception:
            print("[WARN] CLIP not available — object memory will use a basic histogram "
                  "fallback (works, but least precise).")

    def face_embedding(self, rgb_frame, box_trbl):
        """box_trbl = (top, right, bottom, left) in full-frame pixel coords."""
        if self.face_backend:
            try:
                encs = self.face_backend.face_encodings(rgb_frame, known_face_locations=[box_trbl])
                if encs:
                    return "face", encs[0]
            except Exception:
                pass
        top, right, bottom, left = box_trbl
        crop = rgb_frame[max(0, top):bottom, max(0, left):right]
        return self._fallback_or_clip(crop)

    def object_embedding(self, bgr_crop):
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        return self._fallback_or_clip(rgb)

    def _fallback_or_clip(self, rgb_crop):
        if rgb_crop is None or rgb_crop.size == 0:
            return None
        if self.clip_backend:
            try:
                from PIL import Image
                image = Image.fromarray(rgb_crop)
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                with self.torch.no_grad():
                    feat = self.clip_model.get_image_features(**inputs)
                vec = feat[0].cpu().numpy()
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                return "clip", vec
            except Exception:
                pass
        small = cv2.resize(rgb_crop, (32, 32))
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        hist = hist / (np.linalg.norm(hist) + 1e-9)
        return "hist", hist


def embeddings_match(etype_a, vec_a, etype_b, vec_b):
    if etype_a != etype_b:
        return False, 0.0
    a, b = np.asarray(vec_a, dtype=np.float64), np.asarray(vec_b, dtype=np.float64)
    if etype_a == "face":
        dist = float(np.linalg.norm(a - b))
        return dist < FACE_MATCH_MAX_DIST, max(0.0, 1.0 - dist)
    sim = float(np.dot(a, b))
    if etype_a == "clip":
        return sim > CLIP_MATCH_MIN_SIM, sim
    return sim > HIST_MATCH_MIN_SIM, sim


# ════════════════════════════════════════════════════════════════════════
#  TARGET MEMORY DATABASE  (local persistence)
# ════════════════════════════════════════════════════════════════════════
class TargetMemoryDB:
    def __init__(self, root=MEMORY_DIR):
        self.root = root
        self.db_path = os.path.join(root, "db.json")
        os.makedirs(os.path.join(root, "attachments"), exist_ok=True)
        self.records = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    self.records = json.load(f)
                print(f"[MEMORY] Loaded {len(self.records)} saved target(s) from {self.db_path}")
            except Exception as e:
                print(f"[WARN] Could not load target memory DB: {e}")

    def _save(self):
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not save target memory DB: {e}")

    def add_target(self, name, description, tags, etype, vec, photo_paths=None, doc_paths=None):
        tid = str(uuid.uuid4())[:8]
        folder = os.path.join(self.root, "attachments", tid)
        os.makedirs(folder, exist_ok=True)
        saved_photos, saved_docs = [], []
        for p in (photo_paths or []):
            try:
                dest = os.path.join(folder, os.path.basename(p))
                shutil.copy(p, dest)
                saved_photos.append(dest)
            except Exception as e:
                print(f"[WARN] Could not attach photo {p}: {e}")
        for p in (doc_paths or []):
            try:
                dest = os.path.join(folder, os.path.basename(p))
                shutil.copy(p, dest)
                saved_docs.append(dest)
            except Exception as e:
                print(f"[WARN] Could not attach document {p}: {e}")

        record = {
            "id": tid, "name": name or "Unnamed Target", "description": description or "",
            "tags": tags or [], "etype": etype, "embedding": [float(x) for x in vec],
            "photos": saved_photos, "docs": saved_docs, "created": time.time(),
        }
        self.records.append(record)
        self._save()
        return record

    def find_match(self, etype, vec):
        best, best_score = None, 0.0
        for rec in self.records:
            if rec["etype"] != etype:
                continue
            ok, score = embeddings_match(etype, vec, rec["etype"], rec["embedding"])
            if ok and score > best_score:
                best, best_score = rec, score
        return best, best_score


# ════════════════════════════════════════════════════════════════════════
#  PERSISTENT TRACKER  (shared by objects AND faces — lightweight SORT-style)
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
                trk["bbox"] = (int(a * nx1 + (1 - a) * ex1), int(a * ny1 + (1 - a) * ey1),
                               int(a * nx2 + (1 - a) * ex2), int(a * ny2 + (1 - a) * ey2))
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
                "bbox": bbox, "label": label, "conf": conf, "confirmed": 1, "missed": 0,
                "brand": "?", "brand_conf": 0.0, "last_brand_check": -99999,
                "fill_color": None, "hue_shift": None, "effect_created": None,
                "memory_match": None, "memory_score": 0.0, "last_mem_check": -99999,
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
    approx = cv2.approxPolyDP(hull, 0.03 * perimeter, True).reshape(-1, 2)
    vcount = len(approx)
    cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    if not closed:
        step = max(1, len(pts) // 40)
        return {"name": "Line", "pts": pts[::step], "closed": False, "centroid": (cx, cy)}
    if vcount == 3:
        return {"name": "Triangle", "pts": approx, "closed": True, "centroid": (cx, cy)}
    if vcount == 4:
        rect = cv2.minAreaRect(hull)
        rw, rh = rect[1]
        ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
        box = cv2.boxPoints(rect)
        return {"name": "Square" if ratio < 1.25 else "Rectangle", "pts": box, "closed": True, "centroid": (cx, cy)}
    if vcount == 5:
        return {"name": "Pentagon", "pts": approx, "closed": True, "centroid": (cx, cy)}
    if circularity > 0.78:
        (ccx, ccy), radius = cv2.minEnclosingCircle(hull)
        circle_pts = np.array([[ccx + radius * math.cos(t), ccy + radius * math.sin(t)]
                                for t in np.linspace(0, 2 * math.pi, 40)])
        return {"name": "Circle", "pts": circle_pts, "closed": True, "centroid": (ccx, ccy)}
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


def tint_rect(frame, x1, y1, x2, y2, color, alpha=0.35):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    overlay = np.full_like(sub, color)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)
    frame[y1:y2, x1:x2] = sub


def apply_hue_shift(frame, bbox, hue_shift):
    x1, y1, x2, y2 = bbox
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    frame[y1:y2, x1:x2] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


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


def draw_scan_lock(frame, bbox, locked, color, label, t):
    x1, y1, x2, y2 = bbox
    if not locked:
        h_box = max(1, y2 - y1)
        sweep_y = y1 + int((math.sin(t * 4) * 0.5 + 0.5) * h_box)
        overlay = frame.copy()
        for gx in range(x1, x2, 18):
            cv2.line(overlay, (gx, y1), (gx, y2), color, 1, cv2.LINE_AA)
        for gy in range(y1, y2, 18):
            cv2.line(overlay, (x1, gy), (x2, gy), color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.line(frame, (x1, sweep_y), (x2, sweep_y), color, 2, cv2.LINE_AA)
        draw_brackets(frame, x1, y1, x2, y2, color, length=10, thickness=1)
        text(frame, f"{label}  SCANNING...", x1, y1 - 8, color=color, scale=0.45)
    else:
        pulse = 0.6 + 0.4 * math.sin(t * 3)
        col = tuple(int(c * pulse) for c in color)
        draw_brackets(frame, x1, y1, x2, y2, col, length=18, thickness=2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), col, 1)
        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), col, 1)
        text(frame, f"{label}  LOCKED", x1, y1 - 8, color=col, scale=0.48)


def draw_recall_card(frame, x1, y1, record, score, thumb):
    cw, ch = 300, 150
    glass_panel(frame, x1, y1, x1 + cw, y1 + ch, border=NEON_MAGENTA)
    text(frame, "TARGET RECOGNIZED", x1 + 10, y1 + 22, color=NEON_MAGENTA, scale=0.5, thick=1)
    text(frame, f"{record['name']}  ({score * 100:.0f}% match)", x1 + 10, y1 + 44, scale=0.46)
    if record.get("description"):
        text(frame, record["description"][:42], x1 + 10, y1 + 64, scale=0.4, color=C_DIM)
    if record.get("tags"):
        text(frame, "Tags: " + ", ".join(record["tags"][:4]), x1 + 10, y1 + 82, scale=0.38, color=C_DIM)
    text(frame, f"Files: {len(record.get('photos', []))} photo(s), {len(record.get('docs', []))} doc(s)",
         x1 + 10, y1 + 100, scale=0.38, color=C_DIM)
    if thumb is not None:
        th, tw = thumb.shape[:2]
        fx1, fy1 = x1 + cw - tw - 10, y1 + ch - th - 10
        frame[fy1:fy1 + th, fx1:fx1 + tw] = thumb
        cv2.rectangle(frame, (fx1, fy1), (fx1 + tw, fy1 + th), NEON_MAGENTA, 1)


# ════════════════════════════════════════════════════════════════════════
#  UI STATE + FLOATING TOOL DOCK
# ════════════════════════════════════════════════════════════════════════
class UIState:
    def __init__(self):
        self.current_tool = "Eraser"
        self.persistence_mode = "Auto"
        self.color_idx = 0
        self.selected = None             # (kind, track_id) or None
        self.mouse_click = None          # (x,y) set transiently by mouse callback
        self.hover_progress = defaultdict(float)
        self.add_details_requested = False
        self.target_memory_on = ENABLE_TARGET_MEMORY


def get_dock_layout():
    x1, x2 = 10, 180
    y = 90
    layout = {}
    for name in ["Eraser", "Clear", "Color Fill", "Color Changer"]:
        layout[name] = (x1, y, x2, y + 42)
        y += 50
    layout["Persistence"] = (x1, y, x2, y + 38)
    y += 46
    layout["Add Details"] = (x1, y, x2, y + 38)
    y += 50
    sw = 26
    for i in range(len(FILL_PALETTE)):
        cx1 = x1 + i * (sw + 6)
        layout[f"color_{i}"] = (cx1, y, cx1 + sw, y + sw)
    return layout


def draw_dock(frame, layout, ui):
    for name, (x1, y1, x2, y2) in layout.items():
        if name.startswith("color_"):
            idx = int(name.split("_")[1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), FILL_PALETTE[idx], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 255, 255) if ui.color_idx == idx else (70, 70, 70),
                          2 if ui.color_idx == idx else 1)
            continue
        if name == "Persistence":
            label, accent = f"MODE: {ui.persistence_mode.upper()}", NEON_BLUE
        elif name == "Add Details":
            label, accent = "ADD DETAILS", (NEON_MAGENTA if ui.selected else (70, 70, 70))
        else:
            label, accent = name.upper(), (NEON_CYAN if name == ui.current_tool else (90, 90, 90))
        glass_panel(frame, x1, y1, x2, y2, alpha=0.6, color=C_BG, border=accent)
        text(frame, label, x1 + 8, (y1 + y2) // 2 + 5, color=accent, scale=0.42)
        prog = ui.hover_progress.get(name, 0.0)
        if 0 < prog < 1.0:
            cv2.rectangle(frame, (x1, y2 - 4), (x1 + int((x2 - x1) * prog), y2 - 1), NEON_MAGENTA, -1)


def hit_test_dock(x, y, layout):
    for name, (x1, y1, x2, y2) in layout.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None


def on_mouse(event, x, y, flags, ui_state):
    if event == cv2.EVENT_LBUTTONDOWN:
        ui_state.mouse_click = (x, y)


def activate_button(name, ui):
    if name in ("Eraser", "Clear", "Color Fill", "Color Changer"):
        ui.current_tool = name
    elif name == "Persistence":
        ui.persistence_mode = "Manual" if ui.persistence_mode == "Auto" else "Auto"
    elif name == "Add Details":
        ui.add_details_requested = True
    elif name.startswith("color_"):
        ui.color_idx = int(name.split("_")[1])


def _inside(pt, bbox):
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def remove_nearest_ghost_shape(px, py, ghost_shapes, max_dist=60):
    best_i, best_d = None, max_dist
    for i, gs in enumerate(ghost_shapes):
        cx, cy = gs["centroid"]
        d = math.hypot(px - cx, py - cy)
        if d < best_d:
            best_d, best_i = d, i
    if best_i is not None:
        del ghost_shapes[best_i]


def handle_canvas_click(px, py, ui, object_tracks, face_tracks, ghost_shapes):
    candidates = []
    for tid, trk in object_tracks.items():
        if trk["missed"] == 0 and trk["confirmed"] >= CONFIRM_THRESHOLD:
            candidates.append(("object", tid, trk["bbox"]))
    for tid, trk in face_tracks.items():
        if trk["missed"] == 0 and trk["confirmed"] >= CONFIRM_THRESHOLD:
            candidates.append(("face", tid, trk["bbox"]))

    hit, best_area = None, None
    for kind, tid, bbox in candidates:
        if _inside((px, py), bbox):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if best_area is None or area < best_area:
                best_area, hit = area, (kind, tid, bbox)

    tool = ui.current_tool
    if hit:
        kind, tid, bbox = hit
        trk = (object_tracks if kind == "object" else face_tracks)[tid]
        if tool == "Color Fill":
            trk["fill_color"], trk["effect_created"] = FILL_PALETTE[ui.color_idx], time.time()
        elif tool == "Color Changer":
            trk["hue_shift"], trk["effect_created"] = HUE_STEPS[ui.color_idx], time.time()
        elif tool == "Eraser":
            trk["fill_color"], trk["hue_shift"] = None, None
        elif tool == "Clear":
            trk["fill_color"], trk["hue_shift"] = None, None
            ghost_shapes[:] = [gs for gs in ghost_shapes if not _inside(gs["centroid"], bbox)]
        ui.selected = (kind, tid)
    else:
        if tool == "Eraser":
            remove_nearest_ghost_shape(px, py, ghost_shapes)
        ui.selected = None


# ════════════════════════════════════════════════════════════════════════
#  ADD-DETAILS DIALOG (Tkinter) + memory helpers
# ════════════════════════════════════════════════════════════════════════
def show_add_details_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        print(f"[WARN] Tkinter unavailable, can't open Add Details dialog: {e}")
        return None

    result = {}
    root = tk.Tk()
    root.title("Target Memory — Add Details")
    root.geometry("440x420")
    root.attributes("-topmost", True)

    tk.Label(root, text="Name").pack(anchor="w", padx=12, pady=(12, 0))
    name_var = tk.StringVar()
    tk.Entry(root, textvariable=name_var, width=48).pack(padx=12)

    tk.Label(root, text="Description").pack(anchor="w", padx=12, pady=(10, 0))
    desc_text = tk.Text(root, width=48, height=5)
    desc_text.pack(padx=12)

    tk.Label(root, text="Tags (comma separated)").pack(anchor="w", padx=12, pady=(10, 0))
    tags_var = tk.StringVar()
    tk.Entry(root, textvariable=tags_var, width=48).pack(padx=12)

    photo_paths, doc_paths = [], []
    status_var = tk.StringVar(value="No files attached.")

    def pick_photos():
        paths = filedialog.askopenfilenames(title="Select reference photos",
                                             filetypes=[("Images", "*.jpg *.jpeg *.png")])
        photo_paths.extend(paths)
        status_var.set(f"{len(photo_paths)} photo(s), {len(doc_paths)} doc(s) attached.")

    def pick_docs():
        paths = filedialog.askopenfilenames(title="Select documents",
                                             filetypes=[("Documents", "*.pdf *.txt *.docx *.md")])
        doc_paths.extend(paths)
        status_var.set(f"{len(photo_paths)} photo(s), {len(doc_paths)} doc(s) attached.")

    tk.Button(root, text="Attach Photos...", command=pick_photos).pack(pady=(14, 2))
    tk.Button(root, text="Attach Documents...", command=pick_docs).pack(pady=2)
    tk.Label(root, textvariable=status_var, fg="gray").pack(pady=6)

    def submit():
        result["name"] = name_var.get().strip() or "Unnamed Target"
        result["description"] = desc_text.get("1.0", "end").strip()
        result["tags"] = [t.strip() for t in tags_var.get().split(",") if t.strip()]
        result["photos"] = list(photo_paths)
        result["docs"] = list(doc_paths)
        root.destroy()

    def cancel():
        result.clear()
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=16)
    tk.Button(btn_frame, text="Save Target", command=submit, bg="#00c2c7").pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel", command=cancel).pack(side="left", padx=6)

    root.mainloop()
    return result if result else None


def handle_add_details(frame, rgb, ui, object_tracks, face_tracks, embed_engine, memory_db):
    if not ui.selected:
        print("[INFO] Click a locked target first to select it before adding details.")
        return
    kind, tid = ui.selected
    trk = (object_tracks if kind == "object" else face_tracks).get(tid)
    if not trk:
        print("[INFO] Selected target is no longer visible.")
        return
    x1, y1, x2, y2 = trk["bbox"]
    if kind == "face":
        res = embed_engine.face_embedding(rgb, (max(0, y1), x2, y2, max(0, x1)))
    else:
        res = embed_engine.object_embedding(frame[max(0, y1):y2, max(0, x1):x2])
    if res is None:
        print("[WARN] Could not compute an embedding for this target.")
        return
    etype, vec = res

    details = show_add_details_dialog()
    if not details:
        return
    record = memory_db.add_target(details["name"], details["description"], details["tags"],
                                   etype, vec, details["photos"], details["docs"])
    trk["memory_match"], trk["memory_score"] = record, 1.0
    print(f"[MEMORY] Saved target '{record['name']}' (id={record['id']}).")


def update_memory_recognition(frame, rgb, tracks, kind, embed_engine, memory_db, frame_count):
    for tid, trk in tracks.items():
        if trk["missed"] > 0 or trk["confirmed"] < CONFIRM_THRESHOLD:
            continue
        if frame_count - trk.get("last_mem_check", -99999) < TARGET_RECHECK_INTERVAL:
            continue
        trk["last_mem_check"] = frame_count
        x1, y1, x2, y2 = trk["bbox"]
        if kind == "face":
            res = embed_engine.face_embedding(rgb, (max(0, y1), x2, y2, max(0, x1)))
        else:
            res = embed_engine.object_embedding(frame[max(0, y1):y2, max(0, x1):x2])
        if res is None:
            continue
        etype, vec = res
        match, score = memory_db.find_match(etype, vec)
        trk["memory_match"], trk["memory_score"] = (match, score) if match else (None, 0.0)


def get_thumbnail(record, cache, size=64):
    rid = record["id"]
    if rid in cache:
        return cache[rid]
    img = None
    if record.get("photos"):
        p = record["photos"][0]
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                img = cv2.resize(im, (size, size))
    cache[rid] = img
    return img


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
    return (d(1, 5) + d(2, 4)) / (2 * d(0, 3) + 1e-6)


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

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
    embed_engine = EmbeddingEngine(device=device) if ENABLE_TARGET_MEMORY else None
    memory_db = TargetMemoryDB() if ENABLE_TARGET_MEMORY else None

    object_tracker = ObjectTracker()
    face_tracker = ObjectTracker()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try changing CAMERA_INDEX at the top of the file.")
        return

    cv2.namedWindow(WINDOW_NAME)
    ui = UIState()
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, ui)

    show_hud, show_objects, show_brand, show_ar = True, True, ENABLE_BRAND_ID, True
    trails = defaultdict(lambda: deque(maxlen=400))
    drawing_state, palm_hold = defaultdict(bool), defaultdict(int)
    ghost_shapes, thumb_cache, last_detections = [], {}, []

    fps, frame_count, tick, shot_count = 0.0, 0, time.time(), 0

    print("\n[RUNNING] Q quit|S shot|H hud|O objects|B brand|T memory|A ar|C clear|P persist|1-4 tool|[ ] color|D add-details\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_anim = time.time()

        frame_count += 1
        if frame_count % 15 == 0:
            fps, tick = 15.0 / max(time.time() - tick, 1e-9), time.time()

        dock_layout = get_dock_layout()

        # ── 0. Mouse click routing ───────────────────────────────────────
        if ui.mouse_click:
            mx, my = ui.mouse_click
            btn = hit_test_dock(mx, my, dock_layout)
            if btn:
                activate_button(btn, ui)
            else:
                handle_canvas_click(mx, my, ui, object_tracker.tracks, face_tracker.tracks, ghost_shapes)
            ui.mouse_click = None

        # ── 1. YOLO (every Nth frame) + object tracker ───────────────────
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
        object_tracks = object_tracker.update(detections, frame_count)

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

                finger_ups = [is_finger_up(lms, tp, mc, side) for tp, mc in zip(FINGER_TIPS, FINGER_MCPS)]
                gesture = GESTURE_MAP.get(tuple(1 if f else 0 for f in finger_ups), "Unknown")
                hand_hud.append((side, gesture, finger_ups))

                for tip, up in zip(FINGER_TIPS, finger_ups):
                    lm = lms.landmark[tip]
                    tx, ty = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (tx, ty), 7, C_GOOD if up else C_BAD, -1)
                    cv2.circle(frame, (tx, ty), 7, (255, 255, 255), 1)

                index_tip = lms.landmark[FINGER_TIPS[1]]
                fx, fy = int(index_tip.x * w), int(index_tip.y * h)

                # — Finger-hover dock interaction (Pointer gesture) ——————
                hovered_btn = hit_test_dock(fx, fy, dock_layout) if gesture == "Pointer" else None
                for name in list(ui.hover_progress.keys()):
                    if name != hovered_btn:
                        ui.hover_progress[name] = max(0.0, ui.hover_progress[name] - HOVER_DECAY)
                if hovered_btn:
                    ui.hover_progress[hovered_btn] += 1.0 / HOVER_CONFIRM_FRAMES
                    if ui.hover_progress[hovered_btn] >= 1.0:
                        activate_button(hovered_btn, ui)
                        ui.hover_progress[hovered_btn] = 0.0
                    else:
                        ring = ui.hover_progress[hovered_btn]
                        cv2.ellipse(frame, (fx, fy), (16, 16), -90, 0, int(360 * ring), NEON_MAGENTA, 2)

                # — AR drawing state machine ————————————————————————————
                if show_ar:
                    if gesture == "Pointer" and not hovered_btn:
                        trails[side].append((fx, fy))
                        drawing_state[side] = True
                        palm_hold[side] = 0
                    else:
                        if drawing_state[side] and len(trails[side]) >= AR_MIN_POINTS:
                            shape = classify_shape(list(trails[side]))
                            if shape:
                                shape["color"] = SHAPE_COLORS.get(shape["name"], (255, 255, 255))
                                shape["created"] = time.time()
                                shape["persist"] = (ui.persistence_mode == "Manual")
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
        face_detections, eye_state = [], []
        fres = face_mesh.process(rgb)
        face_lms_by_bbox = {}
        if fres.multi_face_landmarks:
            for flms in fres.multi_face_landmarks:
                xs = [lm.x * w for lm in flms.landmark]
                ys = [lm.y * h for lm in flms.landmark]
                fx1, fy1 = max(0, int(min(xs)) - 10), max(0, int(min(ys)) - 10)
                fx2, fy2 = min(w - 1, int(max(xs)) + 10), min(h - 1, int(max(ys)) + 10)
                face_detections.append(((fx1, fy1, fx2, fy2), "face", 1.0))
                face_lms_by_bbox[(fx1, fy1, fx2, fy2)] = flms

                mp_drawing.draw_landmarks(frame, flms, mp_face_mesh.FACEMESH_TESSELATION,
                                           None, mp_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(frame, flms, mp_face_mesh.FACEMESH_IRISES,
                                           None, mp_styles.get_default_face_mesh_iris_connections_style())

                left_ear = eye_aspect_ratio(flms, LEFT_EYE_EAR_IDX, w, h)
                right_ear = eye_aspect_ratio(flms, RIGHT_EYE_EAR_IDX, w, h)
                eye_state.append((left_ear + right_ear) / 2 < EAR_BLINK_THRESHOLD)
        face_tracks = face_tracker.update(face_detections, frame_count)

        # ── 4. Memory recognition (throttled) ────────────────────────────
        if ui.target_memory_on and embed_engine is not None:
            update_memory_recognition(frame, rgb, object_tracks, "object", embed_engine, memory_db, frame_count)
            update_memory_recognition(frame, rgb, face_tracks, "face", embed_engine, memory_db, frame_count)

        # ── 5. Add-details request (opens blocking Tk dialog) ────────────
        if ui.add_details_requested:
            ui.add_details_requested = False
            if embed_engine is not None and memory_db is not None:
                handle_add_details(frame, rgb, ui, object_tracks, face_tracks, embed_engine, memory_db)
            else:
                print("[INFO] Target Memory is disabled (ENABLE_TARGET_MEMORY=False).")

        # ── 6. Draw objects: scan/lock + brand + effects + memory ────────
        visible_tracks, info_card, recall_card = [], None, None
        if show_objects:
            for tid, trk in object_tracks.items():
                if trk["missed"] > 0:
                    continue
                bx1, by1, bx2, by2 = trk["bbox"]
                locked = trk["confirmed"] >= CONFIRM_THRESHOLD
                in_hand = any(boxes_overlap(trk["bbox"], hb) for hb in hand_boxes)
                color = C_OBJ_HAND if in_hand else C_OBJ

                if trk.get("fill_color"):
                    if ui.persistence_mode == "Auto" and trk.get("effect_created") and \
                            time.time() - trk["effect_created"] > AR_SHAPE_LIFETIME:
                        trk["fill_color"] = None
                    else:
                        tint_rect(frame, bx1, by1, bx2, by2, trk["fill_color"])
                if trk.get("hue_shift") is not None:
                    if ui.persistence_mode == "Auto" and trk.get("effect_created") and \
                            time.time() - trk["effect_created"] > AR_SHAPE_LIFETIME:
                        trk["hue_shift"] = None
                    else:
                        apply_hue_shift(frame, (bx1, by1, bx2, by2), trk["hue_shift"])

                if locked and show_brand and trk["label"] in CATEGORY_PROMPTS and \
                        (frame_count - trk["last_brand_check"] > BRAND_RECHECK_INTERVAL):
                    crop = frame[max(0, by1):by2, max(0, bx1):bx2]
                    result = brand_id.classify(crop, trk["label"]) if show_brand else None
                    trk["last_brand_check"] = frame_count
                    if result:
                        trk["brand"], trk["brand_conf"] = result

                draw_scan_lock(frame, trk["bbox"], locked, color, trk["label"].upper(), t_anim)
                if in_hand:
                    text(frame, "IN HAND", bx1, by2 + 16, color=color, scale=0.42)
                if ui.selected == ("object", tid):
                    draw_brackets(frame, bx1 - 4, by1 - 4, bx2 + 4, by2 + 4, NEON_MAGENTA, length=10, thickness=1)

                brand = trk.get("brand")
                if brand and brand != "?" and trk["brand_conf"] >= BRAND_MIN_CONFIDENCE:
                    text(frame, brand, bx1, by2 + 32, color=(255, 255, 255), scale=0.42)
                    if in_hand and (info_card is None or trk["brand_conf"] > info_card["brand_conf"]):
                        info_card = trk

                if locked:
                    visible_tracks.append((trk, in_hand))
                    if trk.get("memory_match") and (recall_card is None or trk["memory_score"] > recall_card[1]):
                        recall_card = (trk["memory_match"], trk["memory_score"])

        # ── 7. Draw faces: scan/lock + memory ─────────────────────────────
        for tid, trk in face_tracks.items():
            if trk["missed"] > 0:
                continue
            locked = trk["confirmed"] >= CONFIRM_THRESHOLD
            bx1, by1, bx2, by2 = trk["bbox"]
            color = C_FACE
            if trk.get("fill_color"):
                tint_rect(frame, bx1, by1, bx2, by2, trk["fill_color"])
            if trk.get("hue_shift") is not None:
                apply_hue_shift(frame, (bx1, by1, bx2, by2), trk["hue_shift"])
            draw_scan_lock(frame, trk["bbox"], locked, color, "FACE", t_anim)
            if ui.selected == ("face", tid):
                draw_brackets(frame, bx1 - 4, by1 - 4, bx2 + 4, by2 + 4, NEON_MAGENTA, length=10, thickness=1)
            if locked and trk.get("memory_match") and (recall_card is None or trk["memory_score"] > recall_card[1]):
                recall_card = (trk["memory_match"], trk["memory_score"])

        # ── 8. AR overlay — trails + finished ghost shapes ────────────────
        if show_ar:
            for pts in trails.values():
                if len(pts) > 1:
                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)
            keep = []
            for gs in ghost_shapes:
                if gs.get("persist"):
                    draw_ghost_shape(frame, gs, 1.0)
                    keep.append(gs)
                else:
                    age = time.time() - gs["created"]
                    if age < AR_SHAPE_LIFETIME:
                        draw_ghost_shape(frame, gs, 1 - age / AR_SHAPE_LIFETIME)
                        keep.append(gs)
            ghost_shapes[:] = keep

        # ── 9. HUD + dock ─────────────────────────────────────────────────
        if show_hud:
            glass_panel(frame, 0, 0, w, 36)
            text(frame, "OBJECT INTEL SYSTEM v3", 12, 24, color=C_ACCENT, scale=0.6, thick=2)
            text(frame, f"FPS {fps:4.1f}", w - 110, 24, color=C_GOOD if fps > 12 else C_BAD, scale=0.55)

            lx = 12
            led(frame, lx, 56, show_objects, "OBJECTS"); lx += 100
            led(frame, lx, 56, show_brand and brand_id.available, "BRAND"); lx += 90
            led(frame, lx, 56, ui.target_memory_on and embed_engine is not None, "MEMORY"); lx += 100
            led(frame, lx, 56, show_ar, "AR DRAW"); lx += 100
            led(frame, lx, 56, True, f"FACES {len(face_tracks)}")

            draw_dock(frame, dock_layout, ui)

            panel_w = 250
            glass_panel(frame, w - panel_w, 80, w, 80 + 26 + len(visible_tracks[:7]) * 46)
            text(frame, "DETECTED OBJECTS", w - panel_w + 10, 102, color=C_ACCENT, scale=0.5)
            for i, (trk, in_hand) in enumerate(sorted(visible_tracks, key=lambda t: -t[0]["conf"])[:7]):
                yy = 130 + i * 46
                col = C_OBJ_HAND if in_hand else C_OBJ
                cv2.circle(frame, (w - panel_w + 16, yy - 5), 4, col, -1)
                text(frame, trk["label"] + (" *" if in_hand else ""), w - panel_w + 26, yy, scale=0.45)
                confidence_bar(frame, w - panel_w + 26, yy + 6, panel_w - 50, 6, trk["conf"], col)

            gy = h - 20 - len(hand_hud) * 22
            for side, gesture, _ in hand_hud:
                text(frame, f"{side} hand: {gesture}", 14, gy, color=C_HAND, scale=0.48)
                gy += 22
            if eye_state:
                text(frame, "Eyes: CLOSED" if any(eye_state) else "Eyes: OPEN", 14, h - 14, color=C_EYE, scale=0.48)

            if info_card is not None:
                cw, ch = 320, 110
                cx1, cy1 = w - cw - 12, h - ch - 12
                glass_panel(frame, cx1, cy1, cx1 + cw, cy1 + ch, border=C_OBJ_HAND)
                comp = COMPANY_DB.get(info_card["brand"], {})
                text(frame, "DEVICE INFO", cx1 + 10, cy1 + 22, color=C_OBJ_HAND, scale=0.5)
                text(frame, f"{info_card['label'].title()} -> {info_card['brand']}", cx1 + 10, cy1 + 44, scale=0.44)
                text(frame, f"{comp.get('company', '-')} | Founded {comp.get('founded', '-')}", cx1 + 10, cy1 + 64, scale=0.4, color=C_DIM)
                text(frame, f"{comp.get('fact', '')}", cx1 + 10, cy1 + 84, scale=0.38, color=C_DIM)

            if recall_card is not None:
                record, score = recall_card
                thumb = get_thumbnail(record, thumb_cache)
                draw_recall_card(frame, 12, h - 162, record, score, thumb)

            text(frame, "Q quit S shot H hud O obj B brand T mem A ar C clear P persist 1-4 tool [ ] color D add",
                 12, h - 4, color=C_DIM, scale=0.36)

        cv2.imshow(WINDOW_NAME, frame)
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
        elif key == ord('t'):
            ui.target_memory_on = not ui.target_memory_on
        elif key == ord('a'):
            show_ar = not show_ar
        elif key == ord('c'):
            ghost_shapes.clear()
            for k in trails:
                trails[k].clear()
        elif key == ord('p'):
            ui.persistence_mode = "Manual" if ui.persistence_mode == "Auto" else "Auto"
        elif key == ord('d'):
            ui.add_details_requested = True
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            ui.current_tool = ["Eraser", "Clear", "Color Fill", "Color Changer"][key - ord('1')]
        elif key == ord('['):
            ui.color_idx = (ui.color_idx - 1) % len(FILL_PALETTE)
        elif key == ord(']'):
            ui.color_idx = (ui.color_idx + 1) % len(FILL_PALETTE)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()