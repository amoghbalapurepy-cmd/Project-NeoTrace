"""
╔══════════════════════════════════════════════════════════════════╗
║           REAL-TIME OBJECT IDENTIFIER & TRACKER                 ║
║  • Finger tracking    (MediaPipe Hands – 21 landmarks/hand)     ║
║  • Face mesh          (MediaPipe FaceMesh – 468 landmarks)      ║
║  • Eye + Iris         (MediaPipe refined landmarks)             ║
║  • Scene objects      (YOLOv8 – 80 COCO classes)               ║
║  • In-hand detection  (YOLO bbox ∩ Hand bbox)                   ║
╚══════════════════════════════════════════════════════════════════╝

Requirements:
    pip install opencv-python mediapipe ultralytics numpy

Run:
    python object_identifier.py

Controls:
    Q  →  Quit
    S  →  Save screenshot
    H  →  Toggle HUD
    O  →  Toggle YOLO objects
"""

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import time
import os

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX          = 0          # 0 = default webcam; change if needed
FRAME_WIDTH           = 1280
FRAME_HEIGHT          = 720
YOLO_CONF_THRESHOLD   = 0.40       # minimum YOLO confidence to show a box
MAX_HANDS             = 2
MAX_FACES             = 2
SCREENSHOT_DIR        = "screenshots"

# Color palette  (BGR)
COL_HAND_BOX    = (0, 255, 140)   # teal-green  – hand bounding box
COL_FINGER_UP   = (0, 255, 60)    # bright green – raised finger tip
COL_FINGER_DOWN = (0, 60, 255)    # red-orange   – closed finger tip
COL_FACE_BOX    = (11, 205, 242) # white        – face bounding box
COL_EYE         = (0, 220, 255)   # cyan         – eye outline
COL_OBJ_NORMAL  = (180, 180, 0)   # yellow       – generic object
COL_OBJ_INHAND  = (0, 100, 255)   # orange-red   – object held in hand
COL_HUD_BG      = (15, 15, 15)    # near-black   – HUD background
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────────────────────────────────────
#  LANDMARK INDICES
# ─────────────────────────────────────────────────────────────────────────────
FINGER_NAMES    = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_TIPS     = [4,  8,  12, 16, 20]   # tip landmark indices
FINGER_PIPS     = [3,  7,  11, 15, 19]   # PIP joint (second joint)
FINGER_MCPS     = [2,  5,   9, 13, 17]   # MCP joint (knuckle)

# Face Mesh – eye outline indices
LEFT_EYE_IDX    = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX   = [362, 385, 387, 263, 373, 380]

# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────────────────────────
mp_hands     = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.55,
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_FACES,
    refine_landmarks=True,          # enables iris tracking
    min_detection_confidence=0.50,
    min_tracking_confidence=0.50,
)

# ─────────────────────────────────────────────────────────────────────────────
#  YOLO SETUP
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Loading YOLOv8 model …")
yolo = YOLO("yolov8n.pt")           # downloads ~6 MB on first run
print("[INFO] Model loaded.")

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def is_finger_up(lms, tip_idx, mcp_idx, handedness: str) -> bool:
    """Return True if the finger is extended (raised)."""
    tip = lms.landmark[tip_idx]
    mcp = lms.landmark[mcp_idx]
    if tip_idx == 4:                # Thumb – compare X axis
        return tip.x < mcp.x if handedness == "Right" else tip.x > mcp.x
    return tip.y < mcp.y           # Other fingers – compare Y axis


def hand_bbox(lms, w: int, h: int, pad: int = 25):
    """Return (x1, y1, x2, y2) pixel bounding box around a hand."""
    xs = [lm.x * w for lm in lms.landmark]
    ys = [lm.y * h for lm in lms.landmark]
    return (
        max(0,     int(min(xs)) - pad),
        max(0,     int(min(ys)) - pad),
        min(w - 1, int(max(xs)) + pad),
        min(h - 1, int(max(ys)) + pad),
    )


def boxes_overlap(a, b) -> bool:
    """Return True if bounding boxes a and b overlap (IoU > 0)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, r=12):
    """Draw a rectangle with rounded corners."""
    pts = [(x1+r,y1),(x2-r,y1),(x2,y1+r),(x2,y2-r),(x2-r,y2),(x1+r,y2),(x1,y2-r),(x1,y1+r)]
    cv2.rectangle(img, (x1+r, y1),   (x2-r, y2),   color, thickness)
    cv2.rectangle(img, (x1,   y1+r), (x2,   y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0, 90, color, thickness)


def label_box(img, text: str, x: int, y: int,
              bg=(30,30,30), fg=(255,255,255), scale=0.50, thick=1):
    """Draw a filled label with text above a point."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 6, y + base + 1), bg, -1)
    cv2.putText(img, text, (x + 3, y), FONT, scale, fg, thick, cv2.LINE_AA)


def alpha_fill(img, x1, y1, x2, y2, color, alpha=0.25):
    """Semi-transparent filled rectangle."""
    sub = img[y1:y2, x1:x2]
    overlay = np.full_like(sub, color[::-1] if len(color)==3 else color)
    cv2.addWeighted(overlay, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Change CAMERA_INDEX at the top of the file.")
        return

    # State
    fps          = 0.0
    frame_count  = 0
    tick         = time.time()
    show_hud     = True
    show_objects = True
    shot_count   = 0

    print("\n[RUNNING]  Press Q to quit  |  S to screenshot  |  H toggle HUD  |  O toggle YOLO\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)          # mirror so left = left
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── FPS counter ─────────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 15 == 0:
            fps  = 15.0 / max(time.time() - tick, 1e-9)
            tick = time.time()

        # ════════════════════════════════════════════════════════════════════
        #  1.  YOLO – detect all objects in the scene
        # ════════════════════════════════════════════════════════════════════
        yolo_detections = []
        if show_objects:
            results = yolo(frame, verbose=False, conf=YOLO_CONF_THRESHOLD)[0]
            for box in results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label  = yolo.names[cls_id]
                conf   = float(box.conf[0])
                yolo_detections.append((bx1, by1, bx2, by2, label, conf))

        # ════════════════════════════════════════════════════════════════════
        #  2.  MEDIAPIPE HANDS – fingers + hand landmarks
        # ════════════════════════════════════════════════════════════════════
        hand_boxes  = []    # pixel bboxes of each detected hand
        hand_info_list = [] # (handedness_str, [finger_up bools])

        hand_results = hands.process(rgb)
        if hand_results.multi_hand_landmarks:
            for lms, info in zip(
                hand_results.multi_hand_landmarks,
                hand_results.multi_handedness,
            ):
                side = info.classification[0].label   # "Left" / "Right"

                # — Draw skeleton ——————————————————————————————————————————
                mp_drawing.draw_landmarks(
                    frame, lms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                # — Bounding box ——————————————————————————————————————————
                hx1, hy1, hx2, hy2 = hand_bbox(lms, w, h)
                hand_boxes.append((hx1, hy1, hx2, hy2))
                alpha_fill(frame, hx1, hy1, hx2, hy2, COL_HAND_BOX, alpha=0.10)
                draw_rounded_rect(frame, hx1, hy1, hx2, hy2, COL_HAND_BOX, 2)
                label_box(frame, f"✋ {side} Hand",
                          hx1, hy1 - 6, bg=COL_HAND_BOX, fg=(0,0,0), scale=0.55)

                # — Finger state ——————————————————————————————————————————
                finger_ups = []
                for i, (tip, mcp) in enumerate(zip(FINGER_TIPS, FINGER_MCPS)):
                    up = is_finger_up(lms, tip, mcp, side)
                    finger_ups.append(up)

                    # Coloured tip dot
                    tip_lm = lms.landmark[tip]
                    tx, ty = int(tip_lm.x * w), int(tip_lm.y * h)
                    dot_col = COL_FINGER_UP if up else COL_FINGER_DOWN
                    cv2.circle(frame, (tx, ty), 9, dot_col, -1)
                    cv2.circle(frame, (tx, ty), 9, (255,255,255), 1)
                    label_box(frame, FINGER_NAMES[i][0], tx - 5, ty - 12,
                              bg=dot_col, fg=(255,255,255), scale=0.35)

                hand_info_list.append((side, finger_ups))

                # — Finger summary near wrist —————————————————————————————
                wrist = lms.landmark[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                up_names = [FINGER_NAMES[i] for i, u in enumerate(finger_ups) if u]
                summary  = ", ".join(up_names) if up_names else "Fist ✊"
                label_box(frame, summary, wx, wy + 22,
                          bg=(40, 40, 40), scale=0.42)

        # ════════════════════════════════════════════════════════════════════
        #  3.  MEDIAPIPE FACE MESH – face + eyes + iris
        # ════════════════════════════════════════════════════════════════════
        face_count   = 0
        face_results = face_mesh.process(rgb)

        if face_results.multi_face_landmarks:
            face_count = len(face_results.multi_face_landmarks)
            for face_lms in face_results.multi_face_landmarks:

                # — Mesh tessellation (subtle grey grid) ——————————————————
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lms,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )
                # — Contours ——————————————————————————————————————————————
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lms,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
                )
                # — Irises ————————————————————————————————————————————————
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lms,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
                )

                # — Face bounding box ————————————————————————————————————
                all_x = [lm.x * w for lm in face_lms.landmark]
                all_y = [lm.y * h for lm in face_lms.landmark]
                fx1 = max(0,     int(min(all_x)) - 12)
                fy1 = max(0,     int(min(all_y)) - 12)
                fx2 = min(w - 1, int(max(all_x)) + 12)
                fy2 = min(h - 1, int(max(all_y)) + 12)
                draw_rounded_rect(frame, fx1, fy1, fx2, fy2, COL_FACE_BOX, 2)
                label_box(frame, "Face 😊", fx1, fy1 - 6,
                          bg=COL_FACE_BOX, fg=(0, 0, 0), scale=0.55)

                # — Eye outlines ——————————————————————————————————————————
                for eye_idxs, eye_name in [(LEFT_EYE_IDX, "L.Eye"),
                                            (RIGHT_EYE_IDX, "R.Eye")]:
                    pts = np.array([
                        [int(face_lms.landmark[i].x * w),
                         int(face_lms.landmark[i].y * h)]
                        for i in eye_idxs
                    ], dtype=np.int32)
                    cv2.polylines(frame, [pts], True, COL_EYE, 1, cv2.LINE_AA)
                    label_box(frame, eye_name,
                              pts[:, 0].min(), pts[:, 1].min() - 4,
                              bg=COL_EYE, fg=(0, 0, 0), scale=0.38)

        # ════════════════════════════════════════════════════════════════════
        #  4.  Draw YOLO detections – highlight if held in hand
        # ════════════════════════════════════════════════════════════════════
        in_hand_count = 0
        if show_objects:
            for (bx1, by1, bx2, by2, label, conf) in yolo_detections:
                in_hand = any(boxes_overlap((bx1, by1, bx2, by2), hb)
                              for hb in hand_boxes)
                color = COL_OBJ_INHAND if in_hand else COL_OBJ_NORMAL
                tag   = f"👋 {label}" if in_hand else label
                if in_hand:
                    in_hand_count += 1
                    # extra fill to make it pop
                    alpha_fill(frame, bx1, by1, bx2, by2, COL_OBJ_INHAND, alpha=0.15)
                draw_rounded_rect(frame, bx1, by1, bx2, by2, color, 2)
                label_box(frame, f"{tag}  {conf:.0%}", bx1, by1 - 6,
                          bg=color, fg=(248, 255, 255), scale=0.50)

        # ════════════════════════════════════════════════════════════════════
        #  5.  HUD overlay
        # ════════════════════════════════════════════════════════════════════
        if show_hud:
            hud_lines = [
                (f"FPS: {fps:5.1f}",                          (0, 255, 100)),
                (f"Faces   : {face_count}",                   (255, 180, 0)),
                (f"Hands   : {len(hand_boxes)}",              (0, 255, 140)),
                (f"Objects : {len(yolo_detections)}",         (180, 220, 0)),
                (f"In Hand : {in_hand_count}",                (0, 120, 255)),
                ("",                                           (200, 200, 200)),
                ("[H] HUD  [O] Objects  [S] Save  [Q] Quit",  (160, 160, 160)),
            ]
            # Background panel
            pad, lh = 10, 22
            ph = len(hud_lines) * lh + pad * 2
            cv2.rectangle(frame, (0, 0), (320, ph), COL_HUD_BG, -1)
            cv2.rectangle(frame, (0, 0), (320, ph), (60, 60, 60), 1)

            for i, (text, color) in enumerate(hud_lines):
                if text:
                    cv2.putText(frame, text,
                                (pad + 4, pad + (i + 1) * lh),
                                FONT, 0.50, color, 1, cv2.LINE_AA)

            # Finger status per hand
            y_off = ph + 8
            for (side, f_ups) in hand_info_list:
                for j, (fname, up) in enumerate(zip(FINGER_NAMES, f_ups)):
                    dot = "●" if up else "○"
                    col = COL_FINGER_UP if up else (100, 100, 100)
                    cv2.putText(frame, f"{side[0]}.{fname[:3]}{dot}",
                                (10 + j * 60, y_off), FONT, 0.38, col, 1, cv2.LINE_AA)
                y_off += 18

        # ════════════════════════════════════════════════════════════════════
        #  6.  Show frame
        # ════════════════════════════════════════════════════════════════════
        cv2.imshow("Real-Time Object Identifier  |  Q = Quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            shot_count += 1
            path = os.path.join(SCREENSHOT_DIR, f"shot_{shot_count:04d}.jpg")
            cv2.imwrite(path, frame)
            print(f"[SCREENSHOT] Saved → {path}")
        elif key == ord('h'):
            show_hud = not show_hud
        elif key == ord('o'):
            show_objects = not show_objects

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()