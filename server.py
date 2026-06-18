import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR

NECKLACE_FILES = {
    1: "necklace.png",
    2: "necklace_2.png",
    3: "necklace_3.png",
}

FACE_MODEL = str(ASSETS_DIR / "face_landmarker.task")
POSE_MODEL = str(ASSETS_DIR / "pose_landmark_heavy.task")

RAM_LIMIT = 90

# ============================================================
# IMAGE HELPERS (ported from run.py)
# ============================================================

GLOW_COLOR = np.array([200, 230, 255], dtype=np.float32)


def edge_glow(img, glow_radius=3, intensity=0.15):
    alpha = img[:, :, 3]
    kernel_size = glow_radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    alpha_f = alpha.astype(np.float32)
    dilated = cv2.filter2D(alpha_f, -1, kernel)
    glow_mask = np.clip(dilated - alpha_f, 0, 255)
    glow_mask = (glow_mask / 255.0 * intensity * 255).astype(np.uint8)
    glow_f = glow_mask.astype(np.float32) / 255.0
    result = img[:, :, :3].astype(np.float32)
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] + glow_f * GLOW_COLOR[c], 0, 255)
    img[:, :, :3] = result.astype(np.uint8)
    return img


def match_color_temperature(overlay, face_crop):
    if face_crop.size == 0:
        return overlay
    face_bgr = face_crop[:, :, :3].astype(np.float32)
    overlay_bgr = overlay[:, :, :3].astype(np.float32)
    face_mean = np.mean(face_bgr, axis=(0, 1))
    overlay_mean = np.mean(
        overlay_bgr[overlay[:, :, 3] > 128], axis=0
    ) if np.any(overlay[:, :, 3] > 128) else np.array([128, 128, 128])
    shift = (face_mean - overlay_mean) * 0.15
    overlay[:, :, :3] = np.clip(
        overlay[:, :, :3].astype(np.float32) + shift, 0, 255
    ).astype(np.uint8)
    return overlay


def overlay_realistic(background, overlay, x, y, lighting_shift=0.0):
    h, w = overlay.shape[:2]
    if x >= background.shape[1] or y >= background.shape[0]:
        return
    if x + w <= 0 or y + h <= 0:
        return
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, background.shape[1]), min(y + h, background.shape[0])
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    overlay_crop = overlay[oy1:oy2, ox1:ox2].astype(np.float32)
    bg_crop = background[y1:y2, x1:x2].astype(np.float32)
    alpha = overlay_crop[:, :, 3:4] / 255.0
    blended = np.clip(overlay_crop[:, :, :3], 0, 255)
    result = alpha * blended + (1.0 - alpha) * bg_crop
    background[y1:y2, x1:x2] = result.astype(np.uint8)


def add_skin_shadow(frame, overlay_alpha, x, y,
                    offset_x=4, offset_y=6, blur=21, opacity=0.35):
    h, w = overlay_alpha.shape[:2]
    fh, fw = frame.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, fw), min(y + h, fh)
    if x2 <= x1 or y2 <= y1:
        return
    ox1, oy1 = x1 - x, y1 - y
    alpha_crop = overlay_alpha[oy1:oy1 + (y2 - y1), ox1:ox1 + (x2 - x1)].astype(np.float32)
    alpha_blurred = cv2.GaussianBlur(alpha_crop, (blur | 1, blur | 1), 0) / 255.0 * opacity
    sx1, sy1 = max(0, x1 + offset_x), max(0, y1 + offset_y)
    sx2, sy2 = min(fw, x2 + offset_x), min(fh, y2 + offset_y)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    a_crop_h, a_crop_w = sy2 - sy1, sx2 - sx1
    a_sx1, a_sy1 = sx1 - (x1 + offset_x), sy1 - (y1 + offset_y)
    a_shifted = alpha_blurred[a_sy1:a_sy1 + a_crop_h, a_sx1:a_sx1 + a_crop_w]
    bg_region = frame[sy1:sy2, sx1:sx2].astype(np.float32)
    shadow_color = np.array([20, 15, 10], dtype=np.float32)
    a_3ch = a_shifted[:, :, np.newaxis]
    blended = bg_region * (1.0 - a_3ch) + shadow_color * a_3ch
    frame[sy1:sy2, sx1:sx2] = np.clip(blended, 0, 255).astype(np.uint8)


def enhance_highlights(img, lighting_shift=0.0, intensity=0.25):
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.GaussianBlur(bright_mask, (5, 5), 0)
    highlight_shift_x = int(lighting_shift * 3)
    M_shift = np.float32([[1, 0, highlight_shift_x], [0, 1, 0]])
    bright_mask = cv2.warpAffine(bright_mask, M_shift, (bright_mask.shape[1], bright_mask.shape[0]))
    highlight_strength = (bright_mask.astype(np.float32) / 255.0 * intensity)
    highlight_color = np.zeros_like(img[:, :, :3], dtype=np.float32)
    highlight_color[:] = [255, 245, 230]
    result = img[:, :, :3].astype(np.float32)
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] + highlight_strength * highlight_color[:, :, c], 0, 255)
    img[:, :, :3] = result.astype(np.uint8)
    return img


shimmer_cache = {}


def add_shimmer(img, intensity=0.2):
    h, w = img.shape[:2]
    key = (h, w)
    if key not in shimmer_cache:
        shimmer = np.zeros((h, w), dtype=np.float32)
        np.random.seed(42)
        num_spots = max(10, (h * w) // 1200)
        for _ in range(num_spots):
            sx = np.random.randint(0, w)
            sy = np.random.randint(0, h)
            radius = np.random.randint(6, max(7, min(w, h) // 5))
            intensity_val = np.random.uniform(0.5, 1.0)
            y_start, y_end = max(0, sy - radius), min(h, sy + radius + 1)
            x_start, x_end = max(0, sx - radius), min(w, sx + radius + 1)
            y_grid, x_grid = np.ogrid[y_start:y_end, x_start:x_end]
            dist = np.sqrt((x_grid - sx) ** 2 + (y_grid - sy) ** 2)
            falloff = np.maximum(0, 1.0 - dist / radius) ** 2.0
            shimmer[y_start:y_end, x_start:x_end] = np.maximum(
                shimmer[y_start:y_end, x_start:x_end], falloff * intensity_val
            )
        shimmer_cache[key] = cv2.merge([shimmer, shimmer, shimmer]) * 80 * intensity
    shimmer_3ch = shimmer_cache[key]
    alpha = img[:, :, 3:4]
    mask = (alpha > 128).astype(np.float32)
    result = img[:, :, :3].astype(np.float32)
    result = np.clip(result + shimmer_3ch * mask, 0, 255)
    img[:, :, :3] = result.astype(np.uint8)
    return img


def enhance_hdr(img, contrast=1.6, saturation=1.05, brightness=0.75):
    rgb = img[:, :, :3].astype(np.float32) / 255.0
    mid = 0.5
    rgb = (rgb - mid) * contrast + mid
    rgb = np.clip(rgb * brightness, 0, 1)
    gray = 0.114 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.299 * rgb[:, :, 2]
    gray = gray[:, :, np.newaxis]
    rgb = gray + (rgb - gray) * saturation
    rgb = np.clip(rgb, 0, 1)
    img[:, :, :3] = (rgb * 255).astype(np.uint8)
    return img


def crop_transparent(img):
    alpha = img[:, :, 3]
    mask = (alpha > 80).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return img
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(labels == largest_label)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def find_necklace_opening(img):
    alpha = img[:, :, 3]
    mask = (alpha > 80).astype(np.uint8)
    h, w = mask.shape
    min_width = w
    min_row = -1
    for row in range(h):
        cols = np.where(mask[row, :] > 0)[0]
        if len(cols) > 2:
            row_width = cols[-1] - cols[0]
            if row_width < min_width:
                min_width = row_width
                min_row = row
    if min_row < 0 or min_width >= w * 0.95 or min_width < 5:
        return None
    return min_width


def find_neck_edges_from_segmentation(seg_mask, jaw_left, jaw_right, shoulder_cy, h, w):
    if seg_mask is None:
        return None
    if hasattr(seg_mask, 'numpy_view'):
        seg_mask = seg_mask.numpy_view()
    seg_mask = np.array(seg_mask)
    mh, mw = seg_mask.shape[:2]
    binary = (seg_mask > 0.5).astype(np.uint8) if seg_mask.max() <= 1.0 else (seg_mask > 128).astype(np.uint8)
    jl_x, jl_y = int(jaw_left[0] * mw), int(jaw_left[1] * mh)
    jr_x, jr_y = int(jaw_right[0] * mw), int(jaw_right[1] * mh)
    shoulder_cy_mask = int(shoulder_cy * mh / h)
    y_start = max(0, min(jl_y, jr_y))
    y_end = min(mh, shoulder_cy_mask)
    if y_end <= y_start + 3:
        return None
    search_x_min = max(0, min(jl_x, jr_x) - int(mw * 0.25))
    search_x_max = min(mw, max(jl_x, jr_x) + int(mw * 0.25))
    region = binary[y_start:y_end, search_x_min:search_x_max]
    if region.sum() < 5:
        return None
    min_width_val, min_row, min_left, min_right = 999999, -1, 0, 0
    for row in range(region.shape[0]):
        cols = np.where(region[row, :] > 0)[0]
        if len(cols) > 2:
            row_width = cols[-1] - cols[0]
            if row_width < min_width_val:
                min_width_val = row_width
                min_row = row
                min_left = cols[0] + search_x_min
                min_right = cols[-1] + search_x_min
    if min_row < 0:
        return None
    neck_width_mask = min_right - min_left
    neck_center_x_mask = (min_left + min_right) // 2
    neck_center_y_mask = y_start + min_row
    neck_center_x = int(neck_center_x_mask * w / mw)
    neck_center_y = int(neck_center_y_mask * h / mh)
    neck_width = int(neck_width_mask * w / mw)
    if neck_width < 10:
        return None
    return {
        'neck_center_x': neck_center_x,
        'neck_center_y': neck_center_y,
        'neck_width': neck_width,
        'neck_left': int(min_left * w / mw),
        'neck_right': int(min_right * w / mw),
    }


# ============================================================
# ONE EURO FILTER
# ============================================================

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        alpha_d = self._smoothing_factor(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def _smoothing_factor(self, dt, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


# ============================================================
# TRY-ON SESSION (per WebSocket connection)
# ============================================================

INF_SCALE = 0.5
SKIP_FRAMES = 5
LERP_FACTOR = 0.8
WIDTH_PADDING = 1.35
Y_LOWER = 35
CALIBRATION_FRAMES = 15


class TryOnSession:
    def __init__(self, face_landmarker, pose_landmarker, necklaces):
        self.face_landmarker = face_landmarker
        self.pose_landmarker = pose_landmarker
        self.necklaces = necklaces
        self.current_necklace_id = 1

        # Calibration
        self.calibrated = False
        self.cal_y_offset = 0.0
        self.cal_width_ratio = 0.65
        self.cal_neck_width = 0
        self.calibration_attempts = 0

        # Smoothing
        self.one_euro_ls_x = OneEuroFilter(min_cutoff=0.3, beta=1.0)
        self.one_euro_ls_y = OneEuroFilter(min_cutoff=0.3, beta=1.0)
        self.one_euro_rs_x = OneEuroFilter(min_cutoff=0.3, beta=1.0)
        self.one_euro_rs_y = OneEuroFilter(min_cutoff=0.3, beta=1.0)

        # Position state
        self.frame_counter = 0
        self.last_neck_cx = 0
        self.last_neck_cy = 0
        self.last_necklace_width = 0
        self.last_angle = 0.0
        self.target_neck_cx = 0
        self.target_neck_cy = 0
        self.target_necklace_width = 0
        self.target_angle = 0.0

        # Overlay cache
        self.last_overlay = None
        self.last_overlay_key = None

        # Precompute necklace data
        self.necklace_data = {}
        for nid, img in self.necklaces.items():
            opening = find_necklace_opening(img)
            self.necklace_data[nid] = {
                'image': img,
                'opening_width': opening,
            }

    def select_necklace(self, nid):
        if nid in self.necklaces:
            self.current_necklace_id = nid
            self.last_overlay = None
            self.last_overlay_key = None

    def reset_calibration(self):
        self.calibrated = False
        self.cal_y_offset = 0.0
        self.cal_width_ratio = 0.65
        self.cal_neck_width = 0
        self.calibration_attempts = 0
        self.one_euro_ls_x.reset()
        self.one_euro_ls_y.reset()
        self.one_euro_rs_x.reset()
        self.one_euro_rs_y.reset()

    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, None, fx=INF_SCALE, fy=INF_SCALE, interpolation=cv2.INTER_LINEAR)
        sh, sw = small.shape[:2]
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self.pose_landmarker.detect(mp_image)

        if not pose_result.pose_landmarks:
            return frame_bgr, "Show upper body"

        plm = pose_result.pose_landmarks[0]
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        ls, rs = plm[LEFT_SHOULDER], plm[RIGHT_SHOULDER]

        now = time.time()
        ls_x = int(self.one_euro_ls_x(ls.x * w, now))
        ls_y = int(self.one_euro_ls_y(ls.y * h, now))
        rs_x = int(self.one_euro_rs_x(rs.x * w, now))
        rs_y = int(self.one_euro_rs_y(rs.y * h, now))

        raw_cx = (ls_x + rs_x) / 2.0
        raw_cy = (ls_y + rs_y) / 2.0
        shoulder_cx = int(raw_cx)
        shoulder_cy = int(raw_cy)
        shoulder_width = abs(rs_x - ls_x)

        angle = -np.degrees(np.arctan2(ls_y - rs_y, ls_x - rs_x))

        # Calibration
        status_msg = None
        if not self.calibrated and self.calibration_attempts < CALIBRATION_FRAMES:
            face_result = self.face_landmarker.detect(mp_image)
            if face_result.face_landmarks:
                lm = face_result.face_landmarks[0]
                jaw_lower_left = lm[172]
                jaw_lower_right = lm[397]
                seg_mask = getattr(pose_result, 'segmentation_masks', None)
                if seg_mask is not None:
                    if isinstance(seg_mask, (list, tuple)):
                        seg_mask = seg_mask[0]
                    neck_from_seg = find_neck_edges_from_segmentation(
                        seg_mask,
                        (jaw_lower_left.x, jaw_lower_left.y),
                        (jaw_lower_right.x, jaw_lower_right.y),
                        shoulder_cy, h, w,
                    )
                    if neck_from_seg:
                        neck_cy_from_seg = neck_from_seg['neck_center_y']
                        neck_w_from_seg = neck_from_seg['neck_width']
                        self.cal_y_offset = (neck_cy_from_seg - shoulder_cy) / max(shoulder_width, 1)
                        self.cal_width_ratio = neck_w_from_seg / max(shoulder_width, 1)
                        self.cal_neck_width = neck_w_from_seg
                        self.calibration_attempts += 1
                        if self.calibration_attempts >= CALIBRATION_FRAMES:
                            self.calibrated = True
                status_msg = f"CALIBRATING... {self.calibration_attempts}/{CALIBRATION_FRAMES}"
            else:
                status_msg = "Show face for calibration"

        # Necklace position
        self.frame_counter += 1
        nd = self.necklace_data[self.current_necklace_id]
        necklace = nd['image']
        necklace_opening_width = nd['opening_width']

        if self.frame_counter % SKIP_FRAMES == 0:
            self.target_neck_cx = shoulder_cx
            self.target_neck_cy = shoulder_cy + int(shoulder_width * self.cal_y_offset) + Y_LOWER
            if necklace_opening_width is not None and self.calibrated:
                open_ratio = necklace_opening_width / necklace.shape[1]
                if open_ratio > 0.15:
                    self.target_necklace_width = int(self.cal_neck_width / open_ratio)
                    self.target_necklace_width = min(self.target_necklace_width, int(shoulder_width * 1.5))
                else:
                    self.target_necklace_width = int(shoulder_width * self.cal_width_ratio * WIDTH_PADDING)
            else:
                self.target_necklace_width = int(shoulder_width * self.cal_width_ratio * WIDTH_PADDING)
            self.target_angle = angle

        self.last_neck_cx = int(self.last_neck_cx + (self.target_neck_cx - self.last_neck_cx) * LERP_FACTOR)
        self.last_neck_cy = int(self.last_neck_cy + (self.target_neck_cy - self.last_neck_cy) * LERP_FACTOR)
        self.last_necklace_width = int(self.last_necklace_width + (self.target_necklace_width - self.last_necklace_width) * LERP_FACTOR)
        self.last_angle = self.last_angle + (self.target_angle - self.last_angle) * LERP_FACTOR

        y = self.last_neck_cy
        necklace_width = self.last_necklace_width
        angle_out = self.last_angle
        scale = necklace_width / necklace.shape[1]
        necklace_height = int(necklace.shape[0] * scale)

        if necklace_width > 20 and necklace_height > 20:
            shoulder_half = shoulder_width // 4
            lx_clamp = max(0, min(ls_x, w - 1))
            rx_clamp = max(0, min(rs_x, w - 1))
            cy_clamp = max(0, min(shoulder_cy, h - 1))

            left_region = frame_bgr[
                max(0, cy_clamp - shoulder_half):min(h, cy_clamp + shoulder_half),
                max(0, lx_clamp - shoulder_half):min(w, lx_clamp + shoulder_half),
            ]
            right_region = frame_bgr[
                max(0, cy_clamp - shoulder_half):min(h, cy_clamp + shoulder_half),
                max(0, rx_clamp - shoulder_half):min(w, rx_clamp + shoulder_half),
            ]
            left_brightness = np.mean(left_region) if left_region.size > 0 else 128
            right_brightness = np.mean(right_region) if right_region.size > 0 else 128
            lighting_shift = (right_brightness - left_brightness) / 255.0

            cur_key = (necklace_width, necklace_height, round(angle_out, 2))
            if cur_key == self.last_overlay_key and self.last_overlay is not None:
                resized = self.last_overlay
            else:
                resized = cv2.resize(necklace, (necklace_width, necklace_height), interpolation=cv2.INTER_AREA)
                resized = edge_glow(resized, glow_radius=3, intensity=0.12)
                center = (resized.shape[1] // 2, resized.shape[0] // 2)
                M = cv2.getRotationMatrix2D(center, angle_out, 1.0)
                resized = cv2.warpAffine(resized, M, (resized.shape[1], resized.shape[0]),
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                face_crop = frame_bgr[
                    max(0, cy_clamp - shoulder_half):min(h, cy_clamp + shoulder_half),
                    max(0, lx_clamp):min(w, rx_clamp),
                ]
                resized = match_color_temperature(resized, face_crop)
                resized = enhance_highlights(resized, lighting_shift, intensity=0.2)
                resized = add_shimmer(resized, intensity=0.2)
                resized = enhance_hdr(resized, contrast=1.6, saturation=1.05, brightness=0.75)
                self.last_overlay = resized
                self.last_overlay_key = cur_key

            x = self.last_neck_cx - resized.shape[1] // 2
            add_skin_shadow(frame_bgr, resized[:, :, 3], x, y, offset_x=4, offset_y=6, blur=21, opacity=0.35)
            overlay_realistic(frame_bgr, resized, x, y, lighting_shift=lighting_shift)

            # Draw debug landmarks
            cv2.circle(frame_bgr, (ls_x, ls_y), 4, (0, 255, 0), -1)
            cv2.circle(frame_bgr, (rs_x, rs_y), 4, (0, 255, 0), -1)
            cv2.circle(frame_bgr, (self.last_neck_cx, self.last_neck_cy), 4, (255, 255, 0), -1)

        return frame_bgr, status_msg


# ============================================================
# LOAD ASSETS
# ============================================================

print("Loading necklace images...")
necklaces = {}
for nid, fname in NECKLACE_FILES.items():
    fpath = str(ASSETS_DIR / fname)
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"WARNING: Could not load {fpath}")
        continue
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    img = crop_transparent(img)
    necklaces[nid] = img
    print(f"  Loaded {fname}: {img.shape}")

print("Loading MediaPipe models...")
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    num_faces=1,
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    num_poses=1,
    output_segmentation_masks=True,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
print("Models loaded.")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Jewelry Virtual Try-On")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.get("/necklaces")
async def list_necklaces():
    items = []
    for nid in sorted(necklaces.keys()):
        items.append({"id": nid, "name": f"Necklace {nid}"})
    return {"necklaces": items}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = TryOnSession(face_landmarker, pose_landmarker, necklaces)
    print(f"Client connected. Total necklaces: {len(necklaces)}")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                # Decode JPEG base64
                img_bytes = base64.b64decode(msg["data"])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Process
                result_frame, status = session.process_frame(frame)

                # Encode result as JPEG base64
                _, buf = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64 = base64.b64encode(buf).decode("utf-8")

                resp = {"type": "frame", "data": b64}
                if status:
                    resp["status"] = status
                resp["calibrated"] = session.calibrated

                await websocket.send_text(json.dumps(resp))

            elif msg.get("type") == "select_necklace":
                nid = msg.get("id", 1)
                session.select_necklace(nid)
                print(f"Switched to necklace {nid}")

            elif msg.get("type") == "reset_calibration":
                session.reset_calibration()
                print("Calibration reset")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
