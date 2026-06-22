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

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

NECKLACE_DIR = BASE_DIR / "static" / "necklaces"
EARRING_DIR = BASE_DIR / "static" / "earrings"

FACE_MODEL = str(ASSETS_DIR / "face_landmarker.task")
POSE_MODEL = str(ASSETS_DIR / os.environ.get("POSE_MODEL_FILE", "pose_landmark_lite.task"))

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


def find_necklace_outline_width(img):
    alpha = img[:, :, 3]
    mask = (alpha > 80).astype(np.uint8)
    h, w = mask.shape
    max_width = 0
    search_end = max(1, h // 3)
    for row in range(search_end):
        cols = np.where(mask[row, :] > 0)[0]
        if len(cols) > 2:
            row_width = cols[-1] - cols[0]
            if row_width > max_width:
                max_width = row_width
    if max_width < 5:
        return None
    return max_width


def find_necklace_opening_anchor(img):
    alpha = img[:, :, 3]
    mask = (alpha > 80).astype(np.uint8)
    h, w = mask.shape
    for row in range(h):
        cols = np.where(mask[row, :] > 0)[0]
        if len(cols) > 2:
            if cols[0] < w // 2 and cols[-1] >= w // 2:
                center_x = (cols[0] + cols[-1]) // 2
                return (center_x, row)
    return (w // 2, 0)


def find_silhouette_edge_at_y(seg_mask, ear_x_norm, ear_y_norm, side, h, w):
    if seg_mask is None:
        return None
    if hasattr(seg_mask, 'numpy_view'):
        seg_mask = seg_mask.numpy_view()
    seg_mask = np.array(seg_mask)
    mh, mw = seg_mask.shape[:2]
    binary = (seg_mask > 0.5).astype(np.uint8) if seg_mask.max() <= 1.0 else (seg_mask > 128).astype(np.uint8)

    ear_y_mask = int(ear_y_norm * mh)
    ear_x_mask = int(ear_x_norm * mw)
    search_half = max(3, mh // 40)
    y_start = max(0, ear_y_mask - search_half)
    y_end = min(mh, ear_y_mask + search_half)

    for row in range(y_start, y_end):
        cols = np.where(binary[row, :] > 0)[0]
        if len(cols) < 2:
            continue
        if side == "left":
            left_edge = cols[0]
            if left_edge < ear_x_mask:
                return int(left_edge * w / mw)
        else:
            right_edge = cols[-1]
            if right_edge > ear_x_mask:
                return int(right_edge * w / mw)

    return None


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


def find_shoulder_contour_from_segmentation(seg_mask, jaw_left, jaw_right, shoulder_cy, h, w):
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
    y_end = min(mh, shoulder_cy_mask + int(mh * 0.1))
    if y_end <= y_start + 5:
        return None
    search_x_min = max(0, min(jl_x, jr_x) - int(mw * 0.3))
    search_x_max = min(mw, max(jl_x, jr_x) + int(mw * 0.3))
    prev_left_x = None
    prev_right_x = None
    prev_row = None
    ANGLE_THRESHOLD = 0.25
    for row in range(y_start, y_end):
        cols = np.where(binary[row, search_x_min:search_x_max] > 0)[0]
        if len(cols) < 2:
            continue
        left_x = cols[0] + search_x_min
        right_x = cols[-1] + search_x_min
        if prev_left_x is not None and prev_row is not None:
            dy = row - prev_row
            left_dx = left_x - prev_left_x
            right_dx = right_x - prev_right_x
            left_angle = left_dx / dy
            right_angle = right_dx / dy
            if left_angle < -ANGLE_THRESHOLD and right_angle > ANGLE_THRESHOLD:
                center_x_mask = (left_x + right_x) // 2
                return {
                    'center_x': int(center_x_mask * w / mw),
                    'center_y': int(row * h / mh),
                    'left_x': int(left_x * w / mw),
                    'right_x': int(right_x * w / mw),
                }
        prev_left_x = left_x
        prev_right_x = right_x
        prev_row = row
    return None


# ============================================================
# FACE / NECK MASK FOR NECKLACE OCCLUSION
# ============================================================

JAW_LEFT_INDICES = [172, 149, 150, 136, 148, 152]
JAW_RIGHT_INDICES = [397, 379, 378, 365, 361, 323]
FACE_TOP_INDEX = 10

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]


def create_face_neck_mask(h, w, face_landmarks, neck_bottom_y):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not face_landmarks:
        return mask
    lm = face_landmarks

    pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))], dtype=np.int32)
    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2)

    left_x = int(hull_pts[:, 0].min())
    right_x = int(hull_pts[:, 0].max())
    jaw_bottom_y = int(hull_pts[:, 1].max())
    top_y = int(hull_pts[:, 1].min())
    ext_y = min(jaw_bottom_y + int((jaw_bottom_y - top_y) * 0.3), neck_bottom_y)

    extra = np.array([[right_x, ext_y], [left_x, ext_y]], dtype=np.int32)
    full = np.vstack([hull_pts, extra])
    full_hull = cv2.convexHull(full)

    cv2.fillConvexPoly(mask, full_hull, 255)
    mask = cv2.GaussianBlur(mask, (11, 11), 3)
    return mask


def create_face_only_mask(h, w, face_landmarks):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not face_landmarks:
        return mask
    lm = face_landmarks

    pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in range(len(lm))], dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    mask = cv2.GaussianBlur(mask, (7, 7), 2)
    return mask


def apply_face_neck_occlusion(resized, mask, x, y, feather=3):
    oh, ow = resized.shape[:2]
    fh, fw = mask.shape[:2]
    x1, y1 = max(int(x), 0), max(int(y), 0)
    x2, y2 = min(int(x) + ow, fw), min(int(y) + oh, fh)
    if x2 <= x1 or y2 <= y1:
        return resized
    ox1, oy1 = x1 - int(x), y1 - int(y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    face_region = mask[y1:y2, x1:x2].astype(np.float32) / 255.0
    if feather > 0:
        kernel = feather * 2 + 1
        face_region = cv2.GaussianBlur(face_region, (kernel, kernel), feather / 3)
    alpha = resized[oy1:oy2, ox1:ox2, 3:4].astype(np.float32) / 255.0
    alpha *= (1.0 - face_region[:, :, np.newaxis])
    resized[oy1:oy2, ox1:ox2, 3] = (alpha[:, :, 0] * 255).astype(np.uint8)
    return resized


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


def smooth_face_landmarks(current_lm, prev_lm, alpha=0.6):
    """Blend current and previous face landmarks for temporal smoothing."""
    if prev_lm is None:
        return current_lm
    smoothed = []
    for i in range(len(current_lm)):
        cx = current_lm[i].x
        cy = current_lm[i].y
        cz = current_lm[i].z
        px = prev_lm[i].x
        py = prev_lm[i].y
        pz = prev_lm[i].z
        # Create a new landmark-like object with blended values
        class BlendedLM:
            pass
        blm = BlendedLM()
        blm.x = alpha * px + (1 - alpha) * cx
        blm.y = alpha * py + (1 - alpha) * cy
        blm.z = alpha * pz + (1 - alpha) * cz
        smoothed.append(blm)
    return smoothed


# ============================================================
# TRY-ON SESSION (per WebSocket connection)
# ============================================================

INF_SCALE = 0.5
SKIP_FRAMES = 5
LERP_FACTOR = 0.8
WIDTH_PADDING = 1.25


class TryOnSession:
    def __init__(self, face_landmarker, pose_landmarker, necklaces, earrings):
        self.face_landmarker = face_landmarker
        self.pose_landmarker = pose_landmarker
        self.necklaces = necklaces
        self.earrings = earrings
        self.current_necklace_id = list(necklaces.keys())[0] if necklaces else None
        self.current_earring_id = None

        # Smoothing - necklace
        self.one_euro_ls_x = OneEuroFilter(min_cutoff=0.3, beta=1.0)
        self.one_euro_ls_y = OneEuroFilter(min_cutoff=0.3, beta=1.0)
        self.one_euro_rs_x = OneEuroFilter(min_cutoff=0.3, beta=1.0)
        self.one_euro_rs_y = OneEuroFilter(min_cutoff=0.3, beta=1.0)

        # Smoothing - earrings
        self.one_euro_left_ear_x = OneEuroFilter(min_cutoff=0.4, beta=0.8)
        self.one_euro_left_ear_y = OneEuroFilter(min_cutoff=0.4, beta=0.8)
        self.one_euro_right_ear_x = OneEuroFilter(min_cutoff=0.4, beta=0.8)
        self.one_euro_right_ear_y = OneEuroFilter(min_cutoff=0.4, beta=0.8)

        # Position state - necklace
        self.frame_counter = 0
        self.last_neck_cx = 0
        self.last_neck_cy = 0
        self.last_necklace_width = 0
        self.last_angle = 0.0
        self.target_neck_cx = 0
        self.target_neck_cy = 0
        self.target_necklace_width = 0
        self.target_angle = 0.0

        # Position state - earrings
        self.last_left_ear_x = 0
        self.last_left_ear_y = 0
        self.last_right_ear_x = 0
        self.last_right_ear_y = 0
        self.last_earring_width = 0
        self.last_right_scale = 1.0
        self.last_left_scale = 1.0
        self.target_left_ear_x = 0
        self.target_left_ear_y = 0
        self.target_right_ear_x = 0
        self.target_right_ear_y = 0
        self.target_earring_width = 0
        self.target_right_scale = 1.0
        self.target_left_scale = 1.0

        # Overlay cache
        self.last_overlay = None
        self.last_overlay_key = None
        self.face_clip_enabled = True
        self.earring_clip_enabled = True

        # Face mask temporal smoothing
        self.prev_face_landmarks = None
        self.face_mask_alpha = 0.6  # blend factor: 0=full new, 1=full old

        # Debug visualization
        self.debug_mask = False
        self.debug_silhouette = False

        # Last known good shoulder contour (for fallback when silhouette fails)
        self.last_good_shoulder_contour = None
        self.good_contour_frame_count = 0

        # Precompute necklace data
        self.necklace_data = {}
        for nid, img in self.necklaces.items():
            opening = find_necklace_outline_width(img)
            anchor = find_necklace_opening_anchor(img)
            self.necklace_data[nid] = {
                'image': img,
                'opening_width': opening,
                'opening_anchor': anchor,
            }

        # Precompute earring data
        self.earring_data = {}
        for eid, img in self.earrings.items():
            self.earring_data[eid] = {
                'image': img,
                'width': img.shape[1],
                'height': img.shape[0],
            }

    def select_necklace(self, nid):
        if nid is None or nid in self.necklaces:
            self.current_necklace_id = nid
            self.last_overlay = None
            self.last_overlay_key = None

    def select_earring(self, eid):
        if eid is None or eid in self.earrings:
            self.current_earring_id = eid

    def reset_calibration(self):
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

        # Get face landmarks
        face_result = self.face_landmarker.detect(mp_image)

        # Extract segmentation mask early for earring use
        seg_mask = getattr(pose_result, 'segmentation_masks', None)
        if seg_mask is not None:
            if isinstance(seg_mask, (list, tuple)):
                seg_mask = seg_mask[0]

        # Live silhouette-based detection
        shoulder_contour = None
        if face_result.face_landmarks:
            lm = face_result.face_landmarks[0]
            jaw_lower_left = lm[172]
            jaw_lower_right = lm[397]
            if seg_mask is not None:
                shoulder_contour = find_shoulder_contour_from_segmentation(
                    seg_mask,
                    (jaw_lower_left.x, jaw_lower_left.y),
                    (jaw_lower_right.x, jaw_lower_right.y),
                    shoulder_cy, h, w,
                )

        # Sanity check: reject contour if it's too far from pose shoulder center
        # or jumps too suddenly (likely caused by hand/arm in silhouette)
        CONTOUR_MAX_OFFSET = 0.4  # max fraction of frame height from shoulder_cy
        CONTOUR_MAX_JUMP = 0.3    # max fraction of frame height jump from last good
        if shoulder_contour is not None:
            cy = shoulder_contour['center_y']
            offset = abs(cy - shoulder_cy) / h
            too_far = offset > CONTOUR_MAX_OFFSET
            jumped = False
            if self.last_good_shoulder_contour is not None:
                jump = abs(cy - self.last_good_shoulder_contour['center_y']) / h
                jumped = jump > CONTOUR_MAX_JUMP
            if too_far or jumped:
                shoulder_contour = None

        # Update last known good contour
        if shoulder_contour is not None:
            self.last_good_shoulder_contour = shoulder_contour
            self.good_contour_frame_count = 0
        else:
            self.good_contour_frame_count += 1

        # Face/neck occlusion mask
        face_neck_mask = None
        face_only_mask = None
        if face_result.face_landmarks:
            face_lm = face_result.face_landmarks[0]

            best_contour_for_mask = shoulder_contour or self.last_good_shoulder_contour
            neck_bottom = best_contour_for_mask['center_y'] if best_contour_for_mask else shoulder_cy
            face_neck_mask = create_face_neck_mask(h, w, face_lm, neck_bottom)
            face_only_mask = create_face_only_mask(h, w, face_lm)

        # Necklace position
        status_msg = None
        self.frame_counter += 1

        # ========== NECKLACE RENDERING ==========
        if self.current_necklace_id and self.current_necklace_id in self.necklace_data:
            nd = self.necklace_data[self.current_necklace_id]
            necklace = nd['image']
            necklace_outline_width = nd['opening_width']

            if self.frame_counter % SKIP_FRAMES == 0:
                self.target_neck_cx = shoulder_cx
                # Use best available: current contour > last good contour > pose fallback
                best_contour = shoulder_contour or self.last_good_shoulder_contour
                if best_contour:
                    self.target_neck_cy = best_contour['center_y']
                    # Tilt-aware offset: shift Y along shoulder line direction
                    if 'left_x' in best_contour and 'right_x' in best_contour:
                        contour_span = best_contour['right_x'] - best_contour['left_x']
                        if contour_span > 0:
                            tilt_ratio = (best_contour['center_x'] - shoulder_cx) / max(contour_span, 1)
                            self.target_neck_cy += int(tilt_ratio * abs(ls_y - rs_y) * 0.3)
                else:
                    self.target_neck_cy = shoulder_cy
                if necklace_outline_width is not None:
                    target_width = shoulder_width * 0.5
                    scale = target_width / necklace_outline_width
                    self.target_necklace_width = int(necklace.shape[1] * scale)
                else:
                    self.target_necklace_width = int(shoulder_width * 0.5)
                self.target_angle = angle

            self.last_neck_cx = int(self.last_neck_cx + (self.target_neck_cx - self.last_neck_cx) * LERP_FACTOR)
            self.last_neck_cy = int(self.last_neck_cy + (self.target_neck_cy - self.last_neck_cy) * LERP_FACTOR)
            self.last_necklace_width = int(self.last_necklace_width + (self.target_necklace_width - self.last_necklace_width) * LERP_FACTOR)
            self.last_angle = self.last_angle + (self.target_angle - self.last_angle) * LERP_FACTOR

            necklace_width = self.last_necklace_width
            angle_out = self.last_angle
            scale = necklace_width / necklace.shape[1]
            necklace_height = int(necklace.shape[0] * scale)
            opening_anchor = nd['opening_anchor']
            anchor_x = opening_anchor[0] * scale
            anchor_y = opening_anchor[1] * scale

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

                cur_key = (self.current_necklace_id, necklace_width, necklace_height, round(angle_out, 2))
                if cur_key == self.last_overlay_key and self.last_overlay is not None:
                    resized = self.last_overlay
                else:
                    resized = cv2.resize(necklace, (necklace_width, necklace_height), interpolation=cv2.INTER_AREA)
                    resized = edge_glow(resized, glow_radius=3, intensity=0.12)
                    center = (anchor_x, anchor_y)
                    M = cv2.getRotationMatrix2D(center, angle_out, 1.0)
                    cos = abs(M[0, 0])
                    sin = abs(M[0, 1])
                    new_w = int(necklace_height * sin + necklace_width * cos)
                    new_h = int(necklace_height * cos + necklace_width * sin)
                    M[0, 2] += (new_w - resized.shape[1]) / 2
                    M[1, 2] += (new_h - resized.shape[0]) / 2
                    resized = cv2.warpAffine(resized, M, (new_w, new_h),
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                    face_crop = frame_bgr[
                        max(0, cy_clamp - shoulder_half):min(h, cy_clamp + shoulder_half),
                        max(0, lx_clamp):min(w, rx_clamp),
                    ]
                    resized = match_color_temperature(resized, face_crop)
                    resized = enhance_highlights(resized, lighting_shift, intensity=0.2)
                    resized = add_shimmer(resized, intensity=0.3)
                    resized = enhance_hdr(resized, contrast=1.6, saturation=1.05, brightness=0.75)
                    self.last_overlay = resized
                    self.last_overlay_key = cur_key

                anchor_canvas_x = anchor_x + (resized.shape[1] - necklace_width) / 2
                anchor_canvas_y = anchor_y + (resized.shape[0] - necklace_height) / 2
                x = self.last_neck_cx - anchor_canvas_x
                y = self.last_neck_cy - anchor_canvas_y

                if face_neck_mask is not None and self.face_clip_enabled:
                    resized = resized.copy()
                    apply_face_neck_occlusion(resized, face_neck_mask, int(x), int(y), feather=3)

                add_skin_shadow(frame_bgr, resized[:, :, 3], int(x), int(y), offset_x=4, offset_y=6, blur=21, opacity=0.35)
                overlay_realistic(frame_bgr, resized, int(x), int(y), lighting_shift=lighting_shift)

                # Draw debug landmarks
                cv2.circle(frame_bgr, (ls_x, ls_y), 4, (0, 255, 0), -1)
                cv2.circle(frame_bgr, (rs_x, rs_y), 4, (0, 255, 0), -1)
                cv2.circle(frame_bgr, (self.last_neck_cx, self.last_neck_cy), 4, (255, 255, 0), -1)

        # ========== EARRING RENDERING ==========
        if self.current_earring_id and self.current_earring_id in self.earring_data:
            ed = self.earring_data[self.current_earring_id]
            earring_img = ed['image']

            if face_result.face_landmarks:
                flm = face_result.face_landmarks[0]

                # MediaPipe face mesh ear landmarks
                # 234 = right tragus, 454 = left tragus
                # 58 = right jaw near ear, 288 = left jaw near ear
                RIGHT_EAR = 234
                LEFT_EAR = 454
                RIGHT_JAW = 58
                LEFT_JAW = 288
                NOSE_TIP = 1

                right_ear_lm = flm[RIGHT_EAR]
                left_ear_lm = flm[LEFT_EAR]
                right_jaw_lm = flm[RIGHT_JAW]
                left_jaw_lm = flm[LEFT_JAW]
                nose_lm = flm[NOSE_TIP]

                # Y from midpoint of tragus and jaw
                raw_right_y = (right_ear_lm.y * h + right_jaw_lm.y * h) / 2
                raw_left_y = (left_ear_lm.y * h + left_jaw_lm.y * h) / 2

                # X from midpoint of tragus and silhouette edge
                mp_right_x = right_ear_lm.x * w
                mp_left_x = left_ear_lm.x * w

                sil_right_x = find_silhouette_edge_at_y(seg_mask, right_ear_lm.x, right_ear_lm.y, "left", h, w)
                sil_left_x = find_silhouette_edge_at_y(seg_mask, left_ear_lm.x, left_ear_lm.y, "right", h, w)

                raw_right_x = mp_right_x * 2/3 + sil_right_x * 1/3 if sil_right_x is not None else mp_right_x
                raw_left_x = mp_left_x * 2/3 + sil_left_x * 1/3 if sil_left_x is not None else mp_left_x

                # Filter ear positions
                now_ear = time.time()
                right_ear_x = int(self.one_euro_right_ear_x(raw_right_x, now_ear))
                right_ear_y = int(self.one_euro_right_ear_y(raw_right_y, now_ear))
                left_ear_x = int(self.one_euro_left_ear_x(raw_left_x, now_ear))
                left_ear_y = int(self.one_euro_left_ear_y(raw_left_y, now_ear))

                # Face width from face oval landmarks for earring scale
                face_oval_pts = [(int(flm[i].x * w), int(flm[i].y * h)) for i in FACE_OVAL]
                face_left_x = min(p[0] for p in face_oval_pts)
                face_right_x = max(p[0] for p in face_oval_pts)
                face_width = face_right_x - face_left_x
                base_earring_width = max(16, int(face_width * 0.14))

                # Perspective scale: lower ear (closer to camera) is bigger
                nose_y = nose_lm.y * h
                ear_distance = np.sqrt((left_ear_x - right_ear_x)**2 + (left_ear_y - right_ear_y)**2)
                right_depth = (right_ear_y - nose_y) / max(ear_distance, 1)
                left_depth = (left_ear_y - nose_y) / max(ear_distance, 1)
                PERSPECTIVE_STRENGTH = 0.35
                right_scale = 1.0 + right_depth * PERSPECTIVE_STRENGTH
                left_scale = 1.0 + left_depth * PERSPECTIVE_STRENGTH
                right_scale = max(0.6, min(1.4, right_scale))
                left_scale = max(0.6, min(1.4, left_scale))

                right_ew = max(12, int(base_earring_width * right_scale))
                right_eh = int(earring_img.shape[0] * (right_ew / earring_img.shape[1]))
                left_ew = max(12, int(base_earring_width * left_scale))
                left_eh = int(earring_img.shape[0] * (left_ew / earring_img.shape[1]))

                self.target_right_ear_x = right_ear_x
                self.target_right_ear_y = right_ear_y
                self.target_left_ear_x = left_ear_x
                self.target_left_ear_y = left_ear_y
                self.target_earring_width = base_earring_width
                self.target_right_scale = right_scale
                self.target_left_scale = left_scale

                # Lerp
                self.last_right_ear_x = int(self.last_right_ear_x + (self.target_right_ear_x - self.last_right_ear_x) * LERP_FACTOR)
                self.last_right_ear_y = int(self.last_right_ear_y + (self.target_right_ear_y - self.last_right_ear_y) * LERP_FACTOR)
                self.last_left_ear_x = int(self.last_left_ear_x + (self.target_left_ear_x - self.last_left_ear_x) * LERP_FACTOR)
                self.last_left_ear_y = int(self.last_left_ear_y + (self.target_left_ear_y - self.last_left_ear_y) * LERP_FACTOR)
                self.last_earring_width = int(self.last_earring_width + (self.target_earring_width - self.last_earring_width) * LERP_FACTOR)
                self.last_right_scale = self.last_right_scale + (self.target_right_scale - self.last_right_scale) * LERP_FACTOR
                self.last_left_scale = self.last_left_scale + (self.target_left_scale - self.last_left_scale) * LERP_FACTOR

                # Get lighting for consistency
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
                lighting_shift_ear = (right_brightness - left_brightness) / 255.0

                # --- Render right earring ---
                right_ew = max(12, int(self.last_earring_width * self.last_right_scale))
                right_eh = int(earring_img.shape[0] * (right_ew / earring_img.shape[1])) if right_ew > 0 else 0

                if right_ew > 10 and right_eh > 10:
                    resized_ear = cv2.resize(earring_img, (right_ew, right_eh), interpolation=cv2.INTER_AREA)
                    resized_ear = edge_glow(resized_ear, glow_radius=2, intensity=0.1)

                    face_region_crop = frame_bgr[
                        max(0, self.last_right_ear_y - right_eh):min(h, self.last_right_ear_y + right_eh),
                        max(0, self.last_right_ear_x - right_ew):min(w, self.last_right_ear_x + right_ew),
                    ]
                    resized_ear = match_color_temperature(resized_ear, face_region_crop)
                    resized_ear = enhance_highlights(resized_ear, lighting_shift_ear, intensity=0.15)
                    resized_ear = add_shimmer(resized_ear, intensity=0.25)
                    resized_ear = enhance_hdr(resized_ear, contrast=1.4, saturation=1.05, brightness=0.8)

                    ear_x = self.last_right_ear_x - resized_ear.shape[1] // 2
                    ear_y = self.last_right_ear_y

                    # Face occlusion: hide earring parts that overlap the face
                    if face_only_mask is not None and self.earring_clip_enabled:
                        resized_ear = resized_ear.copy()
                        apply_face_neck_occlusion(resized_ear, face_only_mask, int(ear_x), int(ear_y), feather=2)

                    add_skin_shadow(frame_bgr, resized_ear[:, :, 3], int(ear_x), int(ear_y),
                                    offset_x=2, offset_y=3, blur=11, opacity=0.25)
                    overlay_realistic(frame_bgr, resized_ear, int(ear_x), int(ear_y),
                                      lighting_shift=lighting_shift_ear)

                # --- Render left earring (mirrored, different size) ---
                left_ew = max(12, int(self.last_earring_width * self.last_left_scale))
                left_eh = int(earring_img.shape[0] * (left_ew / earring_img.shape[1])) if left_ew > 0 else 0

                if left_ew > 10 and left_eh > 10:
                    resized_ear_l = cv2.resize(earring_img, (left_ew, left_eh), interpolation=cv2.INTER_AREA)
                    resized_ear_l = edge_glow(resized_ear_l, glow_radius=2, intensity=0.1)

                    resized_ear_l = cv2.flip(resized_ear_l, 1)

                    face_region_crop_l = frame_bgr[
                        max(0, self.last_left_ear_y - left_eh):min(h, self.last_left_ear_y + left_eh),
                        max(0, self.last_left_ear_x - left_ew):min(w, self.last_left_ear_x + left_ew),
                    ]
                    resized_ear_l = match_color_temperature(resized_ear_l, face_region_crop_l)
                    resized_ear_l = enhance_highlights(resized_ear_l, lighting_shift_ear, intensity=0.15)
                    resized_ear_l = add_shimmer(resized_ear_l, intensity=0.25)
                    resized_ear_l = enhance_hdr(resized_ear_l, contrast=1.4, saturation=1.05, brightness=0.8)

                    ear_x_left = self.last_left_ear_x - resized_ear_l.shape[1] // 2
                    ear_y_left = self.last_left_ear_y

                    # Face occlusion
                    if face_only_mask is not None and self.earring_clip_enabled:
                        resized_ear_l = resized_ear_l.copy()
                        apply_face_neck_occlusion(resized_ear_l, face_only_mask, int(ear_x_left), int(ear_y_left), feather=2)

                    add_skin_shadow(frame_bgr, resized_ear_l[:, :, 3], int(ear_x_left), int(ear_y_left),
                                    offset_x=-2, offset_y=3, blur=11, opacity=0.25)
                    overlay_realistic(frame_bgr, resized_ear_l, int(ear_x_left), int(ear_y_left),
                                      lighting_shift=lighting_shift_ear)

        # Debug: render face mask as transparent green overlay
        if self.debug_mask:
            if face_only_mask is not None:
                green_overlay = frame_bgr.copy()
                green_overlay[:, :] = [0, 255, 0]  # BGR green
                mask_float = face_only_mask.astype(np.float32) / 255.0
                mask_3ch = np.stack([mask_float, mask_float, mask_float], axis=-1)
                frame_bgr = (frame_bgr.astype(np.float32) * (1 - mask_3ch * 0.5) +
                             green_overlay.astype(np.float32) * mask_3ch * 0.5).astype(np.uint8)

            # Draw ear debug landmarks
            if face_result.face_landmarks:
                flm_dbg = face_result.face_landmarks[0]
                # MediaPipe tragus (234/454) - cyan circles
                tragus_r = (int(flm_dbg[234].x * w), int(flm_dbg[234].y * h))
                tragus_l = (int(flm_dbg[454].x * w), int(flm_dbg[454].y * h))
                cv2.circle(frame_bgr, tragus_r, 6, (255, 255, 0), 2)
                cv2.circle(frame_bgr, tragus_l, 6, (255, 255, 0), 2)
                cv2.putText(frame_bgr, "tragus", (tragus_r[0] + 10, tragus_r[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                cv2.putText(frame_bgr, "tragus", (tragus_l[0] - 50, tragus_l[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # Jaw landmarks (58/288) - orange circles
                jaw_r = (int(flm_dbg[58].x * w), int(flm_dbg[58].y * h))
                jaw_l = (int(flm_dbg[288].x * w), int(flm_dbg[288].y * h))
                cv2.circle(frame_bgr, jaw_r, 6, (0, 165, 255), 2)
                cv2.circle(frame_bgr, jaw_l, 6, (0, 165, 255), 2)
                cv2.putText(frame_bgr, "jaw", (jaw_r[0] + 10, jaw_r[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                cv2.putText(frame_bgr, "jaw", (jaw_l[0] - 30, jaw_l[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                # Silhouette edges - green
                sil_right = find_silhouette_edge_at_y(seg_mask, flm_dbg[234].x, flm_dbg[234].y, "left", h, w)
                sil_left = find_silhouette_edge_at_y(seg_mask, flm_dbg[454].x, flm_dbg[454].y, "right", h, w)
                ear_y_r = (tragus_r[1] + jaw_r[1]) // 2
                ear_y_l = (tragus_l[1] + jaw_l[1]) // 2
                if sil_right is not None:
                    cv2.circle(frame_bgr, (sil_right, ear_y_r), 6, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, "sil", (sil_right + 10, ear_y_r),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                if sil_left is not None:
                    cv2.circle(frame_bgr, (sil_left, ear_y_l), 6, (0, 255, 0), 2)
                    cv2.putText(frame_bgr, "sil", (sil_left - 30, ear_y_l),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Computed earlobe (X=mid tragus+silhouette, Y=mid tragus+jaw) - magenta
                cv2.circle(frame_bgr, (self.last_right_ear_x, self.last_right_ear_y), 6, (255, 0, 255), 2)
                cv2.putText(frame_bgr, "earlobe", (self.last_right_ear_x + 10, self.last_right_ear_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                cv2.circle(frame_bgr, (self.last_left_ear_x, self.last_left_ear_y), 6, (255, 0, 255), 2)
                cv2.putText(frame_bgr, "earlobe", (self.last_left_ear_x - 50, self.last_left_ear_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Debug: render full body silhouette
        if self.debug_silhouette and seg_mask is not None:
            if hasattr(seg_mask, 'numpy_view'):
                seg_vis = seg_mask.numpy_view()
            else:
                seg_vis = np.array(seg_mask)
            seg_vis = (seg_vis * 255).astype(np.uint8) if seg_vis.max() <= 1.0 else seg_vis.astype(np.uint8)
            seg_vis = cv2.resize(seg_vis, (w, h), interpolation=cv2.INTER_LINEAR)
            blue_overlay = frame_bgr.copy()
            blue_overlay[:, :] = [255, 0, 0]  # BGR blue
            seg_float = seg_vis.astype(np.float32) / 255.0
            seg_3ch = np.stack([seg_float, seg_float, seg_float], axis=-1)
            frame_bgr = (frame_bgr.astype(np.float32) * (1 - seg_3ch * 0.5) +
                         blue_overlay.astype(np.float32) * seg_3ch * 0.5).astype(np.uint8)

        return frame_bgr, status_msg


# ============================================================
# LOAD ASSETS
# ============================================================

print("Loading necklace images...")
necklaces = {}
NECKLACE_DIR.mkdir(parents=True, exist_ok=True)
for fpath in sorted(NECKLACE_DIR.glob("*.png")):
    nid = fpath.stem  # e.g. "gold_chain" from "gold_chain.png"
    img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"WARNING: Could not load {fpath}")
        continue
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    img = crop_transparent(img)
    necklaces[nid] = img
    print(f"  Loaded {fpath.name}: {img.shape}")

print("Loading earring images...")
earrings = {}
EARRING_DIR.mkdir(parents=True, exist_ok=True)
for fpath in sorted(EARRING_DIR.glob("*.png")):
    eid = fpath.stem
    img = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"WARNING: Could not load {fpath}")
        continue
    if img.ndim == 2:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    img = crop_transparent(img)
    earrings[eid] = img
    print(f"  Loaded {fpath.name}: {img.shape}")

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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.get("/necklaces")
async def list_necklaces():
    items = []
    for nid in sorted(necklaces.keys()):
        items.append({"id": nid, "name": nid.replace("_", " ").title(), "image": f"/static/necklaces/{nid}.png"})
    return {"necklaces": items}


@app.get("/earrings")
async def list_earrings():
    items = []
    for eid in sorted(earrings.keys()):
        items.append({"id": eid, "name": eid.replace("_", " ").title(), "image": f"/static/earrings/{eid}.png"})
    return {"earrings": items}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = TryOnSession(face_landmarker, pose_landmarker, necklaces, earrings)
    print(f"Client connected. Total necklaces: {len(necklaces)}, earrings: {len(earrings)}")

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

                await websocket.send_text(json.dumps(resp))

            elif msg.get("type") == "select_necklace":
                nid = msg.get("id")
                if nid is not None:
                    nid = str(nid)
                session.select_necklace(nid)
                print(f"Switched to necklace {nid}")

            elif msg.get("type") == "reset_calibration":
                session.reset_calibration()
                print("Calibration reset")

            elif msg.get("type") == "toggle_face_clip":
                session.face_clip_enabled = msg.get("enabled", True)
                session.earring_clip_enabled = msg.get("enabled", True)
                print(f"Face clip: {'on' if session.face_clip_enabled else 'off'}")

            elif msg.get("type") == "select_earring":
                eid = msg.get("id")
                if eid is not None:
                    eid = str(eid)
                session.select_earring(eid)
                print(f"Switched to earring {eid}")

            elif msg.get("type") == "toggle_earring_clip":
                session.earring_clip_enabled = msg.get("enabled", True)
                print(f"Earring clip: {'on' if session.earring_clip_enabled else 'off'}")

            elif msg.get("type") == "toggle_debug_mask":
                session.debug_mask = msg.get("enabled", False)
                print(f"Debug mask: {'on' if session.debug_mask else 'off'}")

            elif msg.get("type") == "toggle_debug_silhouette":
                session.debug_silhouette = msg.get("enabled", False)
                print(f"Debug silhouette: {'on' if session.debug_silhouette else 'off'}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
