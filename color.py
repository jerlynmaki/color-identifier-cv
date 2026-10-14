#!/usr/bin/env python3
"""
Real-Time Object Color Identifier for Colorblind Individuals
CSC 126 - Computer Vision Midterm Project

What it does
------------
Reads a laptop webcam feed, samples a rectangular "Sampling Zone" in the centre
of the frame, and tells the user (in plain colour names) which colours are
present inside that zone, with the exact percentage of each colour plus its
RGB and HEX value.

Pipeline (maps to the midterm rubric)
-------------------------------------
(a) IMAGE OPERATIONS
    - Optional gray-world white balance (lighting compensation)
    - Median blur + Gaussian blur on the ROI (noise reduction / colour smoothing)
    - BGR -> HSV   (glare / validity masks)
    - BGR -> CIE-LAB (perceptual colour space used for colour distance)
(b) IMAGE SEGMENTATION
    - Fixed centre Region Of Interest (the Sampling Zone)
    - Threshold masks: glare mask (HSV), and optional background-subtraction
      mask (absdiff + threshold + morphological open/close) that isolates the
      object from the empty background
(d) IMAGE ANALYSIS AND TRANSFORMATION
    - K-Means clustering of ROI pixels in LAB space (multiple colours)
    - Euclidean distance in LAB (Delta-E 76) from each cluster centre to a
      named-colour lookup table (simple names such as "Red", "Blue", "Green")
    - Merge clusters that map to the same name -> percentage distribution
    - Live overlay: zone rectangle, text labels, RGB/HEX, side analytics panel

Keys
----
  q / ESC  quit            s  save screenshot + log to CSV
  SPACE    freeze/unfreeze b  capture background (object isolation ON)
  c        clear background w  toggle white balance (gray-world)
  g        toggle glare ignore   k  cycle K-Means clusters (2..6)
  [ / ]    shrink / grow zone    h  toggle help line
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 1. NAMED COLOUR LOOKUP TABLE  (Member 1 owns this - tune freely)
#    Each entry: (simple name, (R, G, B)).
#    Every simple name has several shades: light, normal, dark and "muted"
#    (the washed-out look a webcam gives). The nearest shade wins, and all
#    shades of one name report as that single simple name.
#    Dark / shadowed yellows (olive-looking under webcam light) are filed
#    under "Yellow" so a yellow object in shade is not reported as Green.
# --------------------------------------------------------------------------
PALETTE = [
    # ---- Neutrals ----
    ("Black", (10, 10, 10)), ("Black", (35, 35, 35)), ("Black", (50, 50, 50)),
    ("Gray", (70, 70, 70)), ("Gray", (100, 100, 100)), ("Gray", (128, 128, 128)),
    ("Gray", (150, 150, 150)), ("Gray", (170, 170, 170)), ("Gray", (190, 190, 190)),
    ("White", (215, 215, 215)), ("White", (230, 230, 230)),
    ("White", (245, 245, 245)), ("White", (255, 255, 255)),

    # ---- Red (bright, crimson, dark, muted) ----
    ("Red", (230, 30, 30)), ("Red", (200, 20, 50)), ("Red", (180, 20, 30)),
    ("Red", (150, 40, 40)), ("Red", (120, 15, 25)), ("Red", (90, 20, 30)),
    ("Red", (190, 70, 70)), ("Red", (160, 60, 60)),

    # ---- Orange (light, normal, dark, muted) ----
    ("Orange", (255, 140, 0)), ("Orange", (230, 110, 20)), ("Orange", (240, 160, 60)),
    ("Orange", (200, 100, 30)), ("Orange", (220, 130, 60)),

    # ---- Yellow (pale, bright, gold, shadowed/olive-looking) ----
    ("Yellow", (255, 245, 120)), ("Yellow", (255, 230, 0)), ("Yellow", (240, 210, 60)),
    ("Yellow", (230, 200, 50)), ("Yellow", (212, 175, 55)), ("Yellow", (200, 180, 70)),
    ("Yellow", (190, 175, 60)), ("Yellow", (150, 140, 50)), ("Yellow", (126, 122, 60)),

    # ---- Brown (light, normal, dark, muted) ----
    ("Brown", (170, 120, 80)), ("Brown", (150, 100, 60)), ("Brown", (120, 70, 30)),
    ("Brown", (100, 60, 40)), ("Brown", (90, 55, 45)), ("Brown", (70, 40, 20)),

    # ---- Beige (light, normal, skin-tone range) ----
    ("Beige", (245, 235, 210)), ("Beige", (235, 220, 190)), ("Beige", (225, 190, 160)),
    ("Beige", (210, 180, 140)), ("Beige", (200, 165, 135)),

    # ---- Pink (pale, normal, hot, muted) ----
    ("Pink", (255, 190, 210)), ("Pink", (255, 150, 185)), ("Pink", (235, 60, 140)),
    ("Pink", (220, 120, 150)), ("Pink", (200, 90, 130)),

    # ---- Green (light, bright, dark, muted) ----
    ("Green", (160, 230, 190)), ("Green", (100, 180, 100)), ("Green", (140, 220, 40)),
    ("Green", (30, 160, 60)), ("Green", (90, 140, 80)), ("Green", (25, 90, 40)),
    ("Green", (50, 90, 60)),

    # ---- Cyan (light, bright, teal, dark, muted) ----
    ("Cyan", (64, 210, 200)), ("Cyan", (0, 200, 230)), ("Cyan", (80, 170, 180)),
    ("Cyan", (0, 128, 128)), ("Cyan", (0, 90, 100)),

    # ---- Blue (light, normal, navy, muted) ----
    ("Blue", (135, 195, 235)), ("Blue", (90, 130, 200)), ("Blue", (30, 80, 230)),
    ("Blue", (60, 100, 180)), ("Blue", (30, 50, 130)), ("Blue", (15, 25, 100)),
    ("Blue", (20, 40, 70)),

    # ---- Purple (light, normal, magenta, dark, muted) ----
    ("Purple", (180, 140, 200)), ("Purple", (150, 110, 220)), ("Purple", (200, 30, 200)),
    ("Purple", (128, 40, 160)), ("Purple", (100, 70, 140)), ("Purple", (75, 0, 130)),
    ("Purple", (60, 30, 90)),
]

PALETTE_NAMES = [n for n, _ in PALETTE]


def _rgb_list_to_lab(rgb_list):
    """Convert Nx3 RGB (0-255) to float LAB (L 0-100, a/b about -128..127)."""
    rgb = np.array(rgb_list, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)


PALETTE_LAB = _rgb_list_to_lab([c for _, c in PALETTE])


def nearest_color_name(lab_vec):
    """Euclidean distance (Delta-E 76) in LAB to every palette colour."""
    dists = np.linalg.norm(PALETTE_LAB - lab_vec.reshape(1, 3), axis=1)
    i = int(np.argmin(dists))
    return PALETTE_NAMES[i], float(dists[i])


# --------------------------------------------------------------------------
# 2. SETTINGS
# --------------------------------------------------------------------------
WINDOW = "Color Identifier - CSC 126"
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 340            # width of the analytics side panel (pixels)
ANALYSIS_SIZE = 64       # ROI is downsampled so its longest side is this many px
MIN_PCT = 8.0            # colours below this share are hidden (edge-blur artefacts)
BG_DIFF_THRESH = 28      # background-subtraction threshold (0-255)
MIN_FG_RATIO = 0.03      # less than 3 % foreground -> "no object in zone"
SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")


# --------------------------------------------------------------------------
# 3. SMALL HELPERS
# --------------------------------------------------------------------------
def bgr_to_hex(bgr):
    b, g, r = [int(round(float(v))) for v in bgr]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def bgr_to_rgb_tuple(bgr):
    b, g, r = [int(round(float(v))) for v in bgr]
    return r, g, b


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1, outline=True):
    """Text on a translucent dark backing so it stays readable on any background."""
    if outline:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        shaded_rect(img, (org[0] - 4, org[1] - th - 4), (org[0] + tw + 4, org[1] + base + 3),
                    (0, 0, 0), 0.55)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def shaded_rect(img, p1, p2, color=(0, 0, 0), alpha=0.55):
    """Semi-transparent filled rectangle (clipped to the image)."""
    h, w = img.shape[:2]
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(w, p2[0]), min(h, p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, roi)


def fit_into(img, w, h):
    """Resize keeping aspect ratio and centre on a (w, h) dark canvas."""
    canvas = np.full((h, w, 3), (45, 45, 45), np.uint8)
    if img is None or img.size == 0:
        return canvas
    ih, iw = img.shape[:2]
    s = min(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = r
    return canvas


def gray_world(frame):
    """Gray-world white balance: scale channels so their means are equal."""
    sample = frame[::8, ::8].reshape(-1, 3).astype(np.float32)
    means = np.maximum(sample.mean(axis=0), 1e-3)
    gain = means.mean() / means
    out = frame.astype(np.float32) * gain.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def zone_rect(w, h, scale):
    zw, zh = int(w * scale), int(h * scale)
    x1, y1 = (w - zw) // 2, (h - zh) // 2
    return x1, y1, x1 + zw, y1 + zh


# --------------------------------------------------------------------------
# 4. CORE ANALYSIS  (Member 1 owns this section)
# --------------------------------------------------------------------------
def analyze_zone(roi_bgr, k=4, bg_roi=None, ignore_glare=False):
    """
    Analyse the pixels of the sampling zone.

    Returns a dict:
        items    : list of {name, pct, bgr, dE} sorted by pct (descending)
        small    : smoothed, downsampled ROI (for the thumbnail)
        valid    : binary mask of pixels that were analysed
        status   : "ok" | "no_object" | "empty"
    """
    out = {"items": [], "small": None, "valid": None, "status": "ok"}
    if roi_bgr is None or roi_bgr.size == 0:
        out["status"] = "empty"
        return out

    # --- (a) downsample + noise reduction ---------------------------------
    h, w = roi_bgr.shape[:2]
    scale = ANALYSIS_SIZE / float(max(h, w))
    if scale < 1.0:
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        small = cv2.resize(roi_bgr, size, interpolation=cv2.INTER_AREA)
    else:
        small = roi_bgr.copy()
    small = cv2.medianBlur(small, 3)             # salt-and-pepper noise
    small = cv2.GaussianBlur(small, (5, 5), 0)   # smooth gradients / sensor noise
    out["small"] = small

    # --- (b) segmentation masks -------------------------------------------
    valid = np.full(small.shape[:2], 255, np.uint8)

    if ignore_glare:  # HSV threshold: very bright + unsaturated = specular glare
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        glare = (hsv[..., 2] >= 250) & (hsv[..., 1] <= 25)
        valid[glare] = 0

    if bg_roi is not None:  # background subtraction -> isolates the object
        bg_small = cv2.resize(bg_roi, (small.shape[1], small.shape[0]),
                              interpolation=cv2.INTER_AREA)
        bg_small = cv2.GaussianBlur(cv2.medianBlur(bg_small, 3), (5, 5), 0)
        diff = cv2.absdiff(small, bg_small).max(axis=2)
        _, fg = cv2.threshold(diff, BG_DIFF_THRESH, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)    # remove speckles
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)   # fill small holes
        valid = cv2.bitwise_and(valid, fg)

    out["valid"] = valid
    sel = valid > 0
    if sel.sum() < max(20, MIN_FG_RATIO * valid.size):
        out["status"] = "no_object"
        return out

    # --- (a) BGR -> LAB (float) -------------------------------------------
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    pix_lab = lab[sel].astype(np.float32)
    pix_bgr = small[sel].astype(np.float32)

    # --- (d) K-Means clustering in LAB ------------------------------------
    kk = max(1, min(int(k), len(pix_lab)))
    cv2.setRNGSeed(42)  # deterministic -> stable results between runs
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(pix_lab, kk, None, criteria, 2,
                                    cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    total = float(len(pix_lab))

    # --- (d) name each cluster (Euclidean distance) and merge duplicates --
    merged = {}
    for ci in range(kk):
        members = labels == ci
        cnt = int(members.sum())
        if cnt == 0:
            continue
        name, dE = nearest_color_name(centers[ci])
        mean_bgr = pix_bgr[members].mean(axis=0)
        e = merged.setdefault(name, {"count": 0, "bgr_sum": np.zeros(3), "dE_sum": 0.0})
        e["count"] += cnt
        e["bgr_sum"] += mean_bgr * cnt
        e["dE_sum"] += dE * cnt

    items = []
    for name, e in merged.items():
        items.append({
            "name": name,
            "pct": 100.0 * e["count"] / total,
            "bgr": e["bgr_sum"] / e["count"],
            "dE": e["dE_sum"] / e["count"],
        })
    items.sort(key=lambda d: d["pct"], reverse=True)
    out["items"] = items
    return out


class ColorSmoother:
    """
    Exponential moving average over colour percentages so the labels do not
    flicker from frame to frame (live demos look much more professional).
    """

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.state = {}

    def reset(self):
        self.state = {}

    def update(self, items):
        a = self.alpha
        new = {i["name"]: i for i in items}
        for name in list(self.state):
            if name not in new:
                self.state[name]["pct"] *= (1 - a)
        for name, i in new.items():
            if name in self.state:
                s = self.state[name]
                s["pct"] = a * i["pct"] + (1 - a) * s["pct"]
                s["bgr"] = a * i["bgr"] + (1 - a) * s["bgr"]
                s["dE"] = a * i["dE"] + (1 - a) * s["dE"]
            else:
                self.state[name] = {"name": name, "pct": a * i["pct"],
                                    "bgr": i["bgr"].copy(), "dE": i["dE"]}
        for name in [n for n, s in self.state.items() if s["pct"] < 0.5]:
            del self.state[name]

    def get(self, min_pct=MIN_PCT):
        """Normalised, rounded (sums to 100) list of colours >= min_pct."""
        total = sum(s["pct"] for s in self.state.values())
        if total <= 0:
            return []
        rows = [dict(s, pct=100.0 * s["pct"] / total) for s in self.state.values()]
        rows = [r for r in rows if r["pct"] >= min_pct]
        if not rows:
            return []
        norm = sum(r["pct"] for r in rows)
        for r in rows:
            r["pct"] = 100.0 * r["pct"] / norm
        rows.sort(key=lambda r: r["pct"], reverse=True)
        ints = [int(round(r["pct"])) for r in rows]
        ints[0] += 100 - sum(ints)          # force the displayed total to 100
        for r, v in zip(rows, ints):
            r["pct"] = v
        return rows


# --------------------------------------------------------------------------
# 5. DRAWING / UI  (Member 2 owns this section)
# --------------------------------------------------------------------------
def draw_zone(frame, rect, dominant):
    x1, y1, x2, y2 = rect
    # double outline = visible on both bright and dark backgrounds
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    L = max(14, (x2 - x1) // 8)
    for (cx, cy, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), (0, 255, 255), 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), (0, 255, 255), 3)
    put_text(frame, "SAMPLING ZONE", (x1, max(18, y1 - 10)), 0.55, (0, 255, 255), 1)

    if dominant is not None:
        label = "{}  {}%".format(dominant["name"], dominant["pct"])
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.8, 2)
        bx1, by1 = x1, y2 + 10
        bx2, by2 = x1 + tw + 56, y2 + 10 + th + 18
        shaded_rect(frame, (bx1, by1), (bx2, by2), (0, 0, 0), 0.6)
        cv2.rectangle(frame, (bx1 + 6, by1 + 6), (bx1 + 36, by2 - 6),
                      tuple(int(c) for c in dominant["bgr"]), -1)
        cv2.rectangle(frame, (bx1 + 6, by1 + 6), (bx1 + 36, by2 - 6), (255, 255, 255), 1)
        put_text(frame, label, (bx1 + 46, by2 - 10), 0.8, (255, 255, 255), 2, outline=False)


def draw_panel(height, items, message, small, valid):
    panel = np.full((height, PANEL_W, 3), (28, 28, 28), np.uint8)
    put_text(panel, "COLOR ANALYSIS", (14, 30), 0.7, (255, 255, 255), 2, outline=False)
    cv2.line(panel, (10, 42), (PANEL_W - 10, 42), (90, 90, 90), 1)

    thumb_h = 100
    row_h = 54
    top = 54
    max_rows = max(1, (height - top - thumb_h - 40) // row_h)

    if not items:
        put_text(panel, message or "Analysing...", (14, top + 24), 0.55,
                 (180, 180, 180), 1, outline=False)
    bar_x1, bar_x2 = 64, PANEL_W - 14
    for idx, it in enumerate(items[:max_rows]):
        y = top + idx * row_h
        bgr = tuple(int(c) for c in it["bgr"])
        cv2.rectangle(panel, (14, y), (54, y + 40), bgr, -1)
        cv2.rectangle(panel, (14, y), (54, y + 40), (200, 200, 200), 1)
        put_text(panel, it["name"], (64, y + 14), 0.55, (255, 255, 255), 1, outline=False)
        pct_txt = "{}%".format(it["pct"])
        (tw, _), _ = cv2.getTextSize(pct_txt, FONT, 0.55, 1)
        put_text(panel, pct_txt, (PANEL_W - 14 - tw, y + 14), 0.55, (255, 255, 255), 1, outline=False)
        cv2.rectangle(panel, (bar_x1, y + 20), (bar_x2, y + 26), (60, 60, 60), -1)
        fill = int((bar_x2 - bar_x1) * it["pct"] / 100.0)
        cv2.rectangle(panel, (bar_x1, y + 20), (bar_x1 + fill, y + 26), bgr, -1)
        r, g, b = bgr_to_rgb_tuple(it["bgr"])
        put_text(panel, "RGB({}, {}, {})  {}".format(r, g, b, bgr_to_hex(it["bgr"])),
                 (64, y + 42), 0.45, (170, 170, 170), 1, outline=False)

    # thumbnails: what the algorithm actually looks at (great for report screenshots)
    ty = height - thumb_h - 8
    tw_each = (PANEL_W - 30) // 2
    put_text(panel, "ROI (smoothed)", (10, ty - 6), 0.4, (150, 150, 150), 1, outline=False)
    put_text(panel, "Segmentation mask", (20 + tw_each, ty - 6), 0.4, (150, 150, 150), 1, outline=False)
    panel[ty:ty + thumb_h, 10:10 + tw_each] = fit_into(small, tw_each, thumb_h)
    mask_bgr = None if valid is None else cv2.cvtColor(valid, cv2.COLOR_GRAY2BGR)
    panel[ty:ty + thumb_h, 20 + tw_each:20 + 2 * tw_each] = fit_into(mask_bgr, tw_each, thumb_h)
    return panel


def draw_banner(img, text, color=(0, 0, 200)):
    h, w = img.shape[:2]
    shaded_rect(img, (0, h // 2 - 26), (w, h // 2 + 26), color, 0.75)
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    put_text(img, text, (max(10, (w - tw) // 2), h // 2 + 8), 0.8, (255, 255, 255), 2, outline=False)


# --------------------------------------------------------------------------
# 6. APPLICATION  (Member 2 owns the loop and key handling)
# --------------------------------------------------------------------------
class ColorApp:
    def __init__(self, args):
        self.k = int(np.clip(args.k, 2, 6))
        self.zone = float(np.clip(args.zone, 0.10, 0.70))
        self.every = max(1, args.every)
        # mirror the live webcam like a mirror; never flip image/video files
        self.mirror = not args.no_mirror and not getattr(args, "source", None)
        self.smoother = ColorSmoother(args.alpha)
        self.use_wb = False
        self.ignore_glare = False
        self.show_help = True
        self.bg_full = None
        self.frame_i = 0
        self.fps = 0.0
        self.t_prev = time.time()
        self.last_result = {"small": None, "valid": None, "status": "empty"}
        self.last_items = []
        self.last_display = None

    # ---- one frame -> annotated display image ----------------------------
    def process(self, frame):
        if self.mirror:
            frame = cv2.flip(frame, 1)
        if self.use_wb:
            frame = gray_world(frame)

        h, w = frame.shape[:2]
        if self.bg_full is not None and self.bg_full.shape != frame.shape:
            self.bg_full = None  # camera resolution changed
        rect = zone_rect(w, h, self.zone)
        x1, y1, x2, y2 = rect

        if self.frame_i % self.every == 0:
            roi = frame[y1:y2, x1:x2]
            bg_roi = None if self.bg_full is None else self.bg_full[y1:y2, x1:x2]
            res = analyze_zone(roi, self.k, bg_roi, self.ignore_glare)
            self.last_result = res
            if res["status"] == "ok":
                self.smoother.update(res["items"])
            else:
                self.smoother.reset()
            self.last_items = self.smoother.get()
        self.frame_i += 1

        now = time.time()
        dt = max(now - self.t_prev, 1e-6)
        self.t_prev = now
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt

        view = frame.copy()
        dominant = self.last_items[0] if self.last_items else None
        draw_zone(view, rect, dominant)

        status = self.last_result.get("status")
        msg = None
        if status == "no_object":
            msg = "No object in zone - place one inside the box"
        elif status == "empty":
            msg = "Analysing..."
        if msg and not self.last_items:
            put_text(view, msg, (x1, y2 + 34), 0.6, (0, 200, 255), 2)

        flags = "FPS {:.0f} | K={} | WB:{} | Glare:{} | BG:{}".format(
            self.fps, self.k, "on" if self.use_wb else "off",
            "ignore" if self.ignore_glare else "keep",
            "set" if self.bg_full is not None else "none")
        put_text(view, flags, (10, 22), 0.5, (0, 255, 0), 1)
        if self.show_help:
            put_text(view, "q quit | s save | b set background | c clear | w white-bal | "
                           "g glare | k clusters | [ ] zone | space freeze",
                     (10, h - 10), 0.42, (230, 230, 230), 1)

        panel = draw_panel(h, self.last_items, msg, self.last_result.get("small"),
                           self.last_result.get("valid"))
        display = np.hstack([view, panel])
        self.last_frame = frame
        self.last_display = display
        return display

    # ---- keyboard ---------------------------------------------------------
    def handle_key(self, key):
        """Returns "quit", "freeze" or None."""
        if key in (ord("q"), 27):
            return "quit"
        if key == ord(" "):
            return "freeze"
        if key == ord("k"):
            self.k = 2 if self.k >= 6 else self.k + 1
            self.smoother.reset()
        elif key == ord("w"):
            self.use_wb = not self.use_wb
            self.bg_full = None   # background must be re-captured after a WB change
            self.smoother.reset()
        elif key == ord("g"):
            self.ignore_glare = not self.ignore_glare
            self.smoother.reset()
        elif key == ord("["):
            self.zone = max(0.10, self.zone - 0.03)
            self.smoother.reset()
        elif key == ord("]"):
            self.zone = min(0.70, self.zone + 0.03)
            self.smoother.reset()
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key == ord("b"):
            if getattr(self, "last_frame", None) is not None:
                self.bg_full = self.last_frame.copy()
                self.smoother.reset()
                print("[INFO] Background captured - object isolation ON")
        elif key == ord("c"):
            self.bg_full = None
            self.smoother.reset()
            print("[INFO] Background cleared - using the whole zone")
        elif key == ord("s"):
            self.save_snapshot()
        return None

    def save_snapshot(self):
        if self.last_display is None:
            return
        try:
            os.makedirs(SHOT_DIR, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SHOT_DIR, "capture_{}.png".format(stamp))
            cv2.imwrite(path, self.last_display)
            log = os.path.join(SHOT_DIR, "results_log.csv")
            new = not os.path.exists(log)
            with open(log, "a", newline="") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow(["timestamp", "file", "dominant", "dominant_hex",
                                 "distribution", "k", "white_balance", "background_set"])
                dom = self.last_items[0] if self.last_items else None
                dist = "; ".join("{} {}%".format(i["name"], i["pct"]) for i in self.last_items)
                wr.writerow([stamp, os.path.basename(path),
                             dom["name"] if dom else "", bgr_to_hex(dom["bgr"]) if dom else "",
                             dist, self.k, self.use_wb, self.bg_full is not None])
            print("[INFO] Saved", path)
        except Exception as exc:  # never crash on a failed save
            print("[WARN] Could not save screenshot:", exc)


# --------------------------------------------------------------------------
# 7. CAMERA + MAIN LOOP
# --------------------------------------------------------------------------
def open_camera(index, width, height):
    """Try the requested camera, then fall back to indexes 0-2."""
    tried = [index] + [i for i in range(3) if i != index]
    for idx in tried:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    print("[INFO] Camera {} opened ({}x{})".format(
                        idx, frame.shape[1], frame.shape[0]))
                    return cap
            if cap is not None:
                cap.release()
        except Exception as exc:
            print("[WARN] Camera {} failed: {}".format(idx, exc))
    return None


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class StaticImageCapture:
    """Mimics cv2.VideoCapture for one still image (backup input for demos/tests)."""

    def __init__(self, path, max_side=960):
        self.frame = None
        img = cv2.imread(path)
        if img is not None:
            s = max_side / float(max(img.shape[:2]))
            if s < 1.0:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            self.frame = img

    def isOpened(self):
        return self.frame is not None

    def read(self):
        if self.frame is None:
            return False, None
        time.sleep(0.03)  # ~30 FPS pacing
        return True, self.frame.copy()

    def release(self):
        pass


class LoopingVideo:
    """Recorded video that restarts when it ends (backup input for demos/tests)."""

    def __init__(self, path, max_side=960):
        self.cap = cv2.VideoCapture(path)
        self.max_side = max_side

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:  # end of file -> rewind
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok and frame is not None:
            s = self.max_side / float(max(frame.shape[:2]))
            if s < 1.0:
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return ok, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


def open_input(args):
    """Webcam by default; --source can point to an image, video, or stream URL."""
    src = getattr(args, "source", None)
    if not src:
        return open_camera(args.camera, args.width, args.height)
    
    # Support HTTP stream URLs (such as mobile IP webcams)
    if src.startswith("http://") or src.startswith("https://"):
        print("[INFO] Connecting to stream URL:", src)
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return cap
        print("[ERROR] Could not open video stream:", src)
        return None

    if not os.path.isfile(src):
        print("[ERROR] File not found:", src)
        return None
    cap = StaticImageCapture(src) if src.lower().endswith(IMAGE_EXT) else LoopingVideo(src)
    if cap.isOpened():
        print("[INFO] Using file input:", src)
        return cap
    print("[ERROR] Could not read:", src)
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Real-Time Object Color Identifier")
    p.add_argument("--source", default=None,
                   help="optional image, video file, or network stream URL instead of the webcam")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--k", type=int, default=4, help="K-Means clusters, 2-6 (default 4)")
    p.add_argument("--zone", type=float, default=0.30,
                   help="sampling-zone size as fraction of the frame (default 0.30)")
    p.add_argument("--every", type=int, default=2, help="analyse every Nth frame (default 2)")
    p.add_argument("--alpha", type=float, default=0.25,
                   help="temporal smoothing 0-1, lower = steadier (default 0.25)")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    return p.parse_args()


def main():
    args = parse_args()
    cap = open_input(args)
    if cap is None:
        if not args.source:
            print("[ERROR] No webcam found. Close other apps using the camera "
                  "(Zoom/Teams) and try --camera 1, or use --source <image/video/URL>.")
        return 1

    app = ColorApp(args)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    paused = False
    fail = 0
    last_err = ""

    while True:
        display = None
        try:
            if paused and app.last_display is not None:
                display = app.last_display.copy()
                draw_banner(display, "FROZEN - press SPACE to resume", (120, 80, 0))
            else:
                ok, frame = (False, None)
                if cap is not None:
                    try:
                        ok, frame = cap.read()
                    except cv2.error:
                        ok = False
                if ok and frame is not None and frame.size > 0:
                    fail = 0
                    display = app.process(frame)
                else:
                    fail += 1
                    if app.last_display is not None:
                        display = app.last_display.copy()
                    else:
                        display = np.zeros((args.height, args.width + PANEL_W, 3), np.uint8)
                    draw_banner(display, "Camera frame error ({}) - retrying...".format(fail))
                    if fail >= 30:  # try to reconnect the camera
                        if cap is not None:
                            cap.release()
                        cap = open_input(args)
                        fail = 0
                        time.sleep(0.3)
        except Exception as exc:  # any processing bug -> show banner, keep running
            msg = "{}: {}".format(type(exc).__name__, exc)
            if msg != last_err:
                print("[WARN] Frame skipped -", msg)
                last_err = msg
            if app.last_display is not None:
                display = app.last_display.copy()
                draw_banner(display, "Processing error - frame skipped")

        if display is not None:
            cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        try:
            action = app.handle_key(key)
        except Exception as exc:
            print("[WARN] Key handler error:", exc)
            action = None
        if action == "quit":
            break
        if action == "freeze":
            paused = not paused
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
        except cv2.error:
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())