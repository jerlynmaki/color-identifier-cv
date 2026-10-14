"""
web.py
Web interface for the Real-Time Object Color Identifier.
The browser captures the webcam and sends the cropped sampling zone here.
This server runs the OpenCV pipeline and returns the colors as JSON.

Run:  python web.py   then open http://127.0.0.1:5000
"""

import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---------------- Configuration ----------------
SAMPLE_SIZE = 48      # ROI is downsampled to SAMPLE_SIZE x SAMPLE_SIZE
K_CLUSTERS = 6        # initial K-Means clusters (merged afterwards)
MERGE_DIST = 14.0     # clusters closer than this in LAB are treated as one color
MIN_PERCENT = 4.0     # ignore colors below this share of the zone

# White balance gains per channel (B, G, R) in linear light. None = not calibrated.
WB_GAIN = None
GAMMA = 2.2


# ---------------- Color naming (rule based) ----------------
def name_color(bgr):
    """Name a BGR color from its hue family, brightness and saturation.
    Uses LAB for lightness/chroma and HSV for hue and saturation."""
    arr = np.clip(np.asarray(bgr, dtype=np.float32), 0, 255).reshape(1, 1, 3)
    lab = cv2.cvtColor(arr / 255.0, cv2.COLOR_BGR2LAB)[0, 0]
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    L = float(lab[0])
    C = float(np.hypot(lab[1], lab[2]))
    h = float(hsv[0]) * 2.0          # hue in degrees 0..360
    S = float(hsv[1]) / 255.0
    V = float(hsv[2]) / 255.0

    # Neutrals: named by brightness only
    if L < 9 or (L < 22 and C < 14):
        return "Black"
    if C < 11 or S < 0.10:
        if L < 38:
            return "Dark Gray"
        if L < 62:
            return "Gray"
        if L < 82:
            return "Light Gray"
        return "White"

    # Red and pink family
    if h >= 345 or h < 10:
        if V < 0.50:
            return "Maroon"
        if V < 0.68:
            return "Dark Red"
        if S < 0.45 and V > 0.75:
            return "Pink"
        if S < 0.62 and V > 0.75:
            return "Salmon"
        if 345 <= h < 357:
            return "Crimson Red"
        return "Red"

    # Orange and brown family
    if h < 40:
        if V < 0.62:
            return "Dark Brown" if V < 0.38 else "Brown"
        if S < 0.20:
            return "Beige"
        if S < 0.50:
            return "Peach" if (V > 0.90 and S < 0.40) else "Tan"
        if h < 20:
            return "Coral" if S < 0.78 else "Red Orange"
        return "Burnt Orange" if V < 0.85 else "Orange"

    # Yellow family
    if h < 68:
        if V < 0.62:
            if V < 0.38:
                return "Dark Brown"
            return "Olive Green" if S >= 0.50 else "Brown"
        if S < 0.22:
            return "Cream" if L > 88 else "Beige"
        if S < 0.50:
            return "Khaki"
        if V < 0.88:
            return "Mustard"
        return "Gold" if h < 52 else "Yellow"

    # Yellow green family
    if h < 95:
        if V < 0.62:
            return "Olive Green" if S >= 0.30 else "Dark Gray"
        if S < 0.35:
            return "Sage Green"
        return "Lime Green" if (S > 0.60 and V > 0.75) else "Olive Green"

    # Green family
    if h < 165:
        if V < 0.30:
            return "Dark Green"
        if V < 0.45:
            return "Forest Green"
        if S < 0.50 and V > 0.80:
            return "Mint Green"
        if S < 0.30:
            return "Sage Green"
        if h < 110 and S > 0.60 and V > 0.75:
            return "Lime Green"
        return "Green"

    # Teal and cyan family
    if h < 190:
        if V < 0.62:
            return "Teal"
        if S < 0.50 and V > 0.80:
            return "Light Blue"
        return "Turquoise" if h < 183 else "Cyan"

    # Blue family
    if h < 258:
        if V < 0.45:
            return "Navy Blue"
        if V > 0.75 and S < 0.45:
            if h >= 235:
                return "Lavender"
            return "Light Blue" if S < 0.33 else "Sky Blue"
        if V > 0.75 and S < 0.65 and h < 225:
            return "Sky Blue"
        return "Blue" if S > 0.85 else "Royal Blue"

    # Purple family
    if h < 315:
        if V < 0.35:
            return "Dark Purple"
        if L > 72 and S < 0.50:
            return "Lavender"
        if h < 285 and V < 0.60:
            return "Indigo"
        if S > 0.80 and V > 0.80 and h >= 290:
            return "Magenta"
        return "Violet" if V >= 0.75 else "Purple"

    # Magenta and pink family (315..345)
    if V < 0.50:
        return "Maroon"
    if S < 0.45 and V > 0.75:
        return "Pink"
    if h < 320 and S > 0.70:
        return "Magenta"
    if S >= 0.45 and V > 0.75:
        return "Hot Pink"
    return "Magenta"


def to_rgb_hex(bgr):
    """Convert a BGR triple to ((R, G, B), '#RRGGBB')."""
    b, g, r = (int(v) for v in np.clip(np.round(bgr), 0, 255))
    return (r, g, b), "#{:02X}{:02X}{:02X}".format(r, g, b)


def apply_white_balance(img_u8):
    """Apply the calibrated gains in linear light (no effect until calibrated)."""
    if WB_GAIN is None:
        return img_u8
    lin = (img_u8.astype(np.float32) / 255.0) ** GAMMA
    lin *= WB_GAIN
    out = np.clip(lin, 0.0, 1.0) ** (1.0 / GAMMA) * 255.0
    return out.astype(np.uint8)


def lab_distance(a, b):
    """Color distance in LAB with reduced weight on brightness (shadow tolerant)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d[0] *= 0.5
    return float(np.linalg.norm(d))


# ---------------- Core analysis ----------------
def analyze_zone(roi_bgr):
    """Return (colors, mask, quantized) for the sampling zone."""
    # Image operations: noise reduction and downsampling
    blurred = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
    small = cv2.resize(blurred, (SAMPLE_SIZE, SAMPLE_SIZE), interpolation=cv2.INTER_AREA)
    small = apply_white_balance(small)

    # Segmentation: HSV threshold removes glare, morphology cleans the mask
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    glare = (hsv[:, :, 2] >= 252) & (hsv[:, :, 1] <= 15)
    mask = np.where(glare, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if cv2.countNonZero(mask) < 0.15 * mask.size:  # object may be truly white
        mask[:] = 255
    valid = mask > 0

    # Color space conversion: BGR -> LAB
    lab_img = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab_px = np.ascontiguousarray(lab_img[valid], dtype=np.float32)
    bgr_px = small[valid]
    total = len(lab_px)

    # K-Means clustering in LAB
    k = max(1, min(K_CLUSTERS, total))
    cv2.setRNGSeed(7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(lab_px, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    # Per-cluster statistics
    clusters = []
    for c in range(k):
        sel = labels == c
        count = int(sel.sum())
        if count == 0:
            continue
        clusters.append({"count": count,
                         "lab": lab_px[sel].mean(axis=0).astype(np.float64),
                         "bgr": bgr_px[sel].mean(axis=0).astype(np.float64),
                         "ids": [c]})

    # Merge clusters that are close in color (light and shadow of the same surface)
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = lab_distance(clusters[i]["lab"], clusters[j]["lab"])
                if d < MERGE_DIST and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = clusters[i], clusters[j]
        n = a["count"] + b["count"]
        a["lab"] = (a["lab"] * a["count"] + b["lab"] * b["count"]) / n
        a["bgr"] = (a["bgr"] * a["count"] + b["bgr"] * b["count"]) / n
        a["count"] = n
        a["ids"] += b["ids"]
        del clusters[j]

    # Name each cluster and group clusters that received the same name
    group_of_cluster = np.zeros(k, dtype=np.int32)
    groups = {}  # name -> {count, acc, gid}
    for cl in clusters:
        name = name_color(cl["bgr"])
        g = groups.setdefault(name, {"count": 0, "acc": np.zeros(3), "gid": len(groups)})
        g["count"] += cl["count"]
        g["acc"] += cl["bgr"] * cl["count"]
        for cid in cl["ids"]:
            group_of_cluster[cid] = g["gid"]

    names = list(groups.keys())
    means = np.array([groups[n]["acc"] / groups[n]["count"] for n in names], dtype=np.float32)

    results = [{"name": n, "pct": 100.0 * groups[n]["count"] / total,
                "bgr": groups[n]["acc"] / groups[n]["count"]} for n in names]
    results.sort(key=lambda r: -r["pct"])

    # Drop tiny colors, renormalize so percentages sum to 100
    kept = [r for r in results if r["pct"] >= MIN_PERCENT] or results[:1]
    scale = 100.0 / sum(r["pct"] for r in kept)
    for r in kept:
        r["pct"] *= scale

    # Preview image: every pixel painted with its color-group mean
    quantized = np.zeros_like(small)
    quantized[valid] = means[group_of_cluster[labels]].astype(np.uint8)

    return kept, mask, quantized


# ---------------- Helpers and routes ----------------
def encode_png(img):
    """Encode an OpenCV image as a base64 PNG data URL."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def decode_roi():
    arr = np.frombuffer(request.get_data(), dtype=np.uint8)
    roi = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if roi is None or roi.size == 0:
        return None
    return roi


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/calibrate")
def calibrate():
    """White calibration: the zone must show plain white paper.
    Computes per-channel gains (in linear light) that map it to near white."""
    global WB_GAIN
    try:
        roi = decode_roi()
        if roi is None:
            return jsonify(error="Invalid frame"), 400
        lin = (roi.astype(np.float32) / 255.0) ** GAMMA
        median_lin = np.median(lin.reshape(-1, 3), axis=0)
        WB_GAIN = np.clip(0.88 / np.maximum(median_lin, 1e-3), 0.4, 6.0).astype(np.float32)
        return jsonify(ok=True, gain=[round(float(g), 2) for g in WB_GAIN])
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.post("/reset")
def reset():
    """Remove white balance calibration."""
    global WB_GAIN
    WB_GAIN = None
    return jsonify(ok=True)


@app.post("/analyze")
def analyze():
    """Receive one JPEG of the sampling zone, return the color distribution."""
    try:
        roi = decode_roi()
        if roi is None:
            return jsonify(error="Invalid frame"), 400

        colors, mask, quantized = analyze_zone(roi)

        out = []
        for c in colors:
            rgb, hex_code = to_rgb_hex(c["bgr"])
            out.append({"name": c["name"], "pct": round(float(c["pct"]), 1),
                        "rgb": list(rgb), "hex": hex_code})

        payload = {"colors": out}
        # Preview images are only sent when requested (keeps replies small)
        if request.args.get("preview") == "1":
            payload["mask"] = encode_png(mask)
            payload["segmented"] = encode_png(quantized)
        return jsonify(payload)
    except Exception as exc:  # never crash a live demo
        return jsonify(error=str(exc)), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)