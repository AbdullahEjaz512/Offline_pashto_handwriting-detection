"""
OCR Technique Benchmark Script
================================
Systematically tests every combination of:
  - Preprocessing (6 variants)
  - Segmentation strategy (3 variants)
  - Variant selection logic (2 modes)

Results are written to data/ocr_benchmark.md.
Run from the PHTI directory:
    ..\.venv\Scripts\python.exe data/benchmark.py
"""

import os
import sys
import cv2
import json
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
from datetime import datetime

# ── Setup paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent  # PHTI/
sys.path.insert(0, str(ROOT))

from src.model import PashtoCRNN

VOCAB_PATH = ROOT / "models" / "vocab.json"
WEIGHTS_PATH = ROOT / "models" / "crnn_pashto.pth"
TEST_DIR = ROOT / "data" / "raw"
OUT_MD = ROOT / "data" / "ocr_benchmark.md"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load model once ───────────────────────────────────────────────────────────
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    VOCAB = json.load(f)
ID2CHAR = {int(v): k for k, v in VOCAB.items()}
BLANK_ID = VOCAB.get("<BLANK>", 0)
NUM_CLASSES = len(VOCAB)

model = PashtoCRNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded. Classes={NUM_CLASSES}")

# ── Test images ───────────────────────────────────────────────────────────────
TEST_IMAGES = sorted(TEST_DIR.glob("*.*"))
print(f"Found {len(TEST_IMAGES)} test images.")

# ── CTC Decode ────────────────────────────────────────────────────────────────
def ctc_decode(logits):
    probs = torch.softmax(logits, dim=2)
    best_ids = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()
    peak_probs = torch.max(probs, dim=2)[0].squeeze(0).cpu().numpy()
    chars, confs, last = [], [], -1
    for t, cid in enumerate(best_ids):
        if cid != BLANK_ID and cid != last:
            chars.append(ID2CHAR.get(int(cid), "?"))
            confs.append(float(peak_probs[t]))
        last = cid
    text = "".join(chars)
    conf = float(np.mean(confs)) if confs else 0.0
    return text, conf

# ── Preprocessing Techniques ──────────────────────────────────────────────────
def pp_adaptive_binary(img_bgr, h=32):
    """Adaptive threshold (51,20) — best for colored backgrounds (validated)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 20)
    binary = cv2.medianBlur(binary, 3)
    return _to_tensor(binary, h)

def pp_adaptive_binary_small(img_bgr, h=32):
    """Adaptive threshold (21,10) — better for large-text images."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
    binary = cv2.medianBlur(binary, 3)
    return _to_tensor(binary, h)

def pp_otsu(img_bgr, h=32):
    """OTSU binarization — works best when ink is clearly dark on white."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _to_tensor(binary, h)

def pp_raw_gray(img_bgr, h=32):
    """Raw grayscale, no binarization — what some models expect."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return _to_tensor(gray, h)

def pp_clahe(img_bgr, h=32):
    """CLAHE contrast enhancement + adaptive binary."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    return _to_tensor(binary, h)

def pp_morph_clean(img_bgr, h=32):
    """Adaptive binary + morphological closing to fix broken cursive joins."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return _to_tensor(binary, h)

def _to_tensor(arr, h):
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    scale = h / arr.shape[0]
    new_w = max(32, min(int(arr.shape[1] * scale), 2048))
    resized = cv2.resize(arr, (new_w, h), interpolation=cv2.INTER_AREA)
    t = TF.to_tensor(Image.fromarray(resized)).unsqueeze(0).to(DEVICE)
    return t

PREPROCESS_FNS = {
    "adaptive_51_20":    pp_adaptive_binary,
    "adaptive_21_10":    pp_adaptive_binary_small,
    "otsu":              pp_otsu,
    "raw_gray":          pp_raw_gray,
    "clahe+adaptive":    pp_clahe,
    "morph_close":       pp_morph_clean,
}

# ── Segmentation Strategies ───────────────────────────────────────────────────
def seg_full_image(img_path):
    """Treat the whole image as one line. Best for single-line images."""
    img = cv2.imread(str(img_path))
    return [img] if img is not None else []

def seg_yolo(img_path):
    """YOLO-based line detection."""
    try:
        from ultralytics import YOLO
        if not hasattr(seg_yolo, "_model"):
            seg_yolo._model = YOLO(str(ROOT / "models" / "best.pt"))
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        results = seg_yolo._model(str(img_path), conf=0.15, iou=0.4, verbose=False)[0]
        if not results.boxes or len(results.boxes) == 0:
            return []
        boxes = sorted(results.boxes.data.cpu().numpy(), key=lambda b: b[1])
        crops = []
        for box in boxes:
            x1, y1, x2, y2, *_ = box
            px1, py1 = max(0, int(x1)-50), max(0, int(y1)-4)
            px2, py2 = min(w, int(x2)+50), min(h, int(y2)+4)
            c = img[py1:py2, px1:px2]
            if c.shape[0] >= 6 and c.shape[1] >= 6:
                crops.append(c)
        return crops
    except Exception as e:
        return []

def seg_projection(img_path):
    """Projection profile — calibrated for dense colored backgrounds."""
    img = cv2.imread(str(img_path))
    if img is None: return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 20)
    proj = np.sum(binary > 0, axis=1)
    gap_thr = max(2, int(w * 0.01))
    in_line, line_start, regions = False, 0, []
    for i, count in enumerate(proj):
        if not in_line and count > gap_thr:
            in_line, line_start = True, i
        elif in_line and count <= gap_thr:
            in_line = False
            if i - line_start >= 6:
                regions.append([line_start, i])
    if in_line:
        regions.append([line_start, h])
    merged = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < 4:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    crops = []
    for y1, y2 in merged:
        c = img[max(0,y1-3):min(h,y2+3), :]
        if c.shape[0] >= 6:
            crops.append(c)
    return crops

SEGMENT_FNS = {
    "full_image": seg_full_image,
    "yolo":       seg_yolo,
    "projection": seg_projection,
}

# ── Run benchmark ─────────────────────────────────────────────────────────────
results = {}  # {img_name: {technique: {text, conf, n_lines, time_ms}}}

for img_path in TEST_IMAGES:
    img_name = img_path.name
    results[img_name] = {}
    img_bgr_full = cv2.imread(str(img_path))
    if img_bgr_full is None:
        continue
    ih, iw = img_bgr_full.shape[:2]
    print(f"\n{'='*60}")
    print(f"Image: {img_name}  ({iw}x{ih})")

    for seg_name, seg_fn in SEGMENT_FNS.items():
        t0 = time.time()
        crops = seg_fn(img_path)
        seg_time = (time.time() - t0) * 1000

        for pp_name, pp_fn in PREPROCESS_FNS.items():
            key = f"{seg_name} + {pp_name}"
            line_texts, line_confs = [], []

            for crop in crops:
                # Test Normal + Flipped, pick best by confidence
                tensor = pp_fn(crop)
                if tensor is None:
                    continue
                flipped = torch.flip(tensor, dims=[3])

                best_t, best_c = "", 0.0
                for variant in [tensor, flipped]:
                    with torch.no_grad():
                        logits = model(variant)
                    text, conf = ctc_decode(logits)
                    if conf > best_c and text.strip():
                        best_t, best_c = text.strip(), conf

                if best_t:
                    line_texts.append(best_t)
                    line_confs.append(best_c)

            total_time = (time.time() - t0) * 1000
            avg_conf = float(np.mean(line_confs)) if line_confs else 0.0
            completeness = len(line_confs) / len(crops) if crops else 0.0
            final_score = avg_conf * completeness

            results[img_name][key] = {
                "n_crops":    len(crops),
                "n_decoded":  len(line_texts),
                "avg_conf":   round(avg_conf, 3),
                "completeness": round(completeness, 3),
                "score":      round(final_score, 3),
                "time_ms":    round(total_time, 1),
                "text":       " | ".join(line_texts[:3]),  # first 3 lines preview
            }
            print(f"  [{key}] crops={len(crops)} decoded={len(line_texts)} "
                  f"conf={avg_conf:.3f} score={final_score:.3f}")

# ── Write Markdown Report ─────────────────────────────────────────────────────
now = datetime.now().strftime("%Y-%m-%d %H:%M")
md_lines = [
    f"# OCR Technique Benchmark",
    f"_Generated: {now}_\n",
    "## Overview",
    "Tests every combination of **Segmentation** × **Preprocessing** across all 5 test images.",
    "**Score** = avg_char_confidence × completeness (fraction of lines decoded).\n",
    "---",
]

for img_name, img_results in results.items():
    md_lines.append(f"\n## 🖼️ {img_name}\n")
    md_lines.append("| Technique | Crops | Decoded | Avg Conf | Completeness | **Score** | Time (ms) | Text Preview |")
    md_lines.append("|---|---|---|---|---|---|---|---|")

    sorted_results = sorted(img_results.items(), key=lambda x: x[1]["score"], reverse=True)
    for tech, r in sorted_results:
        preview = r["text"][:60].replace("|", "/") if r["text"] else "_(empty)_"
        md_lines.append(
            f"| {tech} | {r['n_crops']} | {r['n_decoded']} | {r['avg_conf']:.3f} | "
            f"{r['completeness']:.2f} | **{r['score']:.3f}** | {r['time_ms']} | {preview} |"
        )

# Global best
md_lines.append("\n---\n## 🏆 Best Technique Per Image\n")
md_lines.append("| Image | Best Technique | Score | Text Preview |")
md_lines.append("|---|---|---|---|")
for img_name, img_results in results.items():
    if not img_results: continue
    best_tech, best_r = max(img_results.items(), key=lambda x: x[1]["score"])
    preview = best_r["text"][:80].replace("|", "/") if best_r["text"] else "_(empty)_"
    md_lines.append(f"| {img_name} | {best_tech} | {best_r['score']:.3f} | {preview} |")

md_lines.append("\n---\n## ⚙️ Configuration Used\n")
md_lines.append(f"- Model: `{WEIGHTS_PATH.name}`")
md_lines.append(f"- Vocab size: {NUM_CLASSES}")
md_lines.append(f"- Device: {DEVICE}")
md_lines.append(f"- Test images: {[p.name for p in TEST_IMAGES]}")

OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"\n✅ Benchmark complete. Report written to: {OUT_MD}")
