"""
Synthetic Multi-line Tester
===========================
Generates a realistic multi-line document by stitching single-line
images from PHTI-DATASET. Then tests our production pipeline on it.
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FullPagePashtoRecognition

# 1. Pick 5 random dataset samples
DATASET_DIR = ROOT / "PHTI-DATASET"
all_images = sorted(list(DATASET_DIR.glob("*.jpg")))
random.seed(42)  # For reproducibility

# Ensure we get samples with readable text
selected_samples = []
while len(selected_samples) < 5:
    img_path = random.choice(all_images)
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        if len(text) > 10:  # Avoid very short labels
            selected_samples.append((img_path, text))

print("--- Selected Ground Truth Lines ---")
for i, (path, text) in enumerate(selected_samples):
    print(f" Line {i+1} Ground Truth: {text}")

# 2. Stitch them vertically with a white gap
stitched_img = None
gap_h = 25

for i, (img_path, _) in enumerate(selected_samples):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    # Pad to equal width if necessary
    if stitched_img is not None:
        target_w = max(stitched_img.shape[1], img.shape[1])
        # Pad existing
        if stitched_img.shape[1] < target_w:
            pad = np.ones((stitched_img.shape[0], target_w - stitched_img.shape[1], 3), dtype=np.uint8) * 255
            stitched_img = np.hstack([stitched_img, pad])
        # Pad new
        if img.shape[1] < target_w:
            pad = np.ones((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8) * 255
            img = np.hstack([img, pad])
            
        # Add white gap
        gap = np.ones((gap_h, target_w, 3), dtype=np.uint8) * 255
        stitched_img = np.vstack([stitched_img, gap, img])
    else:
        stitched_img = img

out_path = ROOT / "data" / "raw" / "synthetic_multiline.jpg"
cv2.imwrite(str(out_path), stitched_img)
print(f"\nSynthetic image created at: {out_path} ({stitched_img.shape[1]}x{stitched_img.shape[0]})")

# 3. Run full production pipeline
print("\n--- Running Production Pipeline ---")
recognizer = FullPagePashtoRecognition()
results, overall_score = recognizer.process_page(str(out_path))

print("\n--- Pipeline Predictions ---")
for i, pred in enumerate(results):
    print(f" Line {i+1} Predicted: {pred}")

print(f"\nOverall Confidence Score: {overall_score:.2f}%")
