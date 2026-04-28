"""
Production Pipeline Final Verification
======================================
Runs our production pipeline on all images provided by the user
to assess accuracy, confidence, and readability.
"""

import os
import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FullPagePashtoRecognition

TEST_DIR = ROOT / "data" / "raw"
all_images = sorted(list(TEST_DIR.glob("*.*")))

print("=" * 60)
print("             🚀 PRODUCTION PIPELINE RESULTS 🚀             ")
print("=" * 60)

recognizer = FullPagePashtoRecognition()

for img_path in all_images:
    print(f"\n🖼️ Testing Image: {img_path.name}")
    try:
        results, score = recognizer.process_page(str(img_path))
        print(f"   ↳ Overall Confidence Score: {score:.2f}%")
        print(f"   ↳ Lines Detected: {len(results)}")
        print("   --- Extracted Text ---")
        for i, text in enumerate(results):
            print(f"     Line {i+1}: {text}")
    except Exception as e:
        print(f"   ❌ Error processing image: {e}")

print("\n" + "=" * 60)
