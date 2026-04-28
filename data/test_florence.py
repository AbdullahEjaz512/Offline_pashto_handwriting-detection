"""
Download and Test Florence-2 Fallback Engine
============================================
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def main():
    try:
        from src.florence_fallback import FlorenceFallback
        print("Initializing Florence-2 download...")
        engine = FlorenceFallback()
        
        test_img = ROOT / "test_page.jpg"
        if not test_img.exists():
            test_img = ROOT / "data" / "raw" / "synthetic_multiline.jpg"
            
        print(f"Testing OCR on {test_img}...")
        extracted = engine.extract_text(str(test_img))
        
        print("\n" + "="*40)
        print(f"Florence-2 Success! Extracted Text:\n{extracted}")
        print("="*40)
        
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    main()
