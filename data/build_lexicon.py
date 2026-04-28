"""
Build Lexicon Cache
===================
Iterates through the dataset once, extracting words and caching them in JSON.
"""

import os
import re
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "PHTI-DATASET"
OUT_JSON = ROOT / "models" / "pashto_lexicon.json"

def build():
    valid_words = set()
    word_freq = {}
    
    if not DATASET_DIR.exists():
        print(f"Dataset directory {DATASET_DIR} not found.")
        return

    txt_files = list(DATASET_DIR.glob("*.txt"))
    print(f"Reading {len(txt_files)} text files...")
    
    for i, txt_file in enumerate(txt_files):
        if i % 5000 == 0 and i > 0:
            print(f" Processed {i} / {len(txt_files)} files...")
            
        try:
            text = txt_file.read_text(encoding="utf-8").strip()
            words = re.findall(r'\b\w+\b', text)
            for w in words:
                if len(w) >= 2:
                    valid_words.add(w)
                    word_freq[w] = word_freq.get(w, 0) + 1
        except Exception:
            continue

    # Save to JSON
    data = {
        "valid_words": list(valid_words),
        "word_freq": word_freq
    }
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Lexicon built! Saved {len(valid_words)} unique words to {OUT_JSON}.")

if __name__ == "__main__":
    build()
