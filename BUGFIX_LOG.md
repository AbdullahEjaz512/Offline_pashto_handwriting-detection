# Pashto OCR Pipeline — Bug Fix Progress Log
# Created: 2026-04-27
# ============================================================

## SESSION 5  |  2026-04-27  (Latest)
### Goal Clarification
**Ultimate target:** Any realistic mobile photo of Pashto handwriting on white/off-white paper.
This means the preprocessing must handle: shadows, uneven lighting, faint ink, lined paper.
The CRNN model itself is fine — it just needs clean grayscale input (dark ink, white background).

### Image Under Test
- `test_page.jpg` — real mobile photo of ~9-line handwriting, uneven lighting/shadow

### Problem Observed
- Segmentation: ✅ YOLO correctly found 9 lines
- CRNN output: short lines at top garbled (`يالدم`, `پلاد`), longer lines partially correct
- Root cause: line crops fed to CRNN still contained shadow/lighting gradient → grayscale values ambiguous

### Fix Applied
- Added **Variant 4: Shadow Removal** back to `decode_crop_with_variants` in `pipeline.py`
  - Kernel size is scaled to crop height (`ch // 3`) not fixed 7px — safe for small crops
  - Normalizes uneven background out of each line crop before decoding
  - CTC confidence selector picks this variant only when it actually improves score

### Variant Strategy (Final State)
| Variant | Best for |
|---|---|
| Raw grayscale | Clean paper, ideal lighting |
| CLAHE | Faint ink, low local contrast |
| Adaptive threshold | High-contrast black on white |
| Shadow removal | Mobile photos with shadows / uneven light |

---


## SESSION 4  |  2026-04-27  (Latest)
### Scope Decision (Important)
**Target scope narrowed to:** Dark handwriting on white paper (blank or lined).
All complexity added for colored ink/paper, printed fonts, pink/blue backgrounds is REMOVED.

### Problems Fixed
| Problem | Root Cause | Fix |
|---|---|---|
| 5 lines from 15-line page | Morphological close merged adjacent lines; ruled lines filled projection valleys | Added `_remove_ruled_lines()` + gentler smoothing kernel (`h//120` vs `h//60`) + lower ink threshold (1% vs 2%) |
| Over-engineered variants | 5 variants (HSV, shadow-removal) added for colored paper — not needed | Reduced to 3 variants: raw gray, CLAHE, adaptive threshold |
| Multiple fallback strategies | Morphological + projection both running — unnecessary complexity | Simplified to YOLO + projection profile only |

### Before / After
| Metric | Before | After |
|---|---|---|
| Lines detected (dense 15-line page on lined paper) | 5 (merge problem) | Expected ~15 |
| Preprocessing variants per crop | 5 (HSV, shadow, CLAHE, raw, adaptive) | 3 (raw, CLAHE, adaptive) |
| Fallback strategies | 3 (YOLO + morpho + projection) | 2 (YOLO + projection) |

---



### Problem Observed
- Only 5 fragmented lines extracted from a ~15-line page
- All 5 outputs were garbled: `بي ي`, `اکي ورديه`, etc.

### Root Cause
- The morphological close kernel (wide horizontal, 3px vertical) was merging adjacent, closely-spaced text lines into single large blobs.  
- The contour finder then saw only 5 merged blobs instead of 15 individual lines.

### Fix Applied
- Added **Strategy C: Horizontal Projection Profile** to `segmenter.py`.
  - Sums ink pixels row by row to build a 1D profile of where text is vs empty space.
  - Finds valleys (empty rows between lines) and uses those as line boundaries.
  - This is the standard, robust OCR technique for dense handwriting and is immune to the morphological merge problem.
- Now runs THREE strategies (YOLO, Morphological, Projection) and picks the one with the most valid text crops.

### Before / After
| Metric | Before | After |
|---|---|---|
| Lines detected (pink page ~15 lines) | 5 | Expected ~13-15 |
| Method | YOLO + morphological only | YOLO + morphological + projection profile |

---


# ============================================================

## SESSION 3  |  2026-04-27  (Current)
### Images Under Test
| Ref  | File                                  | Description                        |
|------|---------------------------------------|------------------------------------|
| SS1  | WhatsApp_Image_at_1.04.43_AM.jpeg     | Single line, light-purple ink, white paper |
| SS2  | WhatsApp_Image_at_1.37.50_AM.jpeg     | 7-line printed Naskh text, dark border frame |
| SS3  | WhatsApp_Image_at_1.35.15_AM.jpeg     | 2-line blue handwriting on blue/silver card |

### Status
| Image | Before This Session       | After This Session         |
|-------|---------------------------|----------------------------|
| SS1   | ✅ Working                | ✅ Still working           |
| SS2   | Segmented 6/7 lines (garbled text) | Segmented 7 lines (garbled — model trained on handwriting, not print — unfixable without retraining) |
| SS3   | ❌ Only 1 line detected, output `ي په مه ي` | 🔄 Needs re-test after latest fix |

### Root Causes Found This Session
1. **SS3 — YOLO missed 2nd line**: conf threshold 0.10 was too high for the low-contrast blue card. Second line scored ~0.016 and was dropped.
   - **Fix**: Lowered YOLO conf to `0.08`.

2. **download.jpg — Over-segmentation on lined paper**: Fallback `_run_binarize` had `bh < 6` as minimum height. Horizontal ruled lines (1–5px) passed this filter and were decoded as garbage text lines.
   - **Fix**: `min_h = max(12, h // 35)` — scales with image resolution. Added `bw/bh > 60` aspect ratio filter for ultra-thin slivers.

3. **Scoring chose fallback by raw crop count, not quality**: `fallback_score[0] > yolo_score[0]` rewarded fallback simply for finding more crops, including junk ones.
   - **Fix**: Changed score to use **average ink density per crop**. Fallback now only wins if it found `> YOLO_count + 1` meaningful lines.

4. **Variant selector used string length as proxy for quality**: `max(candidates, key=len)` chose the longest decoded string, which was often a noisy garbage string from a bad preprocessing variant.
   - **Fix**: Changed to use **CTC confidence score** (`mean character probability`) from the neural network itself.

5. **Shadow-removal kernels too large for small line crops in pipeline.py**: Fixed kernel sizes (7×7, 21px blur) were designed for full pages. On a 50px-tall line crop, they blurred everything into uniform gray.
   - **Fix**: Kernels now scale with crop height: `k = max(3, min(ch//4, 15)) | 1`.

---

## SESSION 2  |  2026-04-26  (Previous)
### Changes Made
| File         | Change                                                                 | Result |
|--------------|------------------------------------------------------------------------|--------|
| segmenter.py | Added `_clean_image()` shadow removal + background normalisation       | Better YOLO detection on low-contrast images |
| segmenter.py | Added `_color_aware_binary()` HSV Value-channel binarisation           | Helps detect blue ink on blue paper |
| segmenter.py | Changed `RETR_EXTERNAL` → `RETR_LIST` in contour detection             | Finds text inside dark border frames (SS2) |
| segmenter.py | Added ink-density scoring to compare YOLO vs fallback results          | Pipeline picks the better segmentation |
| pipeline.py  | Added `import numpy as np` (was missing — caused NameError)            | Crash fixed |
| pipeline.py  | Added multi-variant decoding (raw, CLAHE, shadow-removed, adaptive)    | More robust on low-contrast crops |
| pipeline.py  | Changed variant selector from `max(len)` to CTC confidence score       | Picks correct text instead of long garbage |
| app.py       | Added graceful handling for empty OCR results                          | No more 500 errors on failed images |

### Before / After — Key Metric
| Metric                    | Before       | After        |
|---------------------------|--------------|--------------|
| SS3 crops detected        | 0 (crashed)  | 2            |
| SS2 lines detected        | 1            | 6–7          |
| SS1 output                | crash/empty  | ✅ correct   |
| Pipeline crash on blue bg | Yes          | No           |

---

## SESSION 1  |  2026-04-26  (Initial analysis)
### Problem Statement
Pipeline worked on clean black-and-white high-contrast images but failed completely on:
- Low-contrast images (blue ink on blue/silver paper)
- Shadowed or unevenly lit photographs
- Images with dark border frames around text

### Original Architecture
```
Image → YOLO (conf=0.25) → crop → raw grayscale → CRNN → text
```
No fallback. No preprocessing. Single variant decoding. No error handling.

### Initial Failure Modes
- Blue card image: YOLO found 0 crops (conf too high for low contrast)
- Printed text with border: RETR_EXTERNAL missed all inner text contours
- Any failure → 500 server crash (no error handling)
