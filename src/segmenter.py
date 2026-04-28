import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSegmenter:
    """
    Production Segmenter — Evidence-based Design + Universal Binarization.
    
    FORENSIC FINDINGS:
    - YOLO misses lines in images with aspect ratio < 5:1 (multi-line blocks)
    - Crops < 8px tall are noise, but >8px can be valid on dense pages.
    
    STRATEGY:
    1. Run YOLO
    2. For any image where YOLO boxes cover < 70% of page height, 
       also run Projection and merge the results.
    3. Projection uses Universal Binarization (bg division) to perfectly 
       extract lines from colored/pink paper.
    """

    MIN_CROP_HEIGHT = 8  # Restored to 8 to support dense pages (e.g. 13 lines in 236px height)

    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)

    # ── Utility ────────────────────────────────────────────────────────────────
    def _y_overlap(self, r1, r2):
        """IoU in Y-axis only — for deduplication."""
        y1 = max(r1[0], r2[0])
        y2 = min(r1[1], r2[1])
        if y2 <= y1:
            return 0.0
        inter = y2 - y1
        union = max(r1[1], r2[1]) - min(r1[0], r2[0])
        return inter / union if union > 0 else 0.0

    # ── Engine 1: YOLO ─────────────────────────────────────────────────────────
    def _yolo_segment(self, image_path, img):
        h, w = img.shape[:2]
        results = self.model(image_path, conf=0.15, iou=0.4, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return [], []

        boxes = sorted(results.boxes.data.cpu().numpy(), key=lambda b: b[1])
        crops, y_ranges = [], []
        for box in boxes:
            x1, y1, x2, y2, *_ = box
            iy1, iy2 = max(0, int(y1) - 4), min(h, int(y2) + 4)
            ix1, ix2 = max(0, int(x1) - 50), min(w, int(x2) + 50)
            crop = img[iy1:iy2, ix1:ix2]
            if crop.shape[0] >= self.MIN_CROP_HEIGHT and crop.shape[1] >= 10:
                crops.append(crop)
                y_ranges.append((iy1, iy2))

        return crops, y_ranges

    # ── Engine 2: Universal Projection Profile ──────────────────────────────────
    def _binarize_universal(self, img):
        """
        Universal Binarization (Professional Grade).
        Handles: Pink paper, shadows, glare, and low contrast.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        h, w = gray.shape
        
        # 1. Estimate background using morphological dilation
        k_size = max(11, h // 40) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        bg = cv2.dilate(gray, kernel)
        bg = cv2.medianBlur(bg, k_size)
        
        # 2. Divide original by background to remove paper color/shadows perfectly
        norm = cv2.divide(gray, bg, scale=255)
        
        # 3. Final Binarization
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _remove_ruled_lines(self, binary, w):
        """Erase horizontal ruled lines so they don't block gap detection."""
        min_len = max(30, w // 5)
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        ruled = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_h)
        return cv2.subtract(binary, ruled)

    def _projection_split(self, img, exclude_y_ranges=None):
        """
        Splits a full image into lines using a Peak-Based Profile method.
        Highly robust to touching lines.
        """
        h, w = img.shape[:2]
        
        binary = self._binarize_universal(img)
        binary = self._remove_ruled_lines(binary, w)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        profile = binary.sum(axis=1).astype(np.float32)
        smooth_k = max(3, h // 120) | 1
        profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, smooth_k), 0).flatten()

        import scipy.signal
        # Find peaks corresponding to line centers
        distance = max(10, h // 25)
        prominence = float(profile.max()) * 0.05
        peaks, _ = scipy.signal.find_peaks(profile, distance=distance, prominence=prominence)
        
        if len(peaks) == 0:
            return []

        # Find split points (troughs) between adjacent peaks
        splits = [0]
        for i in range(len(peaks) - 1):
            p1 = peaks[i]
            p2 = peaks[i+1]
            trough = p1 + np.argmin(profile[p1:p2])
            splits.append(int(trough))
        splits.append(h)

        crops = []
        for i in range(len(splits) - 1):
            y1, y2 = splits[i], splits[i+1]
            
            # Refine y bounds by removing empty top/bottom space in the band
            band_profile = profile[y1:y2]
            band_threshold = float(band_profile.max()) * 0.02
            above_thresh = np.where(band_profile > band_threshold)[0]
            if above_thresh.size > 0:
                y1_refined = y1 + above_thresh[0]
                y2_refined = y1 + above_thresh[-1]
            else:
                y1_refined, y2_refined = y1, y2
            
            # Skip if already heavily covered by YOLO
            if exclude_y_ranges:
                overlap = max([self._y_overlap((y1_refined, y2_refined), yr) for yr in exclude_y_ranges] + [0.0])
                if overlap > 0.6:  # Increased overlap requirement to skip
                    continue
            
            # Extract crop full width
            col_proj = binary[y1_refined:y2_refined, :].sum(axis=0)
            ink_cols = np.where(col_proj > 0)[0]
            if ink_cols.size < 15 or np.sum(binary[y1_refined:y2_refined, :] > 0) < 150:
                continue
            x1 = max(0, int(ink_cols[0]) - 8)
            x2 = min(w, int(ink_cols[-1]) + 8)
            
            crop = img[y1_refined:y2_refined, x1:x2]
            if crop.shape[0] >= self.MIN_CROP_HEIGHT:
                crops.append((y1_refined, crop))

        print(f"DEBUG Projection: {len(crops)} new line(s) from peak detection.")
        return crops

    # ── Main Entry Point ────────────────────────────────────────────────────────
    def segment_lines(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]

        # 1. YOLO on Raw Image
        yolo_crops_raw, yolo_y_ranges_raw = self._yolo_segment(image_path, img)
        
        # 2. YOLO on Universal Binarized Image (Handles dark/pink paper)
        binary = self._binarize_universal(img)
        binary_no_lines = self._remove_ruled_lines(binary, w)
        binary_yolo = cv2.bitwise_not(binary_no_lines)
        binary_bgr = cv2.cvtColor(binary_yolo, cv2.COLOR_GRAY2BGR)
        
        # Write to temp file so self.model handles it correctly via path
        temp_path = "temp_yolo_bin.jpg"
        cv2.imwrite(temp_path, binary_bgr)
        yolo_crops_bin, yolo_y_ranges_bin = self._yolo_segment(temp_path, img) # Extract from original `img` using boxes from `temp_path`
        import os
        if os.path.exists(temp_path): os.remove(temp_path)

        # Merge Y ranges using NMS (Non-Maximum Suppression)
        all_y_ranges = sorted(yolo_y_ranges_raw + yolo_y_ranges_bin, key=lambda x: x[0])
        keep_ranges = []
        for yr in all_y_ranges:
            max_overlap = max([self._y_overlap(yr, k) for k in keep_ranges] + [0.0])
            if max_overlap < 0.5:
                keep_ranges.append(list(yr))
                
        merged_ranges = keep_ranges

        print(f"DEBUG Segmenter: YOLO Raw → {len(yolo_crops_raw)}, YOLO Bin → {len(yolo_crops_bin)}. Merged YOLO → {len(merged_ranges)} region(s)")

        # Only run Projection fallback if YOLO found nothing or covered very little of the image
        total_yolo_height = sum([yr[1] - yr[0] for yr in merged_ranges])
        yolo_coverage = total_yolo_height / h if h > 0 else 0
        print(f"DEBUG Segmenter: YOLO page coverage = {yolo_coverage * 100:.1f}%")

        if len(merged_ranges) > 0 and yolo_coverage > 0.35:
            proj_crops_with_y = []
        else:
            proj_crops_with_y = self._projection_split(img, exclude_y_ranges=None)

        # Build definitive line list via NMS across YOLO and Projection
        final_crops_with_y = []
        
        # Add YOLO crops first
        for y1, y2 in merged_ranges:
            col_proj = binary_no_lines[y1:y2, :].sum(axis=0)
            ink_cols = np.where(col_proj > 0)[0]
            if ink_cols.size < 15 or np.sum(binary_no_lines[y1:y2, :] > 0) < 150:
                continue
            x1 = max(0, int(ink_cols[0]) - 8)
            x2 = min(w, int(ink_cols[-1]) + 8)
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] >= self.MIN_CROP_HEIGHT:
                final_crops_with_y.append((y1, crop))

        # Add Projection crops ONLY if they don't heavily overlap with already added crops
        for py, pc in proj_crops_with_y:
            ph, pw = pc.shape[:2]
            pyr = [py, py + ph]
            
            max_overlap = 0.0
            for fy, fc in final_crops_with_y:
                fh = fc.shape[0]
                fyr = [fy, fy + fh]
                overlap = self._y_overlap(pyr, fyr)
                if overlap > max_overlap:
                    max_overlap = overlap
            
            if max_overlap < 0.4:  # If overlap < 40%, it's a distinct line missed by YOLO
                final_crops_with_y.append((py, pc))

        # Sort all crops by their Y coordinate to preserve perfect reading order!
        final_crops_with_y.sort(key=lambda x: x[0])
        final = [c for y, c in final_crops_with_y]

        print(f"DEBUG Segmenter: Final crops after filtering: {len(final)}")
        return final
