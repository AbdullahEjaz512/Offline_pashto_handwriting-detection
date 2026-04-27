import cv2
import numpy as np
from ultralytics import YOLO


class YOLOSegmenter:
    """
    Line segmentation for Pashto handwriting on white paper (blank or lined).
    Primary: YOLOv8. Fallback: horizontal ink-projection profile.
    """

    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _binarize(self, img):
        """
        Universal Binarization (Professional Grade).
        Handles: Pink paper, shadows, glare, and low contrast.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Estimate background using a morphological 'closing' equivalent
        # We use a kernel size relative to the expected line height
        k_size = max(11, h // 40) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        
        # Dilation picks the brightest pixels (paper), effectively 'erasing' the ink
        bg = cv2.dilate(gray, kernel)
        bg = cv2.medianBlur(bg, k_size)
        
        # 2. Divide the original by the background to get a pure white-background image
        # This removes paper color and shadows perfectly
        norm = cv2.divide(gray, bg, scale=255)
        
        # 3. Final Binarization
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _remove_ruled_lines(self, binary, w):
        """
        Erase horizontal ruled lines from a binary image.
        A ruled line is a very long (>w/5), very thin (height=1) horizontal run.
        Removing them ensures the projection profile has clean valleys between
        text lines on lined paper.
        """
        min_len = max(30, w // 5)
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        ruled = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_h)
        return cv2.subtract(binary, ruled)

    def _ink_ratio(self, crop):
        """Ink-pixel fraction for a BGR crop (quality metric)."""
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.GaussianBlur(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        r = float((bw > 0).mean())
        if r > 0.5 or r < 0.001:
            avg = gray.mean()
            _, bw = cv2.threshold(
                gray, max(0, avg - 20), 255, cv2.THRESH_BINARY_INV)
            r = float((bw > 0).mean())
        return r

    def _score_crops(self, crops):
        """(n_valid_crops, avg_ink_density). Text ink is in [0.3%, 50%]."""
        if not crops:
            return (0, 0.0)
        ratios = [self._ink_ratio(c)
                  for c in crops if c is not None and c.size > 0]
        likely = [r for r in ratios if 0.003 <= r <= 0.50]
        avg = (sum(likely) / len(likely)) if likely else 0.0
        return (len(likely), avg)

    # ------------------------------------------------------------------
    # Fallback: horizontal projection profile
    # ------------------------------------------------------------------

    def _projection_line_crops(self, img):
        """
        Find text lines using horizontal ink-projection profile.

        Algorithm:
          1. Binarise (Otsu).
          2. Strip ruled lines so their constant ink doesn't fill the valleys.
          3. Sum ink pixels per row → 1D profile.
          4. Find rows where ink ≈ 0 (gaps between text lines).
          5. Return each text band as a cropped image.

        This is robust for any number of lines on white paper.
        """
        h, w = img.shape[:2]

        binary = self._binarize(img)
        binary = self._remove_ruled_lines(binary, w)

        # Remove tiny specks (noise)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        # Row-wise ink sum
        profile = binary.sum(axis=1).astype(np.float32)

        # Smooth gently — kernel must be small relative to inter-line gap
        # so narrow gaps between adjacent lines aren't filled in
        smooth_k = max(3, h // 120) | 1
        profile = cv2.GaussianBlur(
            profile.reshape(-1, 1), (1, smooth_k), 0).flatten()

        # A row counts as "text" if it has >1% of the peak ink density
        threshold = max(1.0, float(profile.max()) * 0.01)
        in_text = profile > threshold

        # Find contiguous text bands
        bands = []
        start = None
        for i, flag in enumerate(in_text):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                bands.append([start, i])
                start = None
        if start is not None:
            bands.append([start, h])

        # Merge bands separated by < 2px (single interrupted stroke)
        merged = []
        for b in bands:
            if merged and b[0] - merged[-1][1] < 2:
                merged[-1][1] = b[1]
            else:
                merged.append(b)

        # Build crops
        min_band_h = max(8, h // 55)
        pad = 4
        crops = []
        for y1, y2 in merged:
            if (y2 - y1) < min_band_h:
                continue                       # too thin → noise / ruled line
            y1c = max(0, y1 - pad)
            y2c = min(h, y2 + pad)
            # Trim empty left/right margins using column projection
            col_proj = binary[y1:y2, :].sum(axis=0)
            ink_cols = np.where(col_proj > 0)[0]
            if ink_cols.size == 0:
                continue
            x1 = max(0, int(ink_cols[0]) - 8)
            x2 = min(w, int(ink_cols[-1]) + 8)
            crops.append(img[y1c:y2c, x1:x2])

        return crops

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_lines(self, image_path):
        """
        Detect and return text line crops (BGR, top-to-bottom).

        Strategy:
          1. Run YOLO on the original image.
          2. Run projection-profile fallback.
          3. Pick whichever found more valid text crops.
             Tie-break on average ink density per crop.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot open: {image_path}")

        h, w = img.shape[:2]

        # --- YOLO ---
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(rgb, conf=0.08, iou=0.40, max_det=100)

        yolo_crops = []
        for result in results:
            for box in sorted(result.boxes, key=lambda b: b.xyxy[0][1]):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                bw, bh = x2 - x1, y2 - y1
                if bw < 15 or bh < 6:
                    continue
                # Reject full-width boxes at exact image edges
                if bw >= int(0.97 * w) and (y1 == 0 or y2 == h):
                    continue
                yolo_crops.append(img[y1:y2, x1:x2])

        # --- Projection-profile fallback ---
        proj_crops = self._projection_line_crops(img)

        yolo_score = self._score_crops(yolo_crops)
        proj_score = self._score_crops(proj_crops)

        print(f"  [SEG] YOLO={len(yolo_crops)} score={yolo_score} | "
              f"proj={len(proj_crops)} score={proj_score}")

        # Pick the better result
        if proj_score[0] > yolo_score[0]:
            return proj_crops
        if proj_score[0] == yolo_score[0] and proj_score[1] > yolo_score[1]:
            return proj_crops
        return yolo_crops if yolo_crops else proj_crops
