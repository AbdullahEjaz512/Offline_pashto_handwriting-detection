import cv2
import numpy as np
import os
import sys
from PIL import Image  

# Add the project root to the Python path so 'src' can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import json
import torch

from src.model import PashtoCRNN
from src.segmenter import YOLOSegmenter
from src.dataset import preprocess_line_crop


class FullPagePashtoRecognition:
    """
    Two-Stage AI Pipeline:
    Stage 1: YOLOv8 detects lines.
    Stage 2: CRNN reads cropped lines sequence-by-sequence.
    """
    def __init__(self, yolo_weights="models/best.pt",
                 crnn_weights="models/crnn_pashto.pth",
                 vocab_path="models/vocab.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pipeline on {self.device}...")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary ({vocab_path}) missing.")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.id_to_char = {int(v): k for k, v in self.vocab.items()}
        self.num_classes = len(self.vocab)
        self.blank_id = self.vocab.get("<BLANK>", 0)

        self.crnn = PashtoCRNN(num_classes=self.num_classes).to(self.device)
        if os.path.exists(crnn_weights):
            self.crnn.load_state_dict(
                torch.load(crnn_weights, map_location=self.device))
            self.crnn.eval()
            print("Loaded PashtoCRNN Success.")
        else:
            print(f"Warning: {crnn_weights} not found.")

        self.segmenter = YOLOSegmenter(model_path=yolo_weights)
        print("Loaded YOLOv8 Segmenter Success.")

    # ------------------------------------------------------------------
    # CTC decoding
    # ------------------------------------------------------------------

    def decode_predictions(self, logits):
        """
        Greedy CTC decoder.
        Returns (decoded_string, mean_character_confidence).
        """
        probs = torch.softmax(logits, dim=2)
        max_probs, preds = torch.max(probs, dim=2)
        preds     = preds.squeeze(0).cpu().numpy()
        max_probs = max_probs.squeeze(0).cpu().numpy()

        decoded, confidences = [], []
        prev = -1
        for p, prob in zip(preds, max_probs):
            if p != prev and p != self.blank_id:
                char = self.id_to_char.get(p, "")
                if char not in ["<PAD>", "<UNK>", "<BLANK>"]:
                    decoded.append(char)
                    confidences.append(float(prob))
            prev = p

        text  = "".join(decoded)
        score = float(np.mean(confidences)) if confidences else 0.0
        return text, score

    # ------------------------------------------------------------------
    # Per-crop decoding with multiple preprocessing variants
    # ------------------------------------------------------------------

    def decode_crop_with_variants(self, crop):
        """
        Decode one line crop via multiple preprocessing variants.
        The variant with the highest CTC confidence score wins.

        Variants cover:
          1. Raw grayscale        — clean paper, good light
          2. CLAHE                — faint ink, low contrast
          3. Adaptive threshold   — high contrast black-on-white
          4. Shadow removal       — uneven lighting / mobile photo shadows
        """
        if crop is None or crop.size == 0:
            return ""

        ch, cw = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # ---- Variant 1: raw grayscale ----
        v1 = gray

        # ---- Variant 2: CLAHE (local contrast boost) ----
        tile  = max(1, min(8, ch // 4))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(tile, tile))
        v2    = clahe.apply(gray)

        # ---- Variant 3: adaptive threshold (clean binary) ----
        block = max(11, (ch // 2) | 1)
        v3    = cv2.adaptiveThreshold(
            v2, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, 8,
        )

        # ---- Variant 4: shadow removal (mobile photo / uneven lighting) ----
        # Kernel scaled to crop size so it bridges over the widest stroke
        # but never larger than the crop itself.
        k = max(3, min(ch // 3, 21)) | 1     # always odd
        dilated = cv2.dilate(gray, np.ones((k, k), np.uint8))
        bg      = cv2.medianBlur(dilated, k)
        diff    = 255 - cv2.absdiff(gray, bg)
        v4      = cv2.normalize(diff, None, 0, 255,
                                cv2.NORM_MINMAX, cv2.CV_8UC1)

        candidates = []
        for v in [v1, v2, v3, v4]:
            pil_img = Image.fromarray(v).convert('L')
            tensor  = preprocess_line_crop(pil_img).to(self.device)
            with torch.no_grad():
                logits = self.crnn(tensor)
            text, score = self.decode_predictions(logits)
            text = text.strip()
            if text:
                if len(text) < 3:
                    score *= 0.5    # penalise likely-noise outputs
                candidates.append((text, score))

        if not candidates:
            return ""

        best_text, _ = max(candidates, key=lambda x: x[1])
        return best_text

    # ------------------------------------------------------------------
    # Full-page pipeline
    # ------------------------------------------------------------------

    def process_page(self, page_image_path, save_crops=True):
        """
        Full pipeline: Image → YOLO/Fallback → CRNN → joined text
        """
        print(f"\nProcessing full page: {page_image_path}")

        line_crops = self.segmenter.segment_lines(page_image_path)
        print(f"Detected line crops: {len(line_crops)}")

        if not line_crops:
            return "No lines detected."

        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)

        recognized_lines = []
        for idx, crop in enumerate(line_crops):
            if save_crops:
                cv2.imwrite(
                    os.path.join(processed_dir, f"line_{idx:03d}.jpg"), crop)

            text = self.decode_crop_with_variants(crop)
            if text:
                recognized_lines.append(text)

        return "\n".join(recognized_lines)


if __name__ == "__main__":
    pipeline = FullPagePashtoRecognition(
        yolo_weights="models/best.pt",
        crnn_weights="models/crnn_pashto.pth",
        vocab_path="models/vocab.json",
    )
    test_img = "data/raw/test_page.jpg"
    if os.path.exists(test_img):
        text = pipeline.process_page(test_img, save_crops=True)
        print("\n==============================")
        print(" EXTRACTED PASHTO DOCUMENT")
        print("==============================\n")
        try:
            from bidi.algorithm import get_display
            print(get_display(text))
        except ImportError:
            print(text)
    else:
        print(f"\nAdd a test document to {test_img} to run demo.")