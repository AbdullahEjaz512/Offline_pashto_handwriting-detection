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
        """Greedy CTC decoder with realistic character-level confidence scores."""
        probs = torch.softmax(logits, dim=2)
        best_path = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()
        probs_numpy = torch.max(probs, dim=2)[0].squeeze(0).cpu().numpy()

        char_list = []
        char_probs = []
        last_char = -1
        current_char_probs = []

        for char_id, prob in zip(best_path, probs_numpy):
            if char_id != self.blank_id:
                if char_id != last_char:
                    if current_char_probs:
                        char_probs.append(max(current_char_probs))
                    current_char_probs = [prob]
                    char_list.append(self.id_to_char.get(char_id, ""))
                else:
                    current_char_probs.append(prob)
            else:
                if current_char_probs:
                    char_probs.append(max(current_char_probs))
                    current_char_probs = []
            last_char = char_id

        if current_char_probs:
            char_probs.append(max(current_char_probs))

        text = "".join(char_list)
        
        if char_probs:
            score = float(np.mean(char_probs))
        else:
            score = float(probs_numpy.mean())
            
        return text, score

    # ------------------------------------------------------------------
    # Per-crop decoding with multiple preprocessing strategies
    # ------------------------------------------------------------------

    def deskew_crop(self, crop):
        """Straightens a line crop by detecting the text angle."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Find all 'ink' pixels
        coords = np.column_stack(np.where(gray < 127))
        if len(coords) < 10: return crop
        
        # Find the rotation angle of the best-fit box
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        
        # Only rotate if the tilt is significant (e.g. > 0.5 degrees)
        if abs(angle) < 0.5: return crop
        
        (h, w) = crop.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _decode_crop_with_score(self, crop):
        """Professional-grade multi-strategy preprocessing with Deskewing. Returns (text, score)."""
        if crop is None or crop.size == 0: return "", 0.0
        
        # 1. First, straighten the line (Deskew)
        crop = self.deskew_crop(crop)
        ch, cw = crop.shape[:2]
        
        # 2. Extract channels for better contrast on colored paper
        b, g, r = cv2.split(crop)
        gray_std = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        variants = []
        # Strategy A: Standard Gray + Dilation-Normalization
        k = max(3, min(ch // 3, 15)) | 1
        bg = cv2.dilate(gray_std, np.ones((k, k), np.uint8))
        norm_gray = cv2.divide(gray_std, bg, scale=255)
        variants.append(norm_gray)
        
        # Strategy B: Green Channel + Dilation-Normalization (Best for Pink Paper)
        bg_g = cv2.dilate(g, np.ones((k, k), np.uint8))
        norm_green = cv2.divide(g, bg_g, scale=255)
        variants.append(norm_green)

        # Strategy C: Sharpened Version
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(norm_gray, -1, sharpen_kernel)
        variants.append(sharpened)

        best_text = ""
        max_score = 0.0

        for v_img in variants:
            pil_img = Image.fromarray(v_img).convert('L')
            tensor = preprocess_line_crop(pil_img).to(self.device)
            with torch.no_grad():
                logits = self.crnn(tensor)
            text, score = self.decode_predictions(logits)
            text = text.strip()
            
            if score > max_score:
                max_score = score
                best_text = text
        
        return best_text, max_score

    def decode_crop_with_variants(self, crop):
        text, _ = self._decode_crop_with_score(crop)
        return text

    def _run_hf_transformer_fallback(self, crops):
        """Runs the offline Hugging Face TrOCR model on a list of image crops."""
        model_dir = os.path.join("models", "hf_pashto_trocr")
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            model_dir = "osamajan90/pashto-trocr-handwritten"
            
        try:
            from transformers import VisionEncoderDecoderModel, TrOCRProcessor
            from PIL import Image
            import torch
            
            print(f"Executing TrOCR fallback using model: {model_dir}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            processor = TrOCRProcessor.from_pretrained(model_dir)
            model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
            
            results = []
            for crop in crops:
                if crop is None or crop.size == 0:
                    continue
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
                
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values)
                
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                if generated_text.strip():
                    results.append(generated_text.strip())
            
            fallback_confidence = 75 if results else 0
            return results, fallback_confidence
            
        except Exception as e:
            print(f"HF Fallback Error: {e}")
            return [], 0

    def process_page_with_confidence(self, image_path):
        """Full pipeline: segment -> decode -> combine. Returns (results, confidence)."""
        if not os.path.exists(image_path):
            return ["Error: Image file not found."], 0

        image = cv2.imread(image_path)
        if image is None:
            return ["Error: Could not load image."], 0

        crops = self.segmenter.segment_lines(image_path)
        
        # Scenario A: Page detects NOTHING
        if not crops:
            print("No lines detected by segmenter. Passing full page to HuggingFace fallback...")
            return self._run_hf_transformer_fallback([image])
            
        results = []
        scores = []
        for crop in crops:
            text, score = self._decode_crop_with_score(crop)
            if text:
                results.append(text)
                scores.append(score)
        
        avg_score = float(np.mean(scores)) if scores else 0.0
        confidence = int(avg_score * 100)
        
        # Scenario B: Confidence score is less than 65%
        if confidence < 65:
            print(f"CRNN Confidence is low ({confidence}%). Executing HuggingFace fallback...")
            hf_results, hf_confidence = self._run_hf_transformer_fallback(crops)
            if hf_results:
                return hf_results, hf_confidence
                
        return results, confidence

    def process_page(self, image_path):
        """Full pipeline: segment -> decode -> combine."""
        if not os.path.exists(image_path):
            return ["Error: Image file not found."]

        image = cv2.imread(image_path)
        if image is None:
            return ["Error: Could not load image."]

        # Stage 1: Segmentation
        # Passing the path directly as expected by segmenter.py
        crops = self.segmenter.segment_lines(image_path)
        
        # Stage 2: Recognition
        results = []
        for crop in crops:
            text = self.decode_crop_with_variants(crop)
            if text:
                results.append(text)
        
        return results