"""
TrOCR Auxiliary OCR Fallback
============================
Provides a multi-layered safety net using Microsoft's TrOCR Vision model.
"""

import os
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class FlorenceFallback:  # Keep name matching the pipeline's expected interface
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "microsoft/trocr-base-handwritten"
        
        print(f"DEBUG TrOCR: Initializing model ({self.model_id}) on {self.device}...")
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_id)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(self.device)
            print("DEBUG TrOCR: Load successful.")
        except Exception as e:
            print(f"WARN TrOCR: Could not download/load model: {e}")
            self.model = None
            self.processor = None

    def extract_text(self, image_path):
        """Extracts OCR text from the line crop or raw page image."""
        if not self.model or not self.processor:
            return ""

        try:
            image = Image.open(image_path).convert("RGB")
            
            # TrOCR expects single-line cropped images. Passing a full page yields generic guesses.
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text.strip()
        except Exception as e:
            print(f"ERROR TrOCR: Execution failed: {e}")
            return ""
