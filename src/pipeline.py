import os
import sys

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
    def __init__(self, yolo_weights="models/best.pt", crnn_weights="models/crnn_pashto.pth", vocab_path="models/vocab.json"):
        # Set Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading pipeline on {self.device}...")
        
        # 1. LOAD VOCAB
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary ({vocab_path}) missing.")
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            
        # Reverse map parsing JSON strings to integer keys
        self.id_to_char = {int(v): k for k, v in self.vocab.items()}
        self.num_classes = len(self.vocab)
        self.blank_id = self.vocab.get("<BLANK>", 0)
        
        # 2. LOAD CRNN
        self.crnn = PashtoCRNN(num_classes=self.num_classes).to(self.device)
        
        if os.path.exists(crnn_weights):
            self.crnn.load_state_dict(torch.load(crnn_weights, map_location=self.device))
            self.crnn.eval()
            print("Loaded PashtoCRNN Success.")
        else:
            print(f"Warning: {crnn_weights} not found. Ensure CRNN weights exist or run `train.py`.")
            
        # 3. LOAD YOLO
        # Using the custom trained best.pt weights
        self.segmenter = YOLOSegmenter(model_path=yolo_weights)
        print("Loaded YOLOv8 Segmenter Success.")

    def decode_predictions(self, logits):
        """
        Greedy CTC Decoder.
        """
        probs = torch.softmax(logits, dim=2)
        preds = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()
        
        decoded = []
        prev = -1
        for p in preds:
            if p != prev and p != self.blank_id:
                char = self.id_to_char.get(p, "")
                if char not in ["<PAD>", "<UNK>", "<BLANK>"]:
                    decoded.append(char)
            prev = p
            
        # --- THE RTL FIX ---
        # Reverse the decoded characters to reconstruct the Right-to-Left Pashto word properly
        decoded.reverse()
            
        # Join into UTF-8 Pashto String
        return "".join(decoded)

    def process_page(self, page_image_path, save_crops=True):
        """
        Runs the Full Pipeline: Page -> YOLO -> Preprocess -> CRNN -> Join strings
        """
        print(f"\nProcessing full page: {page_image_path}")
        
        # --- STAGE 1: Detect and sort lines ---
        line_crops = self.segmenter.segment_lines(page_image_path)
        
        if not line_crops:
            return "No lines detected."
            
        recognized_lines = []
        
        # Optional: save to data/processed to verify YOLO splits properly
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)
            
        # --- STAGE 2: Parse lines ---
        for idx, crop in enumerate(line_crops):
            if save_crops:
                crop.save(os.path.join(processed_dir, f"line_{idx:03d}.jpg"))
                
            # Preprocess crop (Grayscale -> Resize H=32 -> Color Invert -> H-Flip)
            tensor = preprocess_line_crop(crop).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.crnn(tensor)
                
            # Decode to Pashto text
            text = self.decode_predictions(logits)
            recognized_lines.append(text)
            
        # --- FINAL: Stitch the page structure together containing Unicode characters ---
        # Joins all lines vertically with standard newline breaks.
        full_document_text = "\n".join(recognized_lines)
        return full_document_text

if __name__ == "__main__":
    """
    Test script. Put a valid document in data/raw/ and run:
    python src/pipeline.py
    """
    # Initialize with our custom AI models
    stitcher = FullPagePashtoRecognition(
        yolo_weights="models/best.pt", 
        crnn_weights="models/crnn_pashto.pth",
        vocab_path="models/vocab.json"
    )
    
    test_img = "data/raw/test_page.jpg"
    
    if os.path.exists(test_img):
        text = stitcher.process_page(test_img, save_crops=True)
        print("\n===============================")
        print(" EXTRACTED PASHTO DOCUMENT")
        print("===============================\n")
        
        # bidi is used here so your terminal prints RTL perfectly (web browsers handle this natively later)
        try:
            from bidi.algorithm import get_display
            print(get_display(text))
        except ImportError:
            print(text)
            print("\n(Note: Install python-bidi for perfect terminal RTL rendering: pip install python-bidi)")
            
    else:
        print(f"\nPlease add a test document to {test_img} to run the stitcher demo.")