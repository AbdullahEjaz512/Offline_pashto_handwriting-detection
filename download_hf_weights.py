import os
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

def download():
    print("Starting TrOCR download for offline Pashto...")
    model_id = "osamajan90/pashto-trocr-handwritten"
    save_path = "models/hf_pashto_trocr"
    
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Downloading processor from {model_id}...")
    processor = TrOCRProcessor.from_pretrained(model_id)
    processor.save_pretrained(save_path)
    
    print(f"Downloading weights from {model_id}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.save_pretrained(save_path)
    print(f"Success! Model weights saved offline at {save_path}")

if __name__ == "__main__":
    download()
