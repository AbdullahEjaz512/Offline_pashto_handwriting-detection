import os
import json
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from model import PashtoCRNN

def preprocess_image(image_path, target_height=32):
    """
    Loads an image, converts to grayscale, resizes height to target_height 
    while preserving aspect ratio, and normalizes to a PyTorch tensor.
    """
    # 1. Load and convert rigidly to grayscale
    image = Image.open(image_path).convert('L')
    
    # 2. Maintain aspect ratio for resize
    orig_w, orig_h = image.size
    new_w = int(orig_w * (target_height / float(orig_h)))
    new_w = max(1, new_w) # Prevent 0 width
    
    image = image.resize((new_w, target_height), Image.Resampling.BILINEAR)
    
    # 3. Convert PIL image to PyTorch Tensor [1, H, W] (Values from 0.0 to 1.0)
    image_tensor = TF.to_tensor(image)
    
    # 4. Add Batch dimension -> [1, 1, 32, W]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def decode_predictions(predictions, id_to_char, blank_id=0):
    """
    Applies CTC Greedy Decoding to the predictions.
    - predictions: Tensor of shape [Batch, Sequence_Length, Num_Classes]
    - id_to_char: Dictionary mapping integer IDs back to Pashto strings
    """
    # predictions is shape [1, Seq_Len, Num_Classes]
    # 1. Apply softmax (optional for argmax, but theoretically correct)
    probs = torch.softmax(predictions, dim=2)
    
    # 2. Get the index of the highest probability at each time step (Greedy)
    preds = torch.argmax(probs, dim=2)
    
    # Remove batch dimension and convert to numpy array for iteration
    preds = preds.squeeze(0).cpu().numpy()
    
    decoded_sequence = []
    prev_id = -1
    
    # 3. CTC Decoding Logic: Ignore consecutive duplicates and blanks
    for current_id in preds:
        if current_id != prev_id and current_id != blank_id:
            char = id_to_char.get(current_id, "")
            
            # Avoid visually inserting <PAD> or <UNK> tags into the final text
            if char not in ["<PAD>", "<UNK>", "<BLANK>"]:
                decoded_sequence.append(char)
                
        prev_id = current_id
        
    # 4. Join the extracted list of characters into a single Pashto string
    return "".join(decoded_sequence)


def predict(image_path, model_path="crnn_pashto.pth", vocab_path="vocab.json"):
    """
    Runs end-to-end inference on a given handwriting image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==========================================
    # 1. Setup & Loading
    # ==========================================
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary dictionary '{vocab_path}' not found.")
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    # Create Reverse Mapping ({0: "<BLANK>", 3: "ا", ...})
    # Remember JSON keys are loaded as strings, so we convert them to ints
    id_to_char = {int(v): k for k, v in vocab.items()}
    num_classes = len(vocab)
    blank_id = vocab.get("<BLANK>", 0)
    
    # Initialize Model & Load Weights
    model = PashtoCRNN(num_classes=num_classes)
    
    if os.path.exists(model_path):
        # Load state dictionary explicitly mapping to configured device
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from '{model_path}'.")
    else:
        print(f"Warning: Model weights '{model_path}' not found! Using untrained weights.")
        
    model = model.to(device)
    model.eval() # CRITICAL: Sets Dropout and BatchNorm to evaluation mode
    
    # ==========================================
    # 2. Execute Pipeline
    # ==========================================
    # Preprocess
    image_tensor = preprocess_image(image_path).to(device)
    
    # Forward Pass
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # Decode Output
    predicted_text = decode_predictions(outputs, id_to_char, blank_id=blank_id)
    
    return predicted_text


if __name__ == "__main__":
    # Feel free to change this path to an actual item in your dataset
    # e.g., validation/test image path
    test_image_path = "PHTI-DATASET/PHTI-A1-F-0044-10.jpg"
    
    print("========================================")
    print(" Pashto Handwriting Inference Script")
    print("========================================\n")
    
    if os.path.exists(test_image_path):
        print(f"Processing Image: {test_image_path}")
        
        result = predict(test_image_path)
        
        print("\n----------------------------------------")
        print(f"Predicted Text: {result}")
        print("----------------------------------------")
    else:
        print(f"Error: Could not find image '{test_image_path}'.")
        print("Please point 'test_image_path' to a valid .jpg file.")
