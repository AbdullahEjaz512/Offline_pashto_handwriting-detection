import torch
import torchvision.transforms.functional as TF
from PIL import Image

def preprocess_line_crop(crop_image: Image.Image, target_height=32):
    """
    Utility preprocessor for Cropped Line Images from YOLOv8.
    Applies image normalization and transforms before inference.
    """
    # 1. Grayscale Conversion (Must be exactly 1 channel)
    img = crop_image.convert('L')
    
    # 2. Resize to Fixed Height (32px), Keep Aspect ratio
    orig_w, orig_h = img.size
    new_w = int(orig_w * (target_height / float(orig_h)))
    
    # Prevent 0-width error on strange crops
    new_w = max(1, new_w) 
    
    img = img.resize((new_w, target_height), Image.Resampling.BILINEAR)
    
    # 3. Convert PIL to PyTorch Tensor [1, H, W] (Float 0.0 - 1.0)
    tensor = TF.to_tensor(img)
    
    # 4. Color Inversion (Assuming black ink on white page)
    # This turns background to ~0 and text to ~1, helpful for CNN activations
    tensor = 1.0 - tensor
    
    # 5. Horizontal Flip (Right-to-Left Fix)
    # Flips the tensor along the Width dimension (dim=2) so that right-most
    # characters enter the sequence first for proper Pashto CTC decoding.
    tensor = torch.flip(tensor, dims=[2])
    
    # Add Batch Dimension [1, 1, 32, W] for inference
    tensor = tensor.unsqueeze(0)
    
    return tensor
