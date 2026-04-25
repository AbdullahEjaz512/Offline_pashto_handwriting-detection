import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# ==========================================
# Task 1: Vocabulary Extraction & Mapping
# ==========================================
def build_vocab(training_split_path, dataset_dir, output_vocab_path="vocab.json"):
    """
    Reads the text files listed in the training split, extracts unique Pashto 
    characters, and generates a mapping to integer IDs.
    """
    # Reserved IDs required for CTC Loss and padding
    vocab = {
        "<BLANK>": 0, # CTC Blank token (required by PyTorch CTCLoss)
        "<PAD>": 1,   # Padding token for sequence batching
        "<UNK>": 2    # Unknown character token
    }
    
    unique_chars = set()
    
    with open(training_split_path, 'r', encoding='utf-8') as f:
        # Read filenames and ignore empty lines
        filenames = [line.strip() for line in f if line.strip()]
        
    for filename in filenames:
        file_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as text_file:
            text = text_file.read().strip()
            # Add every character inside the string to the set
            for char in text:
                unique_chars.add(char)
                
    # Sort for deterministic ID mapping across different runs
    sorted_chars = sorted(list(unique_chars))
    
    # Assign integer IDs starting from 3
    current_id = 3
    for char in sorted_chars:
        if char not in vocab:
            vocab[char] = current_id
            current_id += 1
            
    # Save the dictionary as JSON (ensure_ascii=False to support Pashto Unicode)
    with open(output_vocab_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)
        
    print(f"Vocabulary built with {len(vocab)} tokens and saved to '{output_vocab_path}'.")
    return vocab

# ==========================================
# Task 2: PyTorch Dataset Class
# ==========================================
class PashtoHandwritingDataset(Dataset):
    def __init__(self, split_file, dataset_dir, vocab_path, transform=None, target_height=32):
        """
        Custom Dataset for Pashto offline handwriting detection.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_height = target_height
        
        # Load the pre-built vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.unk_id = self.vocab["<UNK>"]
            
        # Load sample filenames from the split file (e.g. Training.txt)
        with open(split_file, 'r', encoding='utf-8') as f:
            self.text_filenames = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.text_filenames)

    def __getitem__(self, idx):
        txt_filename = self.text_filenames[idx]
        
        # Image base name matched with .jpg
        base_name = os.path.splitext(txt_filename)[0]
        img_filename = f"{base_name}.jpg" 
        
        txt_path = os.path.join(self.dataset_dir, txt_filename)
        img_path = os.path.join(self.dataset_dir, img_filename)
        
        # --- Handle Image ---
        # Convert rigidly to grayscale ('L' mode)
        image = Image.open(img_path).convert('L')
        
        # Resize while keeping aspect ratio (fixed height, varying width)
        orig_w, orig_h = image.size
        new_w = int(orig_w * (self.target_height / float(orig_h)))
        
        # Prevent zero-width on extremely small images
        new_w = max(1, new_w) 
        
        image = image.resize((new_w, self.target_height), Image.Resampling.BILINEAR)
        
        # Convert PIL Image to PyTorch Tensor [1, H, W] (scales values to 0.0 - 1.0)
        image_tensor = TF.to_tensor(image)
        
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
            
        # --- Handle Text Label ---
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
            
        # Map string to list of IDs, falling back to <UNK> if a character is missing in vocab
        label_ids = [self.vocab.get(char, self.unk_id) for char in text]
        label_tensor = torch.tensor(label_ids, dtype=torch.long)
        
        return image_tensor, label_tensor

# ==========================================
# Task 3: Custom Collate Function
# ==========================================
def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Pads varying-width images and varying-length sequences.
    """
    images, labels = zip(*batch)
    
    # Extract original shapes sizes before padding
    # Image tensor shape: [Channels=1, Height=32, Width=W]
    image_widths = [img.shape[2] for img in images]
    label_lengths = [len(lbl) for lbl in labels]
    
    max_width = max(image_widths)
    max_label_len = max(label_lengths)
    
    padded_images = []
    padded_labels = []
    
    pad_idx = 1 # "<PAD>" ID as defined in vocab
    
    for img, lbl in zip(images, labels):
        # 1. Pad Image
        # F.pad format for 2D inputs is (pad_left, pad_right, pad_top, pad_bottom)
        pad_width = max_width - img.shape[2]
        
        # Using 0 (black). If your background is white, change value=0 to value=1.0 
        # (assuming TF.to_tensor scales 255 -> 1.0)
        img_padded = F.pad(img, (0, pad_width, 0, 0), value=0.0)
        padded_images.append(img_padded)
        
        # 2. Pad LabelSequence
        # Format for 1D input is (pad_left, pad_right)
        pad_len = max_label_len - len(lbl)
        lbl_padded = F.pad(lbl, (0, pad_len), value=pad_idx)
        padded_labels.append(lbl_padded)
        
    # Stack items together forming batched tensors
    padded_images = torch.stack(padded_images)      # [Batch, 1, 32, max_W]
    padded_labels = torch.stack(padded_labels)      # [Batch, max_label_len]
    
    # CTCLoss strictly requires original lengths passed as Int or Long tensors
    image_widths = torch.tensor(image_widths, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    
    return {
        "images": padded_images,
        "labels": padded_labels,
        "image_widths": image_widths,
        "label_lengths": label_lengths
    }

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    train_txt = "PHTI/Training.txt"
    dataset_folder = "PHTI/PHTI-DATASET"
    vocab_file = "vocab.json"
    
    # 1. Build and save vocabulary 
    print("Building vocabulary...")
    build_vocab(train_txt, dataset_folder, vocab_file)
    
    # 2. Instantiate Dataset
    print("Initializing dataset...")
    train_dataset = PashtoHandwritingDataset(
        split_file=train_txt, 
        dataset_dir=dataset_folder, 
        vocab_path=vocab_file
    )
    print(f"Dataset length: {len(train_dataset)}")
    
    # 3. Create DataLoader using the custom collate_fn
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    print("Dataset module ready.")
