import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os

# Import our custom dataset and model classes
from src.dataset import PashtoHandwritingDataset, collate_fn, build_vocab
from src.model import PashtoCRNN

def train_crnn():
    """
    Main training script for the Pashto CRNN Handwriting Detection Model.
    """
    # ==========================================
    # 1. Setup & Initialization
    # ==========================================
    
    # Define paths
    dataset_dir = "PHTI-DATASET"
    train_split = "Training.txt"
    vocab_path = "vocab.json"
    checkpoint_path = "crnn_pashto.pth"
    
    # Device configuration (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure vocabulary exists, or build it
    if not os.path.exists(vocab_path):
        print("Vocabulary not found. Building...")
        build_vocab(train_split, dataset_dir, vocab_path)
        
    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    num_classes = len(vocab)
    blank_idx = vocab.get("<BLANK>", 0) # Reserve 0 for blank token
    
    # Initialize Dataset and DataLoader
    print("Initializing Training Dataset...")
    train_dataset = PashtoHandwritingDataset(
        split_file=train_split,
        dataset_dir=dataset_dir,
        vocab_path=vocab_path,
        target_height=32  # Enforced by architecture
    )
    
    # Batch size is typically 32, 64, or 128 depending on vRAM.
    batch_size = 32
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn, # Custom function to handle variable widths
        num_workers=4,         # Adjust based on CPU cores
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize Model
    print(f"Initializing PashtoCRNN with {num_classes} classes...")
    model = PashtoCRNN(num_classes=num_classes)
    model = model.to(device)
    
    # ==========================================
    # 2. Optimizer and Loss Function
    # ==========================================
    
    # AdamW is generally more stable than standard Adam
    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # CTCLoss Configuration
    # blank: The index of the blank label
    # zero_infinity=True is CRITICAL: it prevents the loss from becoming infinite or NaN 
    # if the target sequence is too long for the predicted sequence in early, random-weight epochs.
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    
    # ==========================================
    # 3. The Training Loop
    # ==========================================
    
    num_epochs = 10
    print("\nStarting Training Loop...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move features to device
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            
            # These must be Int or Long tensors on CPU for CTCLoss
            image_widths = batch["image_widths"]
            label_lengths = batch["label_lengths"]
            
            # --- Forward Pass ---
            # Output from model implies [Batch, Sequence_Length (T), Num_Classes (C)]
            outputs = model(images)
            
            # Format outputs for CTC Loss:
            # 1. CTCLoss expects log probabilities
            log_probs = F.log_softmax(outputs, dim=2)
            
            # 2. CTCLoss expects dimensions [Sequence_Length (T), Batch (N), Num_Classes (C)]
            log_probs = log_probs.permute(1, 0, 2)
            
            # --- Format Lengths for CTC Loss ---
            # IMPORTANT: The CNN feature extractor pools the width by a factor of 4.
            # (two MaxPool2d layers with stride=2 in the width dimension)
            # We MUST scale down the original image widths to match the spatial sequence length.
            input_lengths = image_widths // 4
            
            # --- Calculate Loss ---
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (CRITICAL for RNNs)
            # Prevents the "exploding gradient" problem common in LSTMs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print batch progress occasionally
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        # ==========================================
        # 4. Logging
        # ==========================================
        avg_loss = running_loss / len(train_loader)
        print(f"\n==> Epoch [{epoch+1}/{num_epochs}] Completed - Average Loss: {avg_loss:.4f}\n")
        
    # ==========================================
    # 5. Saving
    # ==========================================
    print("Training finished.")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model weights saved successfully to: {checkpoint_path}")

if __name__ == "__main__":
    train_crnn()
