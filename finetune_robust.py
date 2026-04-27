import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os

# Import our custom dataset and model classes
from dataset import PashtoHandwritingDataset, collate_fn, build_vocab
from model import PashtoCRNN

def finetune_robust():
    """
    Fine-tuning script to improve robustness against mobile-photo noise (shadows, glare, tilt).
    """
    # ==========================================
    # 1. Setup & Paths
    # ==========================================
    
    # Use the new Augmented Mobile dataset
    dataset_dir = "PHTI-MOBILE-AUGMENTED"
    train_split = "Training_Mobile.txt"
    vocab_path = "models/vocab.json"
    
    # Ensure vocabulary exists, or build it
    if not os.path.exists(vocab_path):
        print("Vocabulary not found. Building from dataset...")
        # Note: We build from the original dataset dir to ensure we catch all possible characters
        build_vocab("Training.txt", "PHTI-DATASET", vocab_path)

    # We load the existing model and save to a new "Robust" version
    base_checkpoint = "models/crnn_pashto.pth"
    robust_checkpoint = "models/crnn_pashto_robust.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for Fine-tuning: {device}")
    
    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    num_classes = len(vocab)
    blank_idx = vocab.get("<BLANK>", 0)
    
    # Initialize Dataset and DataLoader (pointing to augmented data)
    print(f"Initializing Robustness Dataset from {dataset_dir}...")
    train_dataset = PashtoHandwritingDataset(
        split_file=train_split,
        dataset_dir=dataset_dir,
        vocab_path=vocab_path,
        target_height=32
    )
    
    print(f"Dataset Size: {len(train_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for stability on Windows CPU
    )
    
    print(f"Total Batches: {len(train_loader)}")
    
    # ==========================================
    # 2. Load Model & Prepare for Fine-Tuning
    # ==========================================
    
    model = PashtoCRNN(num_classes=num_classes)
    
    if os.path.exists(base_checkpoint):
        print(f"Loading base weights from {base_checkpoint}...")
        # Load weights but allow for flexibility (map_location handles CPU/GPU mismatches)
        model.load_state_dict(torch.load(base_checkpoint, map_location=device))
    else:
        print("WARNING: Base checkpoint not found. Starting from scratch!")
    
    model = model.to(device)
    
    # Use a LOWER learning rate for fine-tuning (standard is 1e-3, we use 1e-4)
    # This prevents the model from "forgetting" the base handwriting.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    
    # ==========================================
    # 3. Short Training Loop (5-10 Epochs)
    # ==========================================
    
    num_epochs = 5 
    print(f"\nStarting Robustness Fine-tuning for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            image_widths = batch["image_widths"]
            label_lengths = batch["label_lengths"]
            
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            
            # The model architecture has a final Conv layer with kernel=2, padding=0
            # which reduces the width from (W//4) to (W//4 - 1).
            input_lengths = (image_widths // 4) - 1
            
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_loss = running_loss / len(train_loader)
        print(f"==> Epoch [{epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}\n")
        
    # ==========================================
    # 4. Save the Robust Model
    # ==========================================
    torch.save(model.state_dict(), robust_checkpoint)
    print(f"Success! Robust model saved to: {robust_checkpoint}")
    print("You can now update your pipeline to use this model for mobile photos.")

if __name__ == "__main__":
    finetune_robust()
