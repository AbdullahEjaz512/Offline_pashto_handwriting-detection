import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path

# Import our custom dataset and model classes
from src.user_dataset import UserFinetuneDataset
from src.dataset import collate_fn
from src.model import PashtoCRNN

def train_on_user_data(epochs=10, batch_size=8, lr=1e-4):
    """
    The 'Final Victory' Training Loop.
    Learns specifically from the user's corrected edits.
    """
    user_data_dir = "USER_TRAINING_DATA"
    labels_file = os.path.join(user_data_dir, "labels.txt")
    vocab_path = "models/vocab.json"
    base_checkpoint = "models/crnn_pashto.pth"
    # We save to a 'user' specific model to avoid corrupting the base one immediately
    user_checkpoint = "models/crnn_pashto_user.pth"
    
    if not os.path.exists(labels_file):
        print(f"Error: No user training data found at {labels_file}. Please save some edits first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for User Fine-tuning: {device}")
    
    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    num_classes = len(vocab)
    blank_idx = vocab.get("<BLANK>", 0)
    
    # Initialize Dataset
    dataset = UserFinetuneDataset(
        labels_file=labels_file,
        data_dir=user_data_dir,
        vocab_path=vocab_path,
        target_height=64
    )
    
    if len(dataset) == 0:
        print("Dataset is empty. Add some corrected lines via the UI.")
        return
        
    print(f"User Dataset Size: {len(dataset)} samples")
    
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize Model
    model = PashtoCRNN(num_classes=num_classes)
    
    # Load existing best weights
    load_path = user_checkpoint if os.path.exists(user_checkpoint) else base_checkpoint
    if os.path.exists(load_path):
        print(f"Loading weights from {load_path}...")
        model.load_state_dict(torch.load(load_path, map_location=device))
    
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            image_widths = batch["image_widths"]
            label_lengths = batch["label_lengths"]
            
            outputs = model(images)
            # CRNN output is [W, B, C], CTC needs log_probs
            log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
            
            # Input lengths for CTC (based on model pooling)
            input_lengths = (image_widths // 4) - 1
            
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")
        
    # Save the new "User Optimized" model
    torch.save(model.state_dict(), user_checkpoint)
    print(f"SUCCESS: User-specific model saved to {user_checkpoint}")
    
    # Update the symbolic link or the primary model path if desired
    # For now, we'll keep it as _user.pth
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    
    train_on_user_data(epochs=args.epochs)
