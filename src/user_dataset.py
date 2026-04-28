import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
import json
from pathlib import Path

class UserFinetuneDataset(Dataset):
    """Dataset for user-corrected line crops (Tab-separated labels.txt)."""
    def __init__(self, labels_file, data_dir, vocab_path, target_height=64):
        self.data_dir = Path(data_dir)
        self.labels = []
        self.target_height = target_height
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.unk_id = self.vocab.get("<UNK>", 2)

        if Path(labels_file).exists():
            with open(labels_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        self.labels.append(parts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_rel_path, text = self.labels[idx]
        img_path = self.data_dir / img_rel_path
        
        with Image.open(img_path) as image:
            image = image.convert("L")
            orig_w, orig_h = image.size
            new_w = max(1, int(round(orig_w * (self.target_height / float(orig_h)))))
            image = image.resize((new_w, self.target_height), Image.Resampling.BILINEAR)
            image_tensor = TF.to_tensor(image)

        label_ids = [self.vocab.get(ch, self.unk_id) for ch in text]
        label_tensor = torch.tensor(label_ids, dtype=torch.long)
        
        return image_tensor, label_tensor

def preprocess_line_crop(pil_img, target_h=64):
    """Helper for inference pre-processing consistency."""
    w, h = pil_img.size
    new_w = int(w * (target_h / h))
    new_w = max(128, min(new_w, 2048))
    pil_img = pil_img.resize((new_w, target_h), Image.Resampling.LANCZOS)
    return TF.to_tensor(pil_img).unsqueeze(0)
