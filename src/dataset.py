import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

SPECIAL_TOKENS = {
    "<BLANK>": 0,
    "<PAD>": 1,
    "<UNK>": 2,
}

def _read_split(split_file: Path) -> List[str]:
    with split_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_vocab(
    training_split_path: str,
    dataset_dir: str,
    output_vocab_path: str = "vocab.json",
) -> Dict[str, int]:
    """Build vocabulary from Training.txt file list and save as JSON."""
    split_path = Path(training_split_path)
    dataset_path = Path(dataset_dir)
    output_path = Path(output_vocab_path)

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    unique_chars = set()
    missing_files = 0

    for txt_name in _read_split(split_path):
        txt_path = dataset_path / txt_name
        if not txt_path.exists():
            missing_files += 1
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        unique_chars.update(text)

    vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
    next_id = 3
    for ch in sorted(unique_chars):
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1

    output_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if missing_files > 0:
        print(f"Warning: {missing_files} split entries were missing in dataset directory.")
    print(f"Saved {len(vocab)} tokens to {output_path}")
    return vocab

class PashtoHandwritingDataset(Dataset):
    """PyTorch Dataset for Pashto handwritten line OCR (CRNN + CTC)."""

    def __init__(
        self,
        split_file: str,
        dataset_dir: str,
        vocab_path: str,
        transform=None,
        target_height: int = 32,
    ) -> None:
        self.split_file = Path(split_file)
        self.dataset_dir = Path(dataset_dir)
        self.vocab_path = Path(vocab_path)
        self.transform = transform
        self.target_height = target_height

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")

        self.vocab: Dict[str, int] = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        self.pad_id = self.vocab.get("<PAD>", 1)
        self.unk_id = self.vocab.get("<UNK>", 2)
        self.text_filenames = _read_split(self.split_file)

    def __len__(self) -> int:
        return len(self.text_filenames)

    def _resolve_image_path(self, base_name: str) -> Path:
        preferred = self.dataset_dir / f"{base_name}.jpg"
        if preferred.exists():
            return preferred
        fallback_extensions = [".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
        for ext in fallback_extensions:
            path = self.dataset_dir / f"{base_name}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Image file not found for sample: {base_name}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        txt_filename = self.text_filenames[idx]
        txt_path = self.dataset_dir / txt_filename
        if not txt_path.exists():
            raise FileNotFoundError(f"Text file missing: {txt_path}")

        base_name = Path(txt_filename).stem
        img_path = self._resolve_image_path(base_name)

        with Image.open(img_path) as image:
            image = image.convert("L")
            orig_w, orig_h = image.size
            new_w = max(1, int(round(orig_w * (self.target_height / float(orig_h)))))
            image = image.resize((new_w, self.target_height), Image.Resampling.BILINEAR)
            image_tensor = TF.to_tensor(image)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        text = txt_path.read_text(encoding="utf-8").strip()
        label_ids = [self.vocab.get(ch, self.unk_id) for ch in text]
        label_tensor = torch.tensor(label_ids, dtype=torch.long)

        return image_tensor, label_tensor

def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]], pad_id: int = 1) -> Dict[str, torch.Tensor]:
    """Pad variable-width images and variable-length labels for CTC training."""
    if len(batch) == 0:
        raise ValueError("Empty batch received by collate_fn")

    images, labels = zip(*batch)
    image_widths = torch.tensor([img.shape[2] for img in images], dtype=torch.long)
    label_lengths = torch.tensor([lbl.numel() for lbl in labels], dtype=torch.long)

    max_width = int(image_widths.max().item())
    max_label_len = int(label_lengths.max().item())

    padded_images = []
    padded_labels = []

    for img, lbl in zip(images, labels):
        pad_w = max_width - img.shape[2]
        padded_images.append(F.pad(img, (0, pad_w, 0, 0), value=0.0))

        pad_len = max_label_len - lbl.numel()
        padded_labels.append(F.pad(lbl, (0, pad_len), value=pad_id))

    padded_images_tensor = torch.stack(padded_images, dim=0)
    padded_labels_tensor = torch.stack(padded_labels, dim=0)

    return {
        "padded_images": padded_images_tensor,
        "padded_labels": padded_labels_tensor,
        "image_widths": image_widths,
        "label_lengths": label_lengths,
        "images": padded_images_tensor,
        "labels": padded_labels_tensor,
    }

def preprocess_line_crop(crop_image: Image.Image, target_height=32):
    """
    Utility preprocessor for Cropped Line Images from YOLOv8.
    Applies image normalization and transforms before inference.
    """
    img = crop_image.convert('L')
    orig_w, orig_h = img.size
    new_w = int(orig_w * (target_height / float(orig_h)))
    new_w = max(1, new_w) 
    img = img.resize((new_w, target_height), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = 1.0 - tensor
    tensor = torch.flip(tensor, dims=[2])
    tensor = tensor.unsqueeze(0)
    return tensor
