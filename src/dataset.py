import json
import cv2
import numpy as np
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


def _resize_to_tensor(arr_gray, target_h=32):
    """Resize a grayscale array to target_h, keeping aspect ratio. Returns [1, H, W] tensor."""
    h, w = arr_gray.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = target_h / h
    new_w = max(32, min(int(w * scale), 2048))
    resized = cv2.resize(arr_gray, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return TF.to_tensor(Image.fromarray(resized))


def preprocess_adaptive(img_bgr, target_h=32):
    """
    Preprocessing A: Adaptive threshold (51, 20).
    Benchmark winner for single-line images (score=0.907).
    Best for: Mobile photos with colored/faint backgrounds.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 20
    )
    binary = cv2.medianBlur(binary, 3)
    return _resize_to_tensor(binary, target_h)


def preprocess_morph(img_bgr, target_h=32):
    """
    Preprocessing B: Adaptive threshold (31,12) + morphological closing.
    Benchmark winner for multi-line images (score=0.799, 0.536).
    Best for: Pages where cursive joins are broken by thresholding.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 12
    )
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return _resize_to_tensor(binary, target_h)


def robust_preprocess(img_np, target_h=32):
    """
    Main entry: runs both preprocessing variants and returns the tensor.
    This is called by the Dataset class during training.
    Uses adaptive_51_20 (benchmark-validated best overall).
    """
    if isinstance(img_np, Image.Image):
        img_np = np.array(img_np)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    t = preprocess_adaptive(img_np, target_h)
    return t if t is not None else torch.zeros(1, target_h, 32)


def preprocess_line_crop(crop_image, target_height=32):
    """Wrapper for pipeline.py — adds batch dimension."""
    if isinstance(crop_image, Image.Image):
        img_np = np.array(crop_image)
    else:
        img_np = crop_image
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    t = robust_preprocess(img_np, target_h=target_height)
    return t.unsqueeze(0)


# ── Dataset Utilities (unchanged) ─────────────────────────────────────────────

def _read_split(split_file: Path) -> List[str]:
    with split_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_vocab(
    training_split_path: str,
    dataset_dir: str,
    output_vocab_path: str = "vocab.json",
) -> Dict[str, int]:
    split_path = Path(training_split_path)
    dataset_path = Path(dataset_dir)
    output_path = Path(output_vocab_path)

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    unique_chars = set()
    for txt_name in _read_split(split_path):
        txt_path = dataset_path / txt_name
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        unique_chars.update(text)

    vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
    next_id = 3
    for ch in sorted(unique_chars):
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1

    output_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab


class PashtoHandwritingDataset(Dataset):
    def __init__(self, split_file: str, dataset_dir: str, vocab_path: str,
                 transform=None, target_height: int = 32):
        self.split_file = Path(split_file)
        self.dataset_dir = Path(dataset_dir)
        self.vocab_path = Path(vocab_path)
        self.transform = transform
        self.target_height = target_height
        self.vocab = json.loads(self.vocab_path.read_text(encoding="utf-8"))
        self.pad_id = self.vocab.get("<PAD>", 1)
        self.unk_id = self.vocab.get("<UNK>", 2)
        self.text_filenames = _read_split(self.split_file)

    def __len__(self) -> int:
        return len(self.text_filenames)

    def _resolve_image_path(self, base_name: str) -> Path:
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            path = self.dataset_dir / f"{base_name}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Image not found: {base_name}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        txt_filename = self.text_filenames[idx]
        txt_path = self.dataset_dir / txt_filename
        img_path = self._resolve_image_path(Path(txt_filename).stem)
        image_cv = cv2.imread(str(img_path))
        image_tensor = robust_preprocess(image_cv, target_h=self.target_height)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        text = txt_path.read_text(encoding="utf-8").strip()
        label_ids = [self.vocab.get(ch, self.unk_id) for ch in text]
        return image_tensor, torch.tensor(label_ids, dtype=torch.long)


def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
               pad_id: int = 1) -> Dict[str, torch.Tensor]:
    images, labels = zip(*batch)
    image_widths = torch.tensor([img.shape[2] for img in images], dtype=torch.long)
    label_lengths = torch.tensor([lbl.numel() for lbl in labels], dtype=torch.long)
    max_width = int(image_widths.max())
    max_label_len = int(label_lengths.max())
    # Pad with 0.0 (black background — consistent with THRESH_BINARY_INV output)
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, 0), value=0.0) for img in images]
    padded_labels = [F.pad(lbl, (0, max_label_len - lbl.numel()), value=pad_id) for lbl in labels]
    return {
        "images": torch.stack(padded_images, dim=0),
        "labels": torch.stack(padded_labels, dim=0),
        "image_widths": image_widths,
        "label_lengths": label_lengths,
    }
