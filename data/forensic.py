"""
Forensic Diagnostic Script
Run from PHTI directory: ..\.venv\Scripts\python.exe data/forensic.py
"""
import sys, json, cv2, numpy as np, torch, os
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

print('[YOLO FINDINGS - Already Established]')
print("""
CRITICAL BUGS:
  Bug 1: WhatsApp 1.35AM (2-line image) - YOLO only found 1 box (y=73-111 = SECOND LINE only!)
           First line 'يو زخم نه دے' is NEVER SEEN by the model.
  Bug 2: download.jpg - YOLO found 1 box at y=85-134 (bottom only). Top content missed.
  Bug 3: images.jpg  - 0 boxes. Projection fallback produces 1 huge crop (whole page).
           Model gets a 213x236 image scaled to 32px tall = unreadable mess.
  Bug 4: test_page boxes 4 and 5 are only 13-15px tall = too small = garbage output.
""")

with open('models/vocab.json', encoding='utf-8') as f:
    vocab = json.load(f)

from src.model import PashtoCRNN
from src.segmenter import YOLOSegmenter
from src.dataset import preprocess_adaptive, preprocess_morph

model = PashtoCRNN(num_classes=len(vocab))
state = torch.load('models/crnn_pashto.pth', map_location='cpu')
model.load_state_dict(state)
model.eval()

blank_id = vocab.get('<BLANK>', 0)
id2char_map = {int(v): k for k,v in vocab.items()}

def decode(logits):
    probs = torch.softmax(logits, dim=2)
    best = torch.argmax(probs, dim=2).squeeze(0).numpy()
    peaks = torch.max(probs, dim=2)[0].squeeze(0).numpy()
    chars, confs, last = [], [], -1
    for t, c in enumerate(best):
        if c != blank_id and c != last:
            chars.append(id2char_map.get(int(c), '?'))
            confs.append(float(peaks[t]))
        last = c
    return ''.join(chars), float(np.mean(confs)) if confs else 0.0

segmenter = YOLOSegmenter('models/best.pt')

print('[MODEL INFERENCE ON EACH CROP]')
for img_path in sorted(os.listdir('data/raw')):
    full = f'data/raw/{img_path}'
    if not os.path.isfile(full): continue
    img = cv2.imread(full)
    if img is None: continue
    h, w = img.shape[:2]
    print(f'\n  === {img_path} [{w}x{h}] ===')
    crops = segmenter.segment_lines(full)
    print(f'  YOLO crops: {len(crops)}')
    
    for i, crop in enumerate(crops):
        ch, cw = crop.shape[:2]
        print(f'  Crop {i+1}: {cw}x{ch}px  (aspect ratio={cw/max(ch,1):.1f}:1)')
        for pp_name, pp_fn in [('adaptive', preprocess_adaptive), ('morph', preprocess_morph)]:
            t = pp_fn(crop)
            if t is None:
                print(f'    [{pp_name}] FAILED - None tensor')
                continue
            t_b = t.unsqueeze(0)
            t_f = torch.flip(t_b, [3])
            for flip, tensor in [('N', t_b), ('F', t_f)]:
                with torch.no_grad():
                    logits = model(tensor)
                text, conf = decode(logits)
                print(f'    [{pp_name}/{flip}] conf={conf:.3f} nchars={len(text)} -> {repr(text[:70])}')

print('\n[PROJECTION SPLIT ON images.jpg]')
from src.segmenter import YOLOSegmenter
seg = YOLOSegmenter('models/best.pt')
crops = seg.segment_lines('data/raw/images.jpg')
print(f'  Projection crops: {len(crops)}')
for i, c in enumerate(crops):
    print(f'  Crop {i+1}: {c.shape[1]}x{c.shape[0]}px')
    t = preprocess_adaptive(c)
    if t is not None:
        with torch.no_grad():
            logits = model(t.unsqueeze(0))
        text, conf = decode(logits)
        print(f'    -> conf={conf:.3f} text={repr(text[:50])}')
