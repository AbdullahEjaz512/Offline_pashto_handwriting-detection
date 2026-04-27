import cv2
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm

def apply_perspective_warp(image):
    h, w = image.shape[:2]
    # Small random tilt
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    delta = 0.05 * min(w, h)
    dst_points = np.float32([
        [random.uniform(0, delta), random.uniform(0, delta)],
        [w - random.uniform(0, delta), random.uniform(0, delta)],
        [random.uniform(0, delta), h - random.uniform(0, delta)],
        [w - random.uniform(0, delta), h - random.uniform(0, delta)]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(image, matrix, (w, h), borderValue=(255, 255, 255))

def apply_flashlight_effect(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (random.randint(0, w), random.randint(0, h))
    radius = random.randint(min(w, h) // 2, max(w, h))
    
    # Create radial gradient
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - center[1])**2 + (j - center[0])**2)
            mask[i, j] = max(0, 1 - dist / radius)
            
    # Apply brightness boost
    img_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)
    img_hls[:, :, 1] = img_hls[:, :, 1] * (1 + mask * 0.5)
    img_hls[:, :, 1] = np.clip(img_hls[:, :, 1], 0, 255)
    return cv2.cvtColor(img_hls.astype(np.uint8), cv2.COLOR_HLS2BGR)

def apply_shadows(image):
    h, w = image.shape[:2]
    shadow_mask = np.ones((h, w), dtype=np.float32)
    for _ in range(random.randint(1, 3)):
        center = (random.randint(0, w), random.randint(0, h))
        axes = (random.randint(w // 4, w), random.randint(h // 4, h))
        angle = random.randint(0, 360)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        shadow_mask = shadow_mask * (1 - (mask / 255.0) * random.uniform(0.1, 0.4))
    
    img_f = image.astype(np.float32)
    for i in range(3):
        img_f[:, :, i] *= shadow_mask
    return img_f.astype(np.uint8)

def apply_noise_and_blur(image):
    # Gaussian Noise
    noise = np.random.normal(0, random.uniform(2, 8), image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Slight motion blur
    if random.random() > 0.5:
        size = random.randint(2, 4)
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel /= size
        image = cv2.filter2D(image, -1, kernel)
    return image

def augment_sample(img_path, output_dir):
    image = cv2.imread(str(img_path))
    if image is None: return None
    
    # Chain augmentations
    image = apply_perspective_warp(image)
    image = apply_flashlight_effect(image)
    image = apply_shadows(image)
    image = apply_noise_and_blur(image)
    
    # Save augmented image
    out_name = f"aug_mobile_{img_path.name}"
    cv2.imwrite(str(output_dir / out_name), image)
    return out_name

def main():
    dataset_dir = Path("PHTI-DATASET")
    output_dir = Path("PHTI-MOBILE-AUGMENTED")
    output_dir.mkdir(exist_ok=True)
    
    print("Reading Training.txt...")
    with open("Training.txt", "r", encoding="utf-8") as f:
        samples = [line.strip() for line in f.readlines() if line.strip()]
    
    # Pick a subset for practical fine-tuning (e.g., 2000 samples)
    random.shuffle(samples)
    subset_size = 2000
    subset = samples[:subset_size]
    
    new_manifest = []
    print(f"Generating {subset_size} augmented mobile samples...")
    
    for sample_txt in tqdm(subset):
        img_name = sample_txt.replace(".txt", ".jpg")
        img_path = dataset_dir / img_name
        txt_path = dataset_dir / sample_txt
        
        if not img_path.exists() or not txt_path.exists():
            continue
            
        # Create augmented image
        aug_img_name = augment_sample(img_path, output_dir)
        if aug_img_name:
            # Create corresponding txt file with same name
            aug_txt_name = aug_img_name.replace(".jpg", ".txt")
            with open(txt_path, "r", encoding="utf-8") as fin:
                content = fin.read()
            with open(output_dir / aug_txt_name, "w", encoding="utf-8") as fout:
                fout.write(content)
            
            new_manifest.append(f"{aug_txt_name}\n")

    # Save the new training list
    with open("Training_Mobile.txt", "w", encoding="utf-8") as f:
        f.writelines(new_manifest)
    
    print(f"Done! Created {len(new_manifest)} samples in PHTI-MOBILE-AUGMENTED")
    print("New training list saved to Training_Mobile.txt")

if __name__ == "__main__":
    main()
