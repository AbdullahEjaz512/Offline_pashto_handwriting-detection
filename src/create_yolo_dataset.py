import os
import random
import glob
from PIL import Image

def generate_synthetic_yolo_page(
    source_images: list, 
    num_lines: int, 
    page_size=(1200, 1600), # Width, Height (Standard A4 ratio)
    output_img_path="output.jpg",
    output_lbl_path="output.txt"
):
    """
    Creates a synthetic full page by vertically stacking randomly selected text line images.
    Generates accompanying YOLOv8 bounding box annotations.
    """
    canvas_w, canvas_h = page_size
    # Create plain white canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    
    # We want uniform margins and spacing
    margin_x = 100
    margin_y = 150
    usable_width = canvas_w - (2 * margin_x)
    usable_height = canvas_h - (2 * margin_y)
    
    # Vertical spacing between lines
    spacing = usable_height // (num_lines + 1)
    y_offset = margin_y
    
    yolo_labels = [] # Stores strings of "class_id x_center y_center width height"
    class_id = 0     # 0 = "text_line"
    
    # Select random lines from the source dataset
    selected_images = random.sample(source_images, num_lines)
    
    for img_path in selected_images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping bad image {img_path}: {e}")
            continue
            
        orig_w, orig_h = img.size
        
        # Scale the image so it fits nicely within the page margins while maintaining aspect ratio
        scaling_factor = min(usable_width / orig_w, (spacing * 0.8) / orig_h)
        new_w = int(orig_w * scaling_factor)
        new_h = int(orig_h * scaling_factor)
        new_h = max(1, new_h) # Prevent 0 height
        
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Center horizontally on the page
        x_offset = (canvas_w - new_w) // 2
        
        # Paste onto canvas
        canvas.paste(img, (x_offset, y_offset))
        
        # Calculate YOLO Bounding Box (Normalized coords: 0.0 - 1.0)
        x_center = (x_offset + (new_w / 2)) / canvas_w
        y_center = (y_offset + (new_h / 2)) / canvas_h
        norm_width = new_w / canvas_w
        norm_height = new_h / canvas_h
        
        # Append format: <class_id> <x_center> <y_center> <width> <height>
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Move down for the next line
        y_offset += spacing
        
    # Save Image
    canvas.save(output_img_path, quality=95)
    
    # Save YOLO Labels (.txt)
    with open(output_lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_labels))


def build_synthetic_dataset(dataset_dir="PHTI-DATASET", num_pages=100, out_dir="yolo_synthetic"):
    """
    Driver sequence to build a dataset of N synthetic pages out of PHTI line crops.
    """
    img_out_dir = os.path.join(out_dir, "images", "train")
    lbl_out_dir = os.path.join(out_dir, "labels", "train")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    
    print(f"Scanning {dataset_dir} for JPG images...")
    all_images = glob.glob(os.path.join(dataset_dir, "*.jpg"))
    
    if not all_images:
        print(f"Error: No images found in {dataset_dir}!")
        return
        
    print(f"Found {len(all_images)} images. Generating {num_pages} synthetic full pages...")
    
    for i in range(num_pages):
        # Choose to paste between 4 and 8 lines per simulated page
        num_lines = random.randint(4, 6)
        
        img_path = os.path.join(img_out_dir, f"syn_page_{i:04d}.jpg")
        lbl_path = os.path.join(lbl_out_dir, f"syn_page_{i:04d}.txt")
        
        generate_synthetic_yolo_page(
            source_images=all_images,
            num_lines=num_lines,
            output_img_path=img_path,
            output_lbl_path=lbl_path
        )
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_pages} pages...")
            
    print(f"\nDone! YOLOv8 synthetic dataset saved to `{out_dir}/`")
    print("Folder Structure:")
    print(f"  {out_dir}/images/train/*.jpg")
    print(f"  {out_dir}/labels/train/*.txt")

if __name__ == "__main__":
    # Feel free to adjust the number of synthetic pages you want generated here:
    build_synthetic_dataset(
        dataset_dir="PHTI-DATASET", 
        num_pages=500, 
        out_dir="yolo_synthetic_dataset"
    )
