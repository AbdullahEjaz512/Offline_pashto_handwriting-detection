from ultralytics import YOLO
from PIL import Image

class YOLOSegmenter:
    """
    Line Detection Module for Full-Page documents.
    Detects text lines using YOLOv8, sorts them strictly top-to-bottom, and returns image crops.
    """
    def __init__(self, model_path="models/yolov8n.pt"):
        # Load the custom trained YOLOv8 line detection weights
        # Note: Depending on your exact trained file, this should point to your .pt weights
        self.model = YOLO(model_path)
        
    def segment_lines(self, full_page_image_path):
        """
        Runs YOLOv8 bounding box detection on the full image.
        Returns a list of PIL.Image crop objects ordered from the top of the page to the bottom.
        """
        # Run inference
        results = self.model(full_page_image_path)
        result = results[0] # Take first image result
        
        # Extract bounding boxes (x_min, y_min, x_max, y_max format)
        boxes = result.boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            return []
            
        # IMPORTANT: Sort boxes by y_min (index 1) to ensure reading order is Top-to-Bottom
        sorted_boxes = sorted(boxes, key=lambda b: b[1])
        
        # Crop the images for the line recognition model
        full_img = Image.open(full_page_image_path).convert('RGB')
        crops = []
        
        for box in sorted_boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            # Add slight vertical padding (optional but helpful if bboxes are tight)
            line_crop = full_img.crop((xmin, ymin, xmax, ymax))
            crops.append(line_crop)
            
        return crops
