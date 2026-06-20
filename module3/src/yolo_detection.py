"""
Module 3: YOLOv8 Dog Detection Wrapper
Detect and localize dogs in images before breed classification
"""

import cv2
import numpy as np


# COCO class ID for dog
DOG_CLASS_ID = 16


def load_yolo_model(model_name='yolov8n.pt'):
    """
    Load YOLOv8 model for object detection.
    
    Args:
        model_name: YOLOv8 model variant (n/s/m/l/x)
    
    Returns:
        YOLO model instance
    """
    from ultralytics import YOLO
    return YOLO(model_name)


def detect_dog(model, image_path, conf_threshold=0.5):
    """
    Detect dogs in an image.
    
    Args:
        model: YOLOv8 model
        image_path: Path to image file
        conf_threshold: Confidence threshold for detection
    
    Returns:
        List of dog bounding boxes [(x1, y1, x2, y2, confidence), ...]
    """
    results = model(image_path, verbose=False, conf=conf_threshold)
    
    dog_boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == DOG_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                dog_boxes.append((x1, y1, x2, y2, conf))
    
    return dog_boxes


def crop_dog_region(image_path, bbox, padding=10):
    """
    Crop detected dog region from image.
    
    Args:
        image_path: Path to image file
        bbox: Bounding box (x1, y1, x2, y2, conf)
        padding: Padding around bounding box
    
    Returns:
        Cropped image as numpy array (RGB)
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return img_rgb[y1:y2, x1:x2]


def detect_and_crop(model, image_path):
    """
    Complete detection pipeline: detect dog and crop region.
    
    Args:
        model: YOLOv8 model
        image_path: Path to image file
    
    Returns:
        cropped: Cropped dog image (or full image if no dog detected)
        bbox: Bounding box (or None)
    """
    boxes = detect_dog(model, image_path)
    
    if boxes:
        # Use highest confidence detection
        best_box = max(boxes, key=lambda x: x[4])
        cropped = crop_dog_region(image_path, best_box)
        return cropped, best_box[:4]
    else:
        # Return full image if no dog detected
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb, None


def draw_detection(image_path, bbox, output_path=None):
    """
    Draw bounding box on image.
    
    Args:
        image_path: Path to input image
        bbox: Bounding box (x1, y1, x2, y2)
        output_path: Optional path to save result
    
    Returns:
        Image with drawn bounding box
    """
    img = cv2.imread(image_path)
    
    if bbox:
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, 'Dog', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
