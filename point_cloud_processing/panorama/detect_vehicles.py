import os
import glob
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import json

def detect_vehicles_on_panorama():
    """
    Detects vehicles on panorama using YOLOv8.
    Saves detection results as JSON for point cloud filtering.
    """
    
    # Setup folders
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    panorama_dir = os.path.join(data_dir, "panoramas")
    output_dir = os.path.join(data_dir, "detected")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all panoramas (exclude preview and rgb versions)
    panorama_files = glob.glob(os.path.join(panorama_dir, "*_panorama.jpg"))
    
    if not panorama_files:
        print(f"Error: No panoramas found in {panorama_dir}!")
        print("First run: python extract_panorama.py")
        return
    
    print(f"Found {len(panorama_files)} panorama(s)")
    
    # Load YOLOv8 model
    print("Loading YOLO model...")
    model = YOLO('yolov8l.pt')
    
    # COCO classes for vehicles
    vehicle_classes = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    # Process each panorama
    for panorama_path in panorama_files:
        print(f"\nProcessing: {os.path.basename(panorama_path)}")
        
        # Check if detection JSON already exists
        basename = os.path.splitext(os.path.basename(panorama_path))[0].replace('_panorama', '')
        detection_json = os.path.join(output_dir, f"{basename}_panorama_detections.json")
        
        if os.path.exists(detection_json):
            print(f"  ✓ Detections already exist, skipping")
            continue
        
        # Load panorama
        img = cv2.imread(panorama_path)
        height, width = img.shape[:2]
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # YOLO detection
        results = model.predict(
            source=img,
            conf=0.15,
            iou=0.4,
            classes=list(vehicle_classes.keys()),
            imgsz=1280,
            verbose=False,
            agnostic_nms=True
        )
        
        # Analyze results
        detections = results[0].boxes
        print(f"  Found {len(detections)} vehicle(s)")
        
        # Prepare image for drawing and collect detection data
        img_annotated = img.copy()
        vehicles_found = []
        detection_data = {
            'panorama': os.path.basename(panorama_path),
            'dimensions': {'width': width, 'height': height},
            'vehicles': []
        }
        
        for i, box in enumerate(detections):
            # Detection data
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            vehicle_type = vehicle_classes.get(class_id, 'unknown')
            
            # Save to list for statistics
            vehicles_found.append({
                'type': vehicle_type,
                'confidence': confidence
            })
            
            # Save to JSON data with integer coordinates
            detection_data['vehicles'].append({
                'id': i + 1,
                'type': vehicle_type,
                'confidence': float(confidence),
                'bbox': {
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2)
                }
            })
            
            # Draw bounding box
            thickness = max(int(width / 2000), 10)
            color = (0, 255, 0)
            cv2.rectangle(img_annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Add label
            label = f"#{i+1} {vehicle_type} {confidence:.0%}"
            font_scale = max(width / 8000, 1.5)
            font_thickness = max(int(width / 4000), 3)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Background for text
            cv2.rectangle(img_annotated, 
                         (int(x1), int(y1) - label_size[1] - 20),
                         (int(x1) + label_size[0] + 10, int(y1)),
                         color, -1)
            cv2.putText(img_annotated, label, 
                       (int(x1) + 5, int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        
        # Generate output filename
        basename = os.path.splitext(os.path.basename(panorama_path))[0]
        output_path = os.path.join(output_dir, f"{basename}_detected.jpg")
        json_path = os.path.join(output_dir, f"{basename}_detections.json")
        
        # Save annotated image
        cv2.imwrite(output_path, img_annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  ✓ Saved: {os.path.basename(output_path)}")
        
        # Save JSON with detection data
        with open(json_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        print(f"  ✓ Saved: {os.path.basename(json_path)}")
        
        # Statistics
        if vehicles_found:
            for vehicle_type in vehicle_classes.values():
                count = sum(1 for v in vehicles_found if v['type'] == vehicle_type)
                if count > 0:
                    print(f"    {vehicle_type}: {count}")

if __name__ == "__main__":
    detect_vehicles_on_panorama()
