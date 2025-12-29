# E57 Panorama Vehicle Detection

Extracts panoramas from E57 laser scan files and detects vehicles using YOLOv8.

## Overview

This module:
1. Extracts 360° panoramic images from E57 point cloud files
2. Detects vehicles on panoramas using YOLOv8 deep learning model
3. Saves detection results as JSON for point cloud filtering

## Usage

### Extract Panoramas
```bash
python extract_panorama.py
```

Reads E57 files from main `data/input/` and saves panoramas to `data/panoramas/`

### Detect Vehicles
```bash
python detect_vehicles.py
```

Runs YOLO on extracted panoramas and saves results to `data/detected/`

### Full Pipeline (from parent directory)
```bash
python e57_process.py
```

## Output Files

- **Panoramas**: `../data/panoramas/*.jpg`
- **Detections**: `../data/detected/*_detections.json`

## Detection JSON Format

```json
{
  "image_width": 20714,
  "image_height": 10357,
  "vehicles": [
    {
      "id": 1,
      "type": "car",
      "confidence": 0.92,
      "bbox": {
        "x1": 1000,
        "y1": 500,
        "x2": 1500,
        "y2": 1000
      }
    }
  ]
}
```

## Requirements

```
pye57
ultralytics (YOLO)
opencv-python (cv2)
pillow (PIL)
```

## Notes

- E57 files must be in main `data/input/` folder
- YOLO model (yolov8l.pt) is automatically downloaded on first use
- Vehicle types: car, motorcycle, bus, truck
