# E57 Panorama Vehicle Detection

Extracts panoramas from E57 laser scan files and detects vehicles using YOLOv8.

## Folder Structure

```
e57_panorama_vehicle_detection/
├── data/
│   ├── input/          # Place your E57 files here
│   ├── panoramas/      # Extracted panoramas (auto-generated)
│   └── detected/       # Detection results (auto-generated)
├── extract_panorama.py
├── detect_vehicles.py
├── process_e57_panoramas.py
└── README.md
```

## Usage

1. Place E57 files in `data/input/` folder (create if doesn't exist)
2. Run the pipeline:
```bash
poetry run python e57_panorama_vehicle_detection/process_e57_panoramas.py
```

## Output

- Extracted panoramas: `data/panoramas/`
- Detected vehicles: `data/detected/`

## Requirements

```bash
poetry install
poetry add pye57 ultralytics opencv-python pillow numpy
```

## Individual Scripts

```bash
# Extract panoramas from E57 files
poetry run python e57_panorama_vehicle_detection/extract_panorama.py

# Detect vehicles on panoramas
poetry run python e57_panorama_vehicle_detection/detect_vehicles.py

# Run complete pipeline
poetry run python e57_panorama_vehicle_detection/process_e57_panoramas.py
```
