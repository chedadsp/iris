# Point Cloud Processing Module

Integrated point cloud processing pipeline with modular structure for panorama extraction, vehicle detection, downsampling, and filtering.

## Structure

```
point_cloud_processing/
├── e57_process.py              # Main orchestration script
├── downsampling/
│   ├── downsample.py           # Open3D-based downsampling
│   ├── voxel_downsample.py     # NumPy fallback
│   ├── README.md
│   └── __init__.py
├── panorama/
│   ├── extract_panorama.py     # E57 → Panorama JPG
│   ├── detect_vehicles.py      # YOLO detection
│   ├── README.md
│   └── __init__.py
├── filtering/
│   ├── filter_vehicles.py      # Filter by detections
│   ├── README.md
│   └── __init__.py
├── data/
│   ├── panoramas/              # Extracted panoramas
│   ├── detected/               # YOLO detection JSON
│   └── filtered/               # Filtered E57 + PCD output
├── __init__.py
└── README.md
```

## Quick Start

### Prerequisites
Ensure these steps are completed first:
```bash
cd iris
poetry install
```

### Pipeline Steps (in order)

**1. Extract Panoramas**
```bash
poetry run extract-panorama-pc
```
Input: `data/input/*.e57`  
Output: `point_cloud_processing/data/panoramas/`

**2. Detect Vehicles**
```bash
poetry run detect-vehicles-pc
```
Input: `point_cloud_processing/data/panoramas/*_panorama.jpg`  
Output: `point_cloud_processing/data/detected/*_detections.json`

**3. Downsample E57**
```bash
poetry run downsample-e57
poetry run downsample-e57 --voxel-size 0.05  # 5cm voxels
```
Input: `data/input/*.e57`  
Output: `data/input/downsampled/*_downsampled_*.e57`

**4. Filter by Vehicles**
```bash
poetry run filter-vehicles-pc
```
Input: `data/input/downsampled/*.e57` + detections  
Output: `point_cloud_processing/data/filtered/`

### Complete Pipeline
```bash
poetry run process-point-cloud
```
Runs all 4 steps automatically.

## Data Paths

| Step | Input | Output |
|------|-------|--------|
| Panorama Extract | `data/input/*.e57` | `point_cloud_processing/data/panoramas/` |
| Vehicle Detect | `data/panoramas/*.jpg` | `point_cloud_processing/data/detected/` |
| Downsample | `data/input/*.e57` | `point_cloud_processing/data/downsampled/` |
| Filter | `point_cloud_processing/data/downsampled/*.e57` | `point_cloud_processing/data/filtered/` |

## Modules

### Downsampling (`downsampling/`)
- **downsample.py**: Open3D-based voxel downsampling
  - Processes all E57 files in `data/input/`
  - Saves to `data/input/downsampled/` subfolder
  - Default voxel size: 0.02m (2cm)
  - Uses Open3D for C++ optimization (3-5x faster)

### Panorama (`panorama/`)
- **extract_panorama.py**: Converts E57 to 360° JPG panoramas
  - High-resolution output (20480×10240px)
  - Preserves spherical coordinates for filtering
- **detect_vehicles.py**: YOLO vehicle detection
  - Generates bounding boxes in JSON format
  - Uses yolov8l model (or yolov8n for speed)

### Filtering (`filtering/`)
- **filter_vehicles.py**: Point cloud filtering by detections
  - Converts 2D panorama boxes → 3D spherical angles
  - Filters E57 to keep vehicle regions only
  - Saves both E57 and PCD formats

## Configuration

### Voxel Size
- Default: 0.02m (2cm)
- Adjust with `--voxel-size` flag
- Values: 0.02m (fine), 0.05m (balanced), 0.10m (coarse)

### Performance (Open3D)
Downsampling 50M point cloud with 0.05m voxel:
- Open3D: ~10 seconds
- NumPy: ~45 seconds

## Notes

- Downsampling creates subfolder `data/input/downsampled/` automatically
- Files skipped if already processed
- All steps use Poetry CLI scripts (see `pyproject.toml`)
- Filter requires both E57 and detection JSON

