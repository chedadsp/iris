# Vehicle-Based Point Cloud Filtering

Filters E57 point clouds to retain only points belonging to detected vehicles.

## Overview

Uses vehicle detections from panorama (bounding boxes) to:
1. Convert 2D panorama coordinates to 3D spherical angles
2. Filter point cloud to keep only vehicle regions
3. Save filtered result as E57 + PCD files

## Prerequisites

Before filtering, you must complete these steps:
1. Extract panorama: `poetry run extract-panorama-pc`
2. Detect vehicles: `poetry run detect-vehicles-pc`
3. Downsample scan: `poetry run downsample-e57`

This generates:
- Panorama images: `point_cloud_processing/data/panoramas/`
- Detection JSON: `point_cloud_processing/data/detected/`
- Downsampled E57: `data/input/downsampled/`

## Usage

### From Poetry (recommended)
```bash
cd iris
poetry run filter-vehicles-pc
```

Automatically:
- Reads downsampled E57 from: `data/input/downsampled/`
- Reads detections from: `point_cloud_processing/data/detected/`
- Writes output to: `point_cloud_processing/data/filtered/`

### Direct Python
```bash
python filter_vehicles.py
```

## Data Paths

| Step | Input | Output |
|------|-------|--------|
| **Panorama Extraction** | `data/input/*.e57` | `point_cloud_processing/data/panoramas/` |
| **Vehicle Detection** | `data/panoramas/*.jpg` | `point_cloud_processing/data/detected/` |
| **Downsampling** | `data/input/*.e57` | `point_cloud_processing/data/downsampled/` |
| **Filtering** | `point_cloud_processing/data/downsampled/*.e57` | `point_cloud_processing/data/filtered/` |

## Coordinate Conversion

### Panorama → Spherical Mapping

```
Panorama (pixels)              Spherical angles
X: 0 → width   ════════════>  Azimuth:   0 → 2π   (mirrored)
Y: 0 → height  ════════════>  Elevation: π/2 → -π/2 (90° → -90°)
```

### Formula

```python
# Inverted azimuth (panorama is horizontally flipped)
az_min = (1 - bbox_x2 / img_width) * 2π
az_max = (1 - bbox_x1 / img_width) * 2π

# Inverted elevation (Y increases downward in image)
el_min = π/2 - (bbox_y2 / img_height) * π
el_max = π/2 - (bbox_y1 / img_height) * π
```

## Output

Generates for each input:
- **Filtered E57**: `*_downsampled_*_vehicles_only.e57`
- **Filtered PCD**: `*_downsampled_*_vehicles_only.pcd` (for visualization)
- **Location**: `point_cloud_processing/data/filtered/`

## Statistics

Output includes per-file:
- Total original points
- Points per vehicle detection
- Percentage of points retained
- File sizes (E57 + PCD)

## Supported E57 Data

Reads and preserves:
- **Cartesian**: X, Y, Z coordinates
- **Spherical**: Azimuth, Elevation, Range
- **Color**: R, G, B channels (if available)
- **Intensity**: Reflectance values (if available)

## Notes

- Points are filtered using vectorized boolean masking (fast)
- Vehicle bounding boxes with overlaps are automatically merged
- Downsampled files must have matching detection JSON
- Both E57 and PCD formats saved automatically