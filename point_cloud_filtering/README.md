# Vehicle Filtering from E57 Point Cloud

## Overview

This system automatically filters point cloud data, retaining **only points belonging to detected vehicles** while eliminating background points.

## Prerequisites

Before running this filtering process, you must first:

1. **Extract panoramas** from E57 files using `process_e57_panoramas.py`
2. **Run vehicle detection** with YOLO on the extracted panoramas

This generates the detection JSON files required for filtering.

## Filtering Process

### 1. **Input Data**

**Input files are located in:**
- **E57 files**: `e57_panorama_vehicle_detection/data/input/*.e57`
- **Detection JSON**: `e57_panorama_vehicle_detection/data/detected/*_detections.json`

```
📁 e57_panorama_vehicle_detection/data/
   ├─ input/
   │   └─ auto01.e57                    (10,000,000 points)
   └─ detected/
       └─ auto01_panorama_detections.json
           └─ vehicles: [{bbox}, {bbox}, ...]
```

**E57 Point Cloud contains:**
- **Cartesian coordinates**: (x, y, z) - position in 3D space
- **Spherical coordinates**: (azimuth, elevation, range) - angle and distance
- **RGB colors**: (r, g, b) - color of each point
- **~10 million points** in a typical scan

**Detection JSON contains:**
- List of vehicles with bounding boxes on 360° panorama
- Panorama dimensions (e.g., 8192 x 4096 px)
- Confidence score for each vehicle

---

### 2. **Bounding Box → Spherical Angles Conversion**

YOLO detects vehicles on a **2D panorama**, but point clouds use **3D spherical coordinates**.

#### Mapping:

```
Panorama (8192 x 4096 px)          Spherical coordinates
┌─────────────────────────┐        ┌─────────────────────────┐
│  X: 0 → 8192 px        │  ══>   │  Azimuth: 2π → 0 rad    │ (360° → 0°)
│  Y: 0 → 4096 px        │  ══>   │  Elevation: π/2 → -π/2  │ (90° → -90°)
└─────────────────────────┘        └─────────────────────────┘
```

#### Formula:

```python
# INVERTED AZIMUTH (panorama is horizontally flipped)
az_min = (1 - x2 / image_width) * 2π
az_max = (1 - x1 / image_width) * 2π

# INVERTED ELEVATION (top of image = +90°, bottom = -90°)
el_min = π/2 - (y2 / image_height) * π
el_max = π/2 - (y1 / image_height) * π
```

#### Example:

```
Bounding Box on panorama:
  x1=1000, y1=500, x2=1500, y2=1000
  Panorama: 8192 x 4096 px

Conversion:
  az_min = (1 - 1500/8192) * 2π = 5.13 rad (294°)
  az_max = (1 - 1000/8192) * 2π = 5.51 rad (316°)
  el_min = π/2 - (1000/4096) * π = 0.80 rad (46°)
  el_max = π/2 - (500/4096) * π  = 1.18 rad (68°)

Result:
  Vehicle covers angles:
  - Horizontal: 294° - 316° (22° wide)
  - Vertical: 46° - 68° (22° tall)
```

---

### 3. **Boolean Masking - Finding Points**

The code uses **NumPy boolean indexing** for fast filtering of millions of points.

#### Algorithm:

```python
# STEP 1: Create empty mask (all False)
vehicle_mask = np.zeros(10_000_000, dtype=bool)

# STEP 2: For each vehicle
for vehicle in vehicles:
    # Convert bbox to angles
    az_min, az_max, el_min, el_max = bbox_to_spherical(vehicle)
    
    # Find all points in this range
    bbox_mask = (
        (azimuth >= az_min) & (azimuth <= az_max) &
        (elevation >= el_min) & (elevation <= el_max)
    )
    
    # Add to main mask (OR operation)
    vehicle_mask |= bbox_mask

# STEP 3: Extract only True points
filtered_x = x[vehicle_mask]
filtered_y = y[vehicle_mask]
filtered_z = z[vehicle_mask]
```

---

### 4. **Output Formats**

The system generates **two output formats**:

#### **E57 Format** (with spherical coordinates)

```python
filtered_data = {
    "cartesianX": filtered_x,        # [100K]
    "cartesianY": filtered_y,
    "cartesianZ": filtered_z,
    "sphericalAzimuth": filtered_azimuth,
    "sphericalElevation": filtered_elevation,
    "sphericalRange": filtered_range,
    "colorRed": filtered_r,
    "colorGreen": filtered_g,
    "colorBlue": filtered_b
}

output_e57.write_scan_raw(filtered_data)
```

**Advantages:**
- ✅ Preserves both Cartesian and spherical coordinates
- ✅ Standard format for LiDAR scanners
- ✅ Can be re-imported into scanner software

#### **PCD Format** (Point Cloud Data)

```
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH 100000
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 100000
DATA ascii
-2.4531 1.8945 0.3421 4259906
-2.4528 1.8947 0.3423 4259906
...
```

**Advantages:**
- ✅ Easy to import into Python (Open3D, PCL)
- ✅ Supported by CloudCompare, MeshLab
- ✅ Smaller file size (only XYZ + RGB)

---

## Usage

**Step 1: Extract panoramas and detect vehicles**
```bash
cd e57_panorama_vehicle_detection
python process_e57_panoramas.py
```

**Step 2: Filter point clouds**
```bash
cd point_cloud_filtering
python filter_vehicles.py
```

**Folder structure:**
```
e57_panorama_vehicle_detection/
└─ data/
    ├─ input/
    │   ├─ auto01.e57
    │   ├─ auto02.e57
    │   └─ ...
    ├─ detected/
    │   ├─ auto01_panorama_detections.json
    │   ├─ auto02_panorama_detections.json
    │   └─ ...
    └─ panoramas/
        ├─ auto01_panorama.jpg
        └─ ...

point_cloud_filtering/
└─ output/
    ├─ auto01_vehicles_only.e57
    ├─ auto01_vehicles_only.pcd
    ├─ auto02_vehicles_only.e57
    ├─ auto02_vehicles_only.pcd
    └─ ...
```

---

## Notes

⚠️ **Coordinate transformations are inverted:**
- Panorama is **horizontally flipped** (mirror image)
- Panorama Y-axis is **inverted** (top = +90°, bottom = -90°)

⚠️ **E57 limitations:**
- Cannot be edited "in-place" - must create new file
- Large files (>1GB) can be slow to load

✅ **Advantages:**
- Preserves all vehicle information
- Dramatic reduction in file size
- Fast for analysis in downstream tools
