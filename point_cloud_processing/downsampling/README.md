# Voxel-Based Point Cloud Downsampling

Reduces E57 point cloud density using Open3D voxel grid approach.

## Overview

Divides 3D space into cubic voxels and selects one representative point per voxel. Preserves geometric structure while reducing file size 10-100x depending on voxel size.

## Usage

### From Poetry (Recommended)
```bash
poetry run downsample-e57
poetry run downsample-e57 --voxel-size 0.05  # 5cm voxels
```

Automatically:
- Finds all E57 files in: `data/input/` (main data folder)
- Skips already downsampled files
- Saves to: `point_cloud_processing/data/downsampled/`

### Direct Python
```bash
python downsampling/downsample.py --voxel-size 0.02
```

## Parameters

- `--voxel-size`: Voxel grid size in meters (default: 0.02m)
  - **0.02m** = 2cm (fine detail) — DEFAULT
  - **0.05m** = 5cm (balanced)
  - **0.10m** = 10cm (coarse, faster)

## Performance

**Open3D Speed (50M points, 0.05m voxel):**
- Downsampling: ~10 seconds
- Result: ~2-5M points (typical 5-10% retention)

**Typical Reductions:**
- 0.02m (2cm) = 10-20% of original
- 0.05m (5cm) = 5-10% of original
- 0.10m (10cm) = 2-5% of original

## Output

Files saved to: `point_cloud_processing/data/downsampled/`

Format: `{original}_downsampled_{cm}cm.e57`

Example:
```
Input:  data/input/scan001.e57
Output: point_cloud_processing/data/downsampled/scan001_downsampled_2cm.e57
        point_cloud_processing/data/downsampled/scan001_downsampled_5cm.e57
```

## Algorithm

1. Create PointCloud object from cartesian coordinates
2. Apply voxel_down_sample() from Open3D
3. Extract downsampled points as NumPy array
4. Write to new E57 file

## Notes

- Uses Open3D for C++ optimization (3-5x faster than NumPy)
- Processes all E57 files in `data/input/` automatically
- Skips files that already have downsampled versions
- Output contains only cartesian coordinates (X, Y, Z)
- Used as input for `filter-vehicles-pc` step

