# Voxel-Based Point Cloud Downsampling (E57)

## Overview

This module implements a **voxel-based downsampling algorithm** for large-scale LiDAR point clouds stored in the **E57** format.  
The primary goal is to reduce point cloud density in a **deterministic, efficient, and reproducible** manner, while preserving the global geometric structure of the scene.

The implementation is designed to be:
- independent of CloudCompare GUI and CLI,
- easy to integrate into the IRIS project,
- suitable for generating multiple datasets from a single real-world LiDAR scan.

---

## Motivation

Modern terrestrial LiDAR scans often contain **tens of millions of points**, making direct processing computationally expensive and impractical for algorithmic experimentation.

Voxel-based downsampling enables:
- controlled reduction of point density,
- simulation of different LiDAR sensor resolutions,
- efficient preprocessing for downstream tasks such as detection, segmentation, or scan simulation.

This approach mirrors the core logic used in tools such as **CloudCompare**, but is implemented here in pure Python for transparency and flexibility.

---

## Method: Voxel Downsampling

### Concept

The 3D space is divided into a regular grid of cubic voxels with edge length `voxel_size`.

For each point:

1. Its voxel index is computed as:
voxel_index = floor(point / voxel_size)

2. All points that fall into the same voxel are grouped together.

3. A **single representative point** is selected per voxel.

In this implementation, the **first encountered point per voxel** is retained.  
This strategy is computationally efficient and comparable to the "random point per cell" option used in CloudCompare.

---

### Algorithm Properties

- **One point per voxel**
- **No centroid computation**
- **No explicit loops over voxels**
- **Vectorized NumPy implementation**

This ensures fast execution even for point clouds containing tens of millions of points.

---

## Implementation Details

### Core Function

```python
def voxel_downsample(points, voxel_size):
 voxel_indices = np.floor(points / voxel_size).astype(np.int32)

 _, unique_indices = np.unique(
     voxel_indices,
     axis=0,
     return_index=True
 )

 return points[unique_indices]