# Lidar datasets - large lidar files processing - INITIAL OVERVIEW

## 🚀 Quick Start (TL;DR)

If you just need to **downsample a large E57 LiDAR file**, use the preprocessing script:

```bash
python tools/cc_downsample.py <input.e57> --voxel-size 0.05 --out-format pcd
```

**Output:** `data/lidar_preprocessing/downsampled_e57_5cm/<basename>_downsampled_5cm.pcd`

### Prerequisites
- CloudCompare installed and in PATH (or specify with `--cc-path`)
- Python 3.10+
- Dependencies: `numpy`, `pye57` (optional for validation)

### Common Examples

**Downsample to 5 cm (default PCD output):**
```bash
python tools/cc_downsample.py data/auto01.e57 -v 0.05
```

**Downsample to 3 cm with E57 output:**
```bash
python tools/cc_downsample.py data/auto01.e57 -v 0.03 --out-format e57
```

**Apply Statistical Outlier Removal (SOR) + 5 cm downsampling:**
```bash
python tools/cc_downsample.py data/auto01.e57 -v 0.05 --sor --sor-k 20 --sor-std 1.0
```

**Show CloudCompare logs (verbose):**
```bash
python tools/cc_downsample.py data/auto01.e57 -v 0.05 --no-silent
```

**Validate E57 checksum (requires pye57):**
```bash
python tools/cc_downsample.py data/auto01.e57 -v 0.05 --out-format e57 --validate-e57
```

### Output Formats

| Format | Pros | Cons |
|--------|------|------|
| **PCD** (default) | Fast, no checksum issues, small files | Less standardized |
| **E57** | Industry standard, preserves metadata | CloudCompare CLI has checksum bug |
| **ROS Bag** | ROS-compatible (requires rosbag library) | Needs special setup |

### Metadata

Every run generates a JSON file alongside the output:
```json
{
  "source_file": "...",
  "output_file": "...",
  "output_format": "pcd",
  "voxel_size_m": 0.05,
  "voxel_label": "5cm",
  "sor_used": false,
  "cc_returncode": 0,
  "timestamp": "2025-12-08T17:47:40.123Z"
}
```

---

## Table of Contents

1. [Overview](#1-overview)
2. [Motivation](#2-motivation)
3. [High-level Objectives](#3-high-level-objectives)
4. [Tools & Libraries (recommended)](#4-tools--libraries-recommended)
5. [Key Techniques (How)](#5-key-techniques-how)
   - [5.1 Chunk reading / streaming](#51-chunk-reading--streaming)
   - [5.2 Spatial tiling](#52-spatial-tiling)
   - [5.3 Angular / horizontal slicing](#53-angular--horizontal-slicing)
   - [5.4 Downsampling & decimation](#54-downsampling--decimation)
   - [5.5 Range filtering](#55-range-filtering)
   - [5.6 Parallel processing](#56-parallel-processing)
   - [5.7 Output formats and metadata](#57-output-formats-and-metadata)
   - [5.8 Validation & benchmarking](#58-validation--benchmarking)
6. [Suggested Pipeline (minimal)](#6-suggested-pipeline-minimal)
7. [LiDAR Preprocessing Script (CloudCompare backend)](#7-lidar-preprocessing-script-cloudcompare-backend)

## 1. Overview
This document describes the approach for processing very large E57 LiDAR scans (tens to hundreds of millions of points) and transforming a single mega-scan into multiple *virtual* LiDAR datasets. Each virtual dataset should mimic a different sensor configuration (density, vertical/horizontal coverage, range, noise, etc.) suitable for experiments and evaluation in the master project.

## 2. Motivation
- Single E57 files at ~4 GB / ~170M points are infeasible to load fully into RAM with native readers.
- Generating multiple sensor variants from one scan is more reproducible and cheaper than capturing multiple devices.
- Proper streaming, chunking and format choices reduce I/O, memory usage and processing time.
- Using reference implementations (libE57) and optimized pipelines (PDAL, binary PCD/LAS) increases robustness and performance.

## 3. High-level Objectives
1. Efficiently read and iterate E57 content without loading all points into memory.
2. Split/transform original scan into multiple datasets that simulate different physical sensors:
   - Different point densities (downsampling ratios).
   - Different vertical fields-of-view / horizontal slices (Z-range, azimuth bands).
   - Different ranges (short-range vs long-range).
   - Different angular resolutions (simulate fewer scan-lines).
3. Write outputs in compact binary formats (binary PCD, LAS/LAZ) and produce metadata for each virtual sensor.
4. Automate and parallelize processing per tile/segment to scale across cores and machines.

## 4. Tools & Libraries (recommended)
- pye57 / libE57: for format understanding and low-level access. Use pye57 carefully (see optimizations).
- PDAL: recommended for production pipelines (readers.e57, filters.splitter, filters.decimation, writers.pcd/writers.las).
- Open3D / numpy: local processing (downsample, filtering) for small chunks.
- laszip / LAZ: compressed storage for LAS outputs.
- Multiprocessing / joblib: parallelize independent tile processing.

## 5. Key Techniques (How)

### 5.1 Chunk reading / streaming
- Avoid read_scan() that materializes all points. Use APIs that read records/scanlines or PDAL streaming.
- Process one chunk at a time and write results immediately to disk.

### 5.2 Spatial tiling
- Split scene into tiles (grid, octree, PDAL filters.splitter). Process each tile independently to bound memory.

### 5.3 Angular / horizontal slicing
- Compute azimuth = atan2(y, x) or use scanline metadata to extract angular bands.
- Select points within azimuth or Z ranges to simulate vertical/horizontal FOVs and different numbers of scan-lines.

### 5.4 Downsampling & decimation
- Voxel grid downsample or random/subsample per tile to create lower-density variants.
- Use deterministic downsampling (seeded) for reproducibility.

### 5.5 Range filtering
- Filter points by distance to sensor origin to simulate short-range sensors.

### 5.6 Parallel processing
- Assign tiles/slices to worker processes. Keep I/O contention low (SSD recommended).

### 5.7 Output formats and metadata
- Write binary PCD or LAS/LAZ. Include JSON metadata per dataset describing original source, filters applied, frame counts, and parameters.

### 5.8 Validation & benchmarking
- Measure memory usage, processing time per tile, throughput (points/second).
- Compute simple metrics per virtual sensor (point count, bounding box, density histogram, angular coverage).

## 6. Suggested Pipeline (minimal)
1. Read E57 metadata (poses, scan structure).
2. Decide tiling strategy (spatial grid or angular bands).
3. For each tile:
   - Stream-read points for tile.
   - Apply filters for each desired virtual sensor (downsample, z/azimuth/range cuts).
   - Write outputs as binary PCD / LAS.
   - Write dataset metadata (parameters, counts).
4. Optionally archive datasets (ZIP/LAZ).


## 7. LiDAR Preprocessing Script (CloudCompare Backend)

This document describes the purpose and usage of the Python-based LiDAR preprocessing tool that automates downsampling and filtering of **.e57 point cloud files** using **CloudCompare** in the background.

---

##  Purpose of the Script

The goal of this tool is to enable fast and reproducible preprocessing of large LiDAR scans without manually using the CloudCompare GUI.  
Processing is executed through command-line calls, allowing integration into pipelines, faster batch processing, and standardized results.

The script supports:

### ✔ Input: `.e57` point cloud file  
### ✔ Output: downsampled + optionally filtered `.e57` point cloud  
### ✔ Voxel Grid Downsampling (3 cm / 5 cm or custom value)  
### ✔ Statistical Outlier Removal (SOR)  
### ✔ Automatic output folder and filename generation  

---

##  Processing Steps

1. **Load input `.e57` point cloud**
2. **Apply voxel grid downsampling**  
   - User provides voxel size (e.g. `0.03`, `0.05`)  
   - Output folder is created automatically based on voxel size  
     - `data/lidar_preprocessing/e57_3cm/`  
     - `data/lidar_preprocessing/e57_5cm/`
3. **Apply SOR filtering (optional)**  
   - Removes noisy points based on statistical outlier detection  
   - Parameters used: `k=20`, `std=1.0` by default
4. **Export processed file**
   - Output name format:  
     **`<inputName>_downsampled_<voxelSize>cm.e57`**

---

##  Example Usage

```bash
python tools/cc_downsample.py input.e57 0.05
```