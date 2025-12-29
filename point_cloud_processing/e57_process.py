#!/usr/bin/env python3
"""
COMPLETE POINT CLOUD PROCESSING PIPELINE (ORCHESTRATOR)
========================================================
Coordinates all processing steps by calling individual scripts.
Each script handles ALL files in the input folder.

Pipeline:
1. Extract panorama from ALL E57 files
2. Detect vehicles on ALL panoramas (YOLO)
3. Downsample ALL E57 scans (voxel grid)
4. Filter ALL downsampled scans by vehicle detections

Usage:
  poetry run process-point-cloud
  poetry run process-point-cloud --voxel-size 0.05
  poetry run process-point-cloud --method open3d
"""

import subprocess
import sys
import argparse


def run_command(cmd, description):
    """Run a command and report status."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete point cloud processing pipeline (orchestrator)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel size in meters (default: 0.02)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("POINT CLOUD PROCESSING PIPELINE")
    print("="*70)
    print(f"Voxel size: {args.voxel_size}m")
    
    # Step 1: Extract panoramas from ALL E57 files
    if not run_command(
        ["poetry", "run", "extract-panorama-pc"],
        "STEP 1: EXTRACT PANORAMAS"
    ):
        print("\n⚠ Continuing despite panorama extraction issues...")
    
    # Step 2: Detect vehicles on ALL panoramas
    if not run_command(
        ["poetry", "run", "detect-vehicles-pc"],
        "STEP 2: DETECT VEHICLES"
    ):
        print("\n⚠ Continuing despite vehicle detection issues...")
    
    # Step 3: Downsample ALL E57 files
    downsample_cmd = ["poetry", "run", "downsample-e57", "--voxel-size", str(args.voxel_size)]
    
    if not run_command(
        downsample_cmd,
        f"STEP 3: DOWNSAMPLE (voxel size: {args.voxel_size}m)"
    ):
        print("\n✗ Downsampling failed - cannot continue")
        return False
    
    # Step 4: Filter downsampled E57 files by vehicle detections
    if not run_command(
        ["poetry", "run", "filter-vehicles-pc"],
        "STEP 4: FILTER BY VEHICLES"
    ):
        print("\n✗ Filtering failed")
        return False
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nOutput files:")
    print("  Panoramas:    point_cloud_processing/data/panoramas/")
    print("  Detections:   point_cloud_processing/data/detected/")
    print("  Downsampled:  data/input/downsampled/")
    print("  Filtered:     point_cloud_processing/data/filtered/")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
