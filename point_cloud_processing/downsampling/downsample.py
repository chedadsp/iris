#!/usr/bin/env python3
"""
Downsample E57 scans using Open3D voxel grid.

Reads from data/input folder and saves to data/downsampled/.

Usage:
  poetry run downsample-e57
  poetry run downsample-e57 --voxel-size 0.05
"""

import sys
import glob
import argparse
import numpy as np
from pathlib import Path
import time

try:
    import pye57
except ImportError:
    print("ERROR: pye57 not installed. Install with: pip install pye57")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed. Install with: pip install open3d")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Downsample E57 scans using Open3D")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size in meters (default: 0.02)")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent.parent.resolve()
    input_dir = Path(script_dir).parent / "data" / "input"
    output_dir = script_dir / "data" / "downsampled"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all E57 files
    e57_files = sorted(glob.glob(str(input_dir / "*.e57")))
    
    if not e57_files:
        print(f"No E57 files found in {input_dir}")
        return
    
    print(f"Found {len(e57_files)} E57 files\n")
    
    # Process each file
    for input_e57 in e57_files:
        input_path = Path(input_e57)
        base_name = input_path.stem
        cm = int(round(args.voxel_size * 100))
        output_name = f"{base_name}_downsampled_{cm}cm.e57"
        output_path = output_dir / output_name
        
        # Skip if already downsampled
        if output_path.exists():
            print(f"✓ {output_name}")
            continue
        
        print(f"Processing {input_path.name}...")
        
        # Read E57
        try:
            e57 = pye57.E57(str(input_path))
            data = e57.read_scan(0, ignore_missing_fields=True)
            
            points = np.vstack((
                data["cartesianX"],
                data["cartesianY"],
                data["cartesianZ"]
            )).T
            
            print(f"  Original points: {points.shape[0]:,}")
            
            # Downsample with Open3D
            start = time.time()
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pcd_down = pcd.voxel_down_sample(args.voxel_size)
            ds_points = np.asarray(pcd_down.points)
            elapsed = time.time() - start
            
            print(f"  Downsampled: {ds_points.shape[0]:,} points ({elapsed:.2f}s)")
            
            # Write E57
            e57_out = pye57.E57(str(output_path), mode="w")
            e57_out.write_scan_raw({
                "cartesianX": ds_points[:, 0],
                "cartesianY": ds_points[:, 1],
                "cartesianZ": ds_points[:, 2],
            })
            e57_out.close()
            e57.close()
            
            print(f"  ✓ Saved: {output_name}\n")
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            import traceback
            traceback.print_exc()


def compute_spherical_from_xyz(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.mod(np.arctan2(y, x), 2 * np.pi)
    el = -(np.arcsin(z / r))
    return az, el, r

if __name__ == "__main__":
    main()
