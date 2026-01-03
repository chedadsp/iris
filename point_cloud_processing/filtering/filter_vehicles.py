#!/usr/bin/env python3
"""
FILTER VEHICLES FROM E57 POINT CLOUD
====================================
Filters E57 point cloud to keep only vehicle points based on YOLO detections.

Process:
1. Load E57 point cloud with spherical coordinates
2. Load YOLO vehicle detections (bounding boxes on panorama)
3. Convert bbox pixels to spherical angles
4. Filter points where angles fall within vehicle bboxes
5. Save filtered point cloud as new E57 file
"""

import os
import json
import pye57
import numpy as np
from pathlib import Path
import argparse
from ..downsampling.downsample import compute_spherical_from_xyz

AZ_OFFSET = np.pi  # probaj π, π/2, -π/2

def save_as_pcd(output_path, x, y, z, r=None, g=None, b=None):
    """Save point cloud as PCD file for better compatibility."""
    
    n_points = len(x)
    has_color = r is not None
    
    with open(output_path, 'w') as f:
        # Header
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        
        if has_color:
            f.write("FIELDS x y z rgb\n")
            f.write("SIZE 4 4 4 4\n")
            f.write("TYPE F F F U\n")
            f.write("COUNT 1 1 1 1\n")
        else:
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
        
        f.write(f"WIDTH {n_points}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n_points}\n")
        f.write("DATA ascii\n")
        
        # Data
        if has_color:
            for i in range(n_points):
                rgb = (int(r[i]) << 16) | (int(g[i]) << 8) | int(b[i])
                f.write(f"{x[i]} {y[i]} {z[i]} {rgb}\n")
        else:
            for i in range(n_points):
                f.write(f"{x[i]} {y[i]} {z[i]}\n")


def bbox_to_spherical(bbox, image_width, image_height):
    """
    Convert panorama bbox pixels to spherical angle ranges.
    
    Args:
        bbox: dict with x1, y1, x2, y2 (pixel coordinates)
        image_width: panorama width in pixels
        image_height: panorama height in pixels
    
    Returns:
        tuple: (az_min, az_max, el_min, el_max) in radians
    """
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    # Azimuth: panorama is mirrored
    az_min = (1 - x2 / image_width) * 2 * np.pi
    az_max = (1 - x1 / image_width) * 2 * np.pi

    # Elevation: top = +90°, bottom = -90°
    el_min = np.pi/2 - (y2 / image_height) * np.pi
    el_max = np.pi/2 - (y1 / image_height) * np.pi
    
    return az_min, az_max, el_min, el_max


def az_in_range(az, az_min, az_max):
    if az_min <= az_max:
        return (az >= az_min) & (az <= az_max)
    else:
        return (az >= az_min) | (az <= az_max)

def filter_points_by_vehicle_bboxes(e57_path, detections_json_path, output_path):
    """
    Main filtering function.
    
    Args:
        e57_path: path to input E57 file
        detections_json_path: path to vehicle detections JSON
        output_path: path for output filtered E57 file
    """
    
    print(f"\nProcessing: {Path(e57_path).name}")
    print(f"Detections: {Path(detections_json_path).name}")
    
    try:
        with open(detections_json_path, 'r') as f:
            detections = json.load(f)
        
        vehicles = detections['vehicles']
        image_width = detections['dimensions']['width']
        image_height = detections['dimensions']['height']
        
        print(f"Vehicles: {len(vehicles)}")
        
    except Exception as e:
        print(f"✗ Error loading detections: {e}")
        return
    
    try:
        e57 = pye57.E57(str(e57_path))
        scan_data = e57.read_scan(0, ignore_missing_fields=True)
        
        # Extract coordinates and spherical data
        x = scan_data['cartesianX']
        y = scan_data['cartesianY']
        z = scan_data['cartesianZ']
        if (
            'sphericalAzimuth' in scan_data and
            'sphericalElevation' in scan_data and
            'sphericalRange' in scan_data
        ):
            azimuth = scan_data['sphericalAzimuth']
            elevation = scan_data['sphericalElevation']
            range_data = scan_data['sphericalRange']
        else:
            azimuth, elevation, range_data = compute_spherical_from_xyz(
                scan_data['cartesianX'],
                scan_data['cartesianY'],
                scan_data['cartesianZ']
            )

        print("POINT CLOUD SPHERICAL RANGES:")
        print("  azimuth   min/max:", azimuth.min(), azimuth.max())
        print("  elevation min/max:", elevation.min(), elevation.max())
        print("  range     min/max:", range_data.min(), range_data.max())
        
        total_points = len(x)
        print(f"Points: {total_points:,}")
        
        # Check for color data
        has_color = 'colorRed' in scan_data
        if has_color:
            color_r = scan_data['colorRed']
            color_g = scan_data['colorGreen']
            color_b = scan_data['colorBlue']
        
    except Exception as e:
        print(f"✗ Error loading E57: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nFiltering...")
    
    # Create empty mask (all False initially)
    vehicle_mask = np.zeros(total_points, dtype=bool)
    
    for i, vehicle in enumerate(vehicles):
        # Convert bbox to angles
        az_min, az_max, el_min, el_max = bbox_to_spherical(
            vehicle['bbox'], image_width, image_height
        )
        
        # ---- auto azimuth alignment (panorama ↔ LiDAR) ----
        az_center = 0.5 * (az_min + az_max)
        pc_az_mean = np.mean(azimuth)
        az_offset = pc_az_mean - az_center

        az_min = (az_min + az_offset) % (2 * np.pi)
        az_max = (az_max + az_offset) % (2 * np.pi)

        # ---- dynamic AZ padding (bbox width–based) ----
        az_width = (az_max - az_min) % (2 * np.pi)
        AZ_PAD = np.clip(
            0.5 * az_width,
            np.deg2rad(1.0),
            np.deg2rad(4.0)
        )

        az_min = (az_min - AZ_PAD) % (2 * np.pi)
        az_max = (az_max + AZ_PAD) % (2 * np.pi)

        # ---- estimate distance of this vehicle ----
        rough_az_mask = az_in_range(azimuth, az_min, az_max)
        if np.any(rough_az_mask):
            mean_range = np.mean(range_data[rough_az_mask])
        else:
            mean_range = np.mean(range_data)

        # ---- dynamic EL padding based on distance ----
        EL_PAD = np.deg2rad(
            np.clip(
                0.15 * mean_range,   # 🔑 ključni faktor
                3.0,                 # min 3°
                15.0                 # max 15°
            )
        )

        # ---- final bbox mask ----
        bbox_mask = (
            az_in_range(azimuth, az_min, az_max) &
            (elevation >= el_min - EL_PAD) &
            (elevation <= el_max + np.deg2rad(2.0))
        )

        points_in_bbox = np.count_nonzero(bbox_mask)

        print(f"\nVEHICLE #{vehicle['id']} ({vehicle['type']}):")
        print(f"  az range: {az_min:.3f} – {az_max:.3f}")
        print(f"  el range: {el_min:.3f} – {el_max:.3f}")
        print(f"  mean range: {mean_range:.1f} m")
        print(f"  AZ pad: {np.rad2deg(AZ_PAD):.1f}°")
        print(f"  EL pad: {np.rad2deg(EL_PAD):.1f}°")
        print(f"  points: {points_in_bbox:,}")

        if points_in_bbox == 0:
            print("  ⚠️ NO POINTS FOUND")

        vehicle_mask |= bbox_mask

    
    total_vehicle_points = np.sum(vehicle_mask)
    print(f"\nTotal: {total_vehicle_points:,} / {total_points:,} points ({100*total_vehicle_points/total_points:.2f}%)")
    
    if total_vehicle_points == 0:
        print(f"✗ No points found!")
        return
    
    # Extract filtered data
    filtered_x = x[vehicle_mask]
    filtered_y = y[vehicle_mask]
    filtered_z = z[vehicle_mask]
    filtered_azimuth = azimuth[vehicle_mask]
    filtered_elevation = elevation[vehicle_mask]
    filtered_range = range_data[vehicle_mask]
    
    if has_color:
        filtered_r = color_r[vehicle_mask]
        filtered_g = color_g[vehicle_mask]
        filtered_b = color_b[vehicle_mask]
    
    print(f"Saving: {Path(output_path).name}")
    
    try:
        # Prepare data for writing
        filtered_data = {
            "cartesianX": filtered_x,
            "cartesianY": filtered_y,
            "cartesianZ": filtered_z,
            "sphericalAzimuth": filtered_azimuth,
            "sphericalElevation": filtered_elevation,
            "sphericalRange": filtered_range,
        }
        
        if has_color:
            filtered_data["colorRed"] = filtered_r
            filtered_data["colorGreen"] = filtered_g
            filtered_data["colorBlue"] = filtered_b
        
        # Create new E57 file
        output_e57 = pye57.E57(str(output_path), mode='w')
        output_e57.write_scan_raw(filtered_data)
        
        print(f"  E57: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
        
        # Also save as PCD
        pcd_path = output_path.parent / (output_path.stem + ".pcd")
        if has_color:
            save_as_pcd(pcd_path, filtered_x, filtered_y, filtered_z, filtered_r, filtered_g, filtered_b)
        else:
            save_as_pcd(pcd_path, filtered_x, filtered_y, filtered_z)
        
        print(f"  PCD: {pcd_path.stat().st_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"✗ Error saving: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"✓ Complete ({100*(1 - total_vehicle_points/total_points):.1f}% reduction)\n")


def main():
    """Main entry point."""
    
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    processing_dir = script_dir.parent
    iris_dir = processing_dir.parent
    
    input_dir = processing_dir / "data" / "downsampled"
    detection_dir = processing_dir / "data" / "detected"
    output_dir = processing_dir / "data" / "filtered"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find files
    e57_files = sorted(input_dir.glob("*.e57"))
    detection_files = sorted(detection_dir.glob("*_panorama_detections.json"))
    
    if not e57_files:
        print(f"✗ No E57 files found in {input_dir}")
        return
    
    if not detection_files:
        print(f"✗ No detection files found in {detection_dir}")
        return
    
    print(f"\nFound {len(e57_files)} E57 file(s)")
    
    # Process each E57 file
    for e57_path in e57_files:
        # Find matching detection file
        detection_path = None
        for det_file in detection_files:
            # Match by base name (e.g., scan001_downsampled_5cm.e57 -> scan001_panorama_detections.json)
            base_name = e57_path.name.split('_downsampled')[0] if '_downsampled' in e57_path.name else e57_path.stem
            if det_file.name.startswith(base_name + "_panorama"):
                detection_path = det_file
                break
        
        if not detection_path:
            print(f"✗ No detection file for {e57_path.name}")
            continue
        
        # Generate output filename
        output_filename = f"{e57_path.stem}_vehicles_only.e57"
        output_path = output_dir / output_filename
        
        # Check if already processed
        if output_path.exists():
            print(f"⚠ Skipping {e57_path.name} (already processed)")
            continue
        
        # Run filtering
        filter_points_by_vehicle_bboxes(e57_path, detection_path, output_path)
    
    print(f"\nAll done!")


if __name__ == "__main__":
    main()
