#!/usr/bin/env python3
"""
E57 Point Cloud Analysis Tool

Used during development to understand E57 file structure and validate
the vehicle filtering pipeline. Extracts and displays:
  - Point cloud fields and statistics (cartesian/spherical coords, RGB)
  - Panorama dimensions and projection parameters
  - Vehicle detections from JSON files
  - Bbox-to-spherical angle conversion formulas
  - Sample filtering calculation to verify logic

Outputs a JSON file with all extracted metadata for debugging.
"""

import pye57
import numpy as np
import json
from pathlib import Path


def get_int_value(node):
    """Extract integer from E57 node (handles both value() and direct access)."""
    try:
        return int(node.value())
    except:
        try:
            return int(node)
        except:
            return None


def get_float_value(node):
    """Extract float from E57 node (handles both value() and direct access)."""
    try:
        return float(node.value())
    except:
        try:
            return float(node)
        except:
            return None


def analyze_e57_data():
    """Analyze E57 file and extract all panorama filtering parameters."""
    
    print("\n" + "="*70)
    print("E57 Analysis - Structure & Filtering Validation")
    print("="*70)
    
    script_dir = Path(__file__).parent.resolve()
    iris_dir = script_dir.parent
    input_dir = iris_dir / "e57_panorama_vehicle_detection" / "data" / "input"
    detection_dir = iris_dir / "e57_panorama_vehicle_detection" / "data" / "detected"
    
    print(f"\nInput: {input_dir}")
    print(f"Detections: {detection_dir}")
    
    # Find first E57 file
    e57_files = list(input_dir.glob("*.e57"))
    if not e57_files:
        print("ERROR: No E57 files found")
        return None
    
    e57_path = e57_files[0]
    print(f"\nAnalyzing: {e57_path.name}")
    
    try:
        e57 = pye57.E57(str(e57_path))
    except Exception as e:
        print(f"Failed to load: {e}")
        return None
    
    root = e57.root
    analysis_data = {}
    
    # Part 1: Point Cloud Data
    print(f"\n{'-'*70}")
    print("Point Cloud Data")
    print(f"{'-'*70}")
    
    try:
        scan_data = e57.read_scan(0, ignore_missing_fields=True)
        
        fields = list(scan_data.keys())
        print(f"Available fields ({len(fields)}):")
        for f in sorted(fields):
            print(f"  - {f}")
        
        x = scan_data['cartesianX']
        y = scan_data['cartesianY']
        z = scan_data['cartesianZ']
        az = scan_data['sphericalAzimuth']
        el = scan_data['sphericalElevation']
        rng = scan_data['sphericalRange']
        
        print(f"\nPoint Cloud Stats:")
        print(f"  Points: {len(x):,}")
        print(f"  X: [{x.min():.2f}, {x.max():.2f}] m")
        print(f"  Y: [{y.min():.2f}, {y.max():.2f}] m")
        print(f"  Z: [{z.min():.2f}, {z.max():.2f}] m")
        
        print(f"\nSpherical Coordinates:")
        print(f"  Azimuth: [{np.degrees(az.min()):.1f}, {np.degrees(az.max()):.1f}] deg")
        print(f"  Elevation: [{np.degrees(el.min()):.1f}, {np.degrees(el.max()):.1f}] deg")
        print(f"  Range: [{rng.min():.2f}, {rng.max():.2f}] m")
        
        analysis_data['point_cloud'] = {
            'total_points': int(len(x)),
            'x_range': [float(x.min()), float(x.max())],
            'y_range': [float(y.min()), float(y.max())],
            'z_range': [float(z.min()), float(z.max())],
            'azimuth_rad': [float(az.min()), float(az.max())],
            'elevation_rad': [float(el.min()), float(el.max())],
            'distance_m': [float(rng.min()), float(rng.max())]
        }
        
    except Exception as e:
        print(f"Error reading point cloud: {e}")
        return None
    
    # Part 2: Panorama parameters
    print(f"\n{'-'*70}")
    print("Panorama Parameters")
    print(f"{'-'*70}")
    
    panorama_info = {
        "width": None,
        "height": None,
        "projection": "spherical"
    }
    
    try:
        images2d = root["images2D"]
        image = images2d[0]
        rep = image["sphericalRepresentation"]
        
        print("Found spherical representation")
        
        # Get image dimensions
        try:
            width_node = rep["imageWidth"]
            width = get_int_value(width_node)
            print(f"  Width: {width} px")
            panorama_info["width"] = width
        except Exception as e:
            print(f"  Width: failed ({type(e).__name__})")
        
        try:
            height_node = rep["imageHeight"]
            height = get_int_value(height_node)
            print(f"  Height: {height} px")
            panorama_info["height"] = height
        except Exception as e:
            print(f"  Height: failed ({type(e).__name__})")
        
        # Check for JPEG data
        try:
            jpeg = rep["jpegImage"]
            size = jpeg.byteCount()
            print(f"  JPEG: {size:,} bytes")
        except:
            print("  JPEG: not found")
        
        analysis_data['panorama'] = panorama_info
        
    except Exception as e:
        print(f"Error reading panorama: {e}")
    
    # Part 3: Load vehicle detections
    print(f"\n{'-'*70}")
    print("Vehicle Detections")
    print(f"{'-'*70}")
    
    detection_files = list(detection_dir.glob("*_detections.json"))
    
    if detection_files:
        detection_file = detection_files[0]
        print(f"Found: {detection_file.name}")
        
        try:
            with open(detection_file, 'r') as f:
                detections = json.load(f)
            
            print(f"\nDetection Info:")
            print(f"  Panorama: {detections.get('panorama')}")
            print(f"  Dimensions: {detections.get('dimensions')}")
            
            vehicles = detections.get('vehicles', [])
            print(f"  Vehicles: {len(vehicles)}")
            
            if vehicles:
                print(f"\nFirst 3 vehicles:")
                for i, v in enumerate(vehicles[:3]):
                    bbox = v['bbox']
                    conf = v.get('confidence', 0)
                    vtype = v.get('type', 'unknown')
                    print(f"  {i+1}. {vtype} (conf: {conf:.1%})")
                    print(f"     bbox: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) -> ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
                
                if len(vehicles) > 3:
                    print(f"  ... and {len(vehicles)-3} more")
            
            analysis_data['detections'] = {
                'file': detection_file.name,
                'total_vehicles': len(vehicles),
                'detections': detections
            }
            
        except Exception as e:
            print(f"Error loading detections: {e}")
    else:
        print(f"No detection files found")
    
    # Part 4: Mapping formula
    print(f"\n{'-'*70}")
    print("Pixel to 3D Mapping Formula")
    print(f"{'-'*70}")
    
    if panorama_info["width"] and panorama_info["height"]:
        iw = panorama_info["width"]
        ih = panorama_info["height"]
        
        print(f"""
Spherical Panorama Mapping:
  Image size: {iw} × {ih} pixels
  Projection: Spherical (360° × 180°)

Conversion formulas:
  azimuth(rad) = (pixel_x / {iw}) × 2π
  elevation(rad) = (pixel_y / {ih}) × π - π/2

Angle ranges:
  Azimuth: [0, 2π) rad = [0°, 360°) (horizontal)
  Elevation: [-π/2, π/2] = [-90°, 90°] (vertical)

Filtering algorithm:
  1. Convert bbox to angle ranges:
     az_min = (x1 / {iw}) × 2π
     az_max = (x2 / {iw}) × 2π
     el_min = (y1 / {ih}) × π - π/2
     el_max = (y2 / {ih}) × π - π/2
  
  2. Filter points from point cloud:
     mask = (azimuth >= az_min) & (azimuth <= az_max) &
            (elevation >= el_min) & (elevation <= el_max)
  
  3. Save filtered points to new E57 file
""")
        
        analysis_data['mapping'] = {
            'width': iw,
            'height': ih,
            'projection': 'spherical'
        }
    else:
        print("Could not extract image dimensions")
    
    # Part 5: Sample calculation
    print(f"\n{'-'*70}")
    print("Sample Calculation")
    print(f"{'-'*70}")
    
    if (panorama_info["width"] and 
        panorama_info["height"] and 
        'detections' in analysis_data and 
        analysis_data['detections']['total_vehicles'] > 0):
        
        iw = panorama_info["width"]
        ih = panorama_info["height"]
        first_vehicle = analysis_data['detections']['detections']['vehicles'][0]
        bbox = first_vehicle['bbox']
        
        # Calculate angle ranges for first vehicle
        az_min = (bbox['x1'] / iw) * 2 * np.pi
        az_max = (bbox['x2'] / iw) * 2 * np.pi
        el_min = (bbox['y1'] / ih) * np.pi - np.pi/2
        el_max = (bbox['y2'] / ih) * np.pi - np.pi/2
        
        # Count points in this region
        az = scan_data['sphericalAzimuth']
        el = scan_data['sphericalElevation']
        
        mask = (az >= az_min) & (az <= az_max) & (el >= el_min) & (el <= el_max)
        n_points = np.sum(mask)
        
        print(f"\nFirst vehicle:")
        print(f"  Bbox: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) -> ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
        print(f"  Azimuth: [{np.degrees(az_min):.1f}, {np.degrees(az_max):.1f}] deg")
        print(f"  Elevation: [{np.degrees(el_min):.1f}, {np.degrees(el_max):.1f}] deg")
        print(f"  Points: {n_points:,} / {len(x):,} ({100*n_points/len(x):.2f}%)")
        
        analysis_data['sample'] = {
            'bbox': [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']],
            'azimuth_deg': [float(np.degrees(az_min)), float(np.degrees(az_max))],
            'elevation_deg': [float(np.degrees(el_min)), float(np.degrees(el_max))],
            'points': int(n_points)
        }
    
    # Save analysis to JSON
    output_file = script_dir / "e57_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"\n{'-'*70}")
    print("Analysis Complete")
    print(f"{'-'*70}")
    print(f"Saved to: {output_file}")
    
    return analysis_data


if __name__ == "__main__":
    analyze_e57_data()
