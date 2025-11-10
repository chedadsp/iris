import numpy as np
from pathlib import Path
import pye57
import struct

def read_e57_file(e57_path):
    """Read E57 file and extract point cloud data"""
    print(f"    Reading E57 file...")
    e57 = pye57.E57(e57_path)
    
    n_scans = e57.scan_count
    print(f"    Found {n_scans} scan(s) in file")
    
    data = e57.read_scan(0, intensity=True, colors=False, ignore_missing_fields=True)
    
    x = data["cartesianX"]
    y = data["cartesianY"] 
    z = data["cartesianZ"]
    
    if "intensity" in data:
        intensity = data["intensity"]
    else:
        intensity = np.zeros(len(x))
    
    points = np.column_stack([x, y, z, intensity])
    
    return points

def save_as_pcd_xtreme1(points, output_path):
    """Save points as BINARY PCD file with RGB uint32 format for Xtreme1"""
    n_points = len(points)
    
    # Normalize intensity to 0-255 and create grayscale RGB
    intensity = points[:, 3]
    intensity_norm = ((intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-10) * 255).astype(np.uint8)
    
    # Pack RGB into uint32: (R << 16) | (G << 8) | B
    r = intensity_norm.astype(np.uint32)
    g = intensity_norm.astype(np.uint32)
    b = intensity_norm.astype(np.uint32)
    rgb_packed = (r << 16) | (g << 8) | b
    
    # Write PCD file in BINARY format
    with open(output_path, 'wb') as f:
        # Header (ASCII)
        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z rgb\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F U\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {n_points}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n_points}\n"
            "DATA binary\n"
        )
        f.write(header.encode('ascii'))
        
        # Data (BINARY)
        for i in range(n_points):
            x, y, z = points[i, :3]
            rgb = rgb_packed[i]
            # Pack as: float32, float32, float32, uint32
            f.write(struct.pack('fffI', x, y, z, rgb))

def convert_e57_to_pcd(input_dir, output_dir, xtreme1_format=True):
    """Convert all E57 files in input directory to PCD format"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"E57 to PCD Converter")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Format: {'Xtreme1 (RGB uint32 BINARY)' if xtreme1_format else 'Standard'}")
    print(f"{'='*70}\n")
    
    e57_files = sorted(input_path.glob("*.e57"))
    
    if not e57_files:
        print(f"❌ No .e57 files found in {input_dir}")
        return []
    
    print(f"Found {len(e57_files)} E57 file(s)\n")
    
    results = []
    successful = 0
    
    for idx, e57_file in enumerate(e57_files, 1):
        print(f"[{idx}/{len(e57_files)}] Processing: {e57_file.name}")
        
        try:
            points = read_e57_file(str(e57_file))
            print(f"    Loaded {len(points):,} points")
            
            output_filename = e57_file.stem + ".pcd"
            output_pcd = output_path / output_filename
            
            print(f"    Saving to {output_filename}...")
            save_as_pcd_xtreme1(points, output_pcd)
            
            file_size = output_pcd.stat().st_size / (1024 * 1024)
            print(f"    ✓ Saved ({file_size:.2f} MB)\n")
            
            results.append({
                "success": True,
                "original_file": e57_file.name,
                "output_file": output_filename,
                "output_path": str(output_pcd),
                "num_points": len(points),
                "file_size_mb": round(file_size, 2)
            })
            
            successful += 1
            
        except Exception as e:
            print(f"    ✗ Error: {e}\n")
            results.append({
                "success": False,
                "original_file": e57_file.name,
                "error": str(e)
            })
    
    print(f"{'='*70}")
    print(f"✓ Converted: {successful}/{len(e57_files)} files")
    print(f"{'='*70}\n")
    
    return results

if __name__ == "__main__":
    INPUT_DIR = "data/ns-scans/14 auto scan"
    OUTPUT_DIR = "data/pcd-scans"
    
    convert_e57_to_pcd(INPUT_DIR, OUTPUT_DIR, xtreme1_format=True)