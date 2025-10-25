import numpy as np
from pathlib import Path
import pye57
import open3d as o3d

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

def save_as_pcd(points, output_path, binary=False):
    """Save points as PCD file using Open3D"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if points.shape[1] >= 4:
        intensity_norm = (points[:, 3] - points[:, 3].min()) / (points[:, 3].max() - points[:, 3].min() + 1e-10)
        colors = np.column_stack([intensity_norm, intensity_norm, intensity_norm])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=not binary)

def convert_e57_to_pcd(input_dir, output_dir, binary=False):
    """Convert all E57 files in input directory to PCD format"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print(f"E57 to PCD Converter")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
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
            save_as_pcd(points, output_pcd, binary=binary)
            
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
    
    convert_e57_to_pcd(INPUT_DIR, OUTPUT_DIR, binary=False)