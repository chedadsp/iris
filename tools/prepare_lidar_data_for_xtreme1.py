import json
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
from e57_to_pcd import convert_e57_to_pcd

def create_xtreme1_structure(pcd_files_info, output_path):
    """Create Xtreme1 folder structure from PCD files"""
    output_path = Path(output_path)
    pcd_dir = output_path / "lidar_point_cloud_0"
    pcd_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "dataset_name": "auto_scan_14",
        "total_frames": 0,
        "frames": []
    }
    
    print(f"\nCreating Xtreme1 structure in: {output_path}\n")
    
    successful_files = [f for f in pcd_files_info if f["success"]]
    
    for idx, file_info in enumerate(successful_files, 1):
        frame_name = f"frame_{idx:06d}.pcd"
        source_pcd = Path(file_info["output_path"])
        dest_pcd = pcd_dir / frame_name
        
        shutil.copy2(source_pcd, dest_pcd)
        print(f"  [{idx}/{len(successful_files)}] {frame_name}")
        
        metadata["frames"].append({
            "frame_id": idx,
            "file_name": frame_name,
            "original_file": file_info["original_file"]
        })
    
    metadata["total_frames"] = len(successful_files)
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def create_zip_archive(source_dir, zip_path):
    """Create ZIP archive of the dataset"""
    print(f"\nCreating ZIP archive...")
    source_path = Path(source_dir)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(source_path)
                zipf.write(file_path, arcname)
    
    file_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"✓ ZIP created: {zip_path.name} ({file_size:.2f} MB)")

def prepare_xtreme1_dataset(e57_input_dir, output_base_dir):
    """Convert E57 to PCD and prepare Xtreme1 package"""
    print(f"{'='*70}")
    print(f"Xtreme1 Dataset Preparation")
    print(f"{'='*70}\n")
    
    # Step 1: Convert E57 to PCD
    temp_pcd_dir = Path(output_base_dir) / "temp_pcd_conversion"
    pcd_results = convert_e57_to_pcd(e57_input_dir, temp_pcd_dir, xtreme1_format=True)
    
    if not any(r["success"] for r in pcd_results):
        print("❌ No files converted. Aborting.")
        if temp_pcd_dir.exists():
            shutil.rmtree(temp_pcd_dir)
        return
    
    # Step 2: Create Xtreme1 structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"auto_scan_14_xtreme1_{timestamp}"
    xtreme1_dir = Path(output_base_dir) / dataset_name
    
    metadata = create_xtreme1_structure(pcd_results, xtreme1_dir)
    
    # Step 3: Create ZIP
    zip_path = Path(output_base_dir) / f"{dataset_name}.zip"
    create_zip_archive(xtreme1_dir, zip_path)
    
    # Step 4: Cleanup
    print(f"\nCleaning up...")
    if temp_pcd_dir.exists():
        shutil.rmtree(temp_pcd_dir)
        print(f"✓ Removed: {temp_pcd_dir}")
    if xtreme1_dir.exists():
        shutil.rmtree(xtreme1_dir)
        print(f"✓ Removed: {xtreme1_dir}")
    
    print(f"\n{'='*70}")
    print(f"✓ COMPLETED")
    print(f"ZIP: {zip_path.name}")
    print(f"Frames: {metadata['total_frames']}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    E57_INPUT_DIR = "data/ns-scans/14 auto scan"
    OUTPUT_BASE_DIR = "data"
    
    prepare_xtreme1_dataset(E57_INPUT_DIR, OUTPUT_BASE_DIR)