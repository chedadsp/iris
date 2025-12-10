import os
import sys
import subprocess
import argparse
import json
from datetime import datetime
import numpy as np

# Default CloudCompare path (override with --cc-path)
DEFAULT_CC_PATH = r"C:\Program Files\CloudCompare\CloudCompare.exe"


def parse_args():
    p = argparse.ArgumentParser(description="Downsample and optionally SOR-filter .e57 files using CloudCompare CLI")
    p.add_argument("input", help="Input .e57 file")
    p.add_argument("--voxel-size", "-v", type=float, required=True, help="Voxel size in meters (e.g. 0.05 for 5 cm)")
    p.add_argument("--sor", action="store_true", help="Apply Statistical Outlier Removal (SOR)")
    p.add_argument("--sor-k", type=int, default=20, help="SOR: number of neighbours (default: 20)")
    p.add_argument("--sor-std", type=float, default=1.0, help="SOR: std dev multiplier (default: 1.0)")
    p.add_argument("--out-dir", "-o", default=os.path.join("data", "lidar_preprocessing"), help="Base output directory")
    p.add_argument("--cc-path", default=DEFAULT_CC_PATH, help="Path to CloudCompare executable")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--no-save", action="store_true", help="Run processing but do not save output (for dry-run)")
    p.add_argument("--validate-e57", action="store_true", help="Validate output E57 checksum using pye57 (requires pye57 installed)")
    p.add_argument("--no-silent", action="store_true", help="Run CloudCompare without -SILENT flag to see detailed logs")
    p.add_argument("--out-format", choices=["e57", "pcd", "rosbag"], default="pcd", help="Output format: pcd (default, no checksum issues), e57, or rosbag (ROS Bag)")
    return p.parse_args()


def voxel_label_from_size(voxel_m):
    """Convert voxel size to label string (e.g. 0.05 -> '5cm')."""
    cm = int(round(voxel_m * 100))
    if cm > 0 and cm < 1000:
        return f"{cm}cm"
    return f"{voxel_m}m"


def build_cc_command(cc_path, input_file, voxel, sor, sor_k, sor_std, output_file, no_silent=False):
    """Build CloudCompare command for downsampling and optional SOR filtering."""
    cmd = [cc_path]
    if not no_silent:
        cmd.append("-SILENT")
    cmd.extend(["-NO_TIMESTAMP", "-O", input_file])
    if sor:
        cmd.extend(["-SOR", str(sor_k), str(sor_std)])
    # spatial subsample
    cmd.extend(["-SS", "SPATIAL", str(voxel)])
    if not output_file:
        return cmd
    cmd.extend(["-SAVE_CLOUDS", "FILE", output_file])
    return cmd


def validate_e57_checksum(filepath):
    """
    Attempt to validate E57 checksum using pye57.
    Returns (is_valid, error_msg, point_count).
    """
    try:
        import pye57
    except ImportError:
        return (None, "pye57 not installed; cannot validate", None)
    
    try:
        e57 = pye57.E57(filepath)
        total_points = 0
        for scan in e57.scans:
            total_points += len(scan.read_points())
        e57.close()
        return (True, "checksum valid", total_points)
    except Exception as e:
        error_str = str(e)
        if "checksum" in error_str.lower():
            return (False, f"E57 checksum validation failed: {error_str}", None)
        else:
            return (False, f"E57 read error: {error_str}", None)


def pcd_to_rosbag(pcd_file, rosbag_file, frame_id="lidar"):
    """Convert PCD file to ROS Bag format using rosbag library."""
    try:
        import rosbag
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
        import sensor_msgs.point_cloud2 as pc2
    except ImportError as e:
        return (False, f"rosbag library not installed: {e}. Install with: pip install rosbag")
    
    try:
        # Read PCD using Open3D
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(pcd_file)
            points = np.asarray(pcd.points)
        except ImportError:
            # Fallback: read PCD manually (simple ASCII format)
            points = read_pcd_simple(pcd_file)
        
        if points is None or len(points) == 0:
            return (False, f"No points read from PCD file {pcd_file}")
        
        # Create ROS Bag and PointCloud2 message
        with rosbag.Bag(rosbag_file, 'w') as bag:
            header = Header()
            header.frame_id = frame_id
            header.stamp = rosbag.rostime.Time.from_sec(0)
            
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]
            
            cloud_msg = pc2.create_cloud(header, fields, points)
            bag.write('/lidar/points', cloud_msg)
        
        return (True, f"Successfully converted {len(points)} points to ROS Bag")
    
    except Exception as e:
        return (False, f"Error converting PCD to ROS Bag: {str(e)}")


def read_pcd_simple(pcd_file):
    """Simple PCD reader for ASCII format (fallback if open3d not available)."""
    try:
        with open(pcd_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('DATA'):
                data_start = i + 1
                break
        
        if data_start == 0:
            return None
        
        points = []
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
                except ValueError:
                    continue
        
        return np.array(points) if points else None
    except Exception:
        return None


def main():
    args = parse_args()

    input_file = args.input
    voxel = args.voxel_size
    sor = args.sor
    sor_k = args.sor_k
    sor_std = args.sor_std
    out_base = args.out_dir
    cc_path = args.cc_path
    out_format = args.out_format

    if not os.path.isfile(input_file):
        print(f"ERROR: input file not found: {input_file}")
        sys.exit(2)

    if not os.path.isfile(cc_path):
        print(f"WARNING: CloudCompare executable not found at {cc_path}. You may need to set --cc-path to the correct path.")

    label = voxel_label_from_size(voxel)

    output_subfolder = f"downsampled_e57_{label}"
    output_folder = os.path.join(out_base, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Determine intermediate and final output formats
    if out_format == "rosbag":
        # For ROS Bag, we first create PCD, then convert to ROS Bag
        intermediate_file = os.path.join(output_folder, f"{base_name}_downsampled_{label}.pcd")
        output_file = os.path.join(output_folder, f"{base_name}_downsampled_{label}.bag")
    else:
        intermediate_file = None
        output_file = os.path.join(output_folder, f"{base_name}_downsampled_{label}.{out_format}")
    
    metadata_file = os.path.join(output_folder, f"{base_name}_downsampled_{label}.json")

    if os.path.exists(output_file) and not args.overwrite:
        print(f"ERROR: output file already exists: {output_file} (use --overwrite to replace)")
        sys.exit(3)

    # Build CloudCompare command with intermediate output if needed
    cc_output = intermediate_file if intermediate_file else output_file
    cmd = build_cc_command(cc_path, input_file, voxel, sor, sor_k, sor_std, None if args.no_save else cc_output, no_silent=args.no_silent)

    print("\nRunning CloudCompare (constructed command):")
    print(" ".join([f'\"{c}\"' if ' ' in c else c for c in cmd]))
    print(f"Input : {input_file}")
    print(f"Voxel : {voxel} m ({label})")
    print(f"SOR   : {'enabled' if sor else 'disabled'}")
    print(f"Output format: {out_format}")
    print(f"Output folder: {output_folder}\n")

    try:
        res = subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"ERROR: Failed to run CloudCompare: {e}")
        sys.exit(4)

    metadata = {
        "source_file": os.path.abspath(input_file),
        "output_file": os.path.abspath(output_file) if not args.no_save else None,
        "output_format": out_format,
        "voxel_size_m": voxel,
        "voxel_label": label,
        "sor_used": sor,
        "sor_k": sor_k if sor else None,
        "sor_std": sor_std if sor else None,
        "cc_returncode": res.returncode,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    if res.returncode != 0:
        print(f"CloudCompare returned non-zero exit code: {res.returncode}")
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except:
            pass
        print(f"Metadata written to: {metadata_file}")
        sys.exit(res.returncode)

    if not args.no_save:
        # Handle ROS Bag conversion if needed
        if out_format == "rosbag" and intermediate_file and os.path.exists(intermediate_file):
            print(f"Converting PCD to ROS Bag...")
            success, msg = pcd_to_rosbag(intermediate_file, output_file)
            metadata["rosbag_conversion"] = {"success": success, "message": msg}
            print(f"{'✓' if success else '✗'} {msg}")
            if success:
                try:
                    os.remove(intermediate_file)
                except:
                    pass
            else:
                print(f"WARNING: ROS Bag conversion failed. PCD file remains at: {intermediate_file}")
                sys.exit(5)

        # Validate E57 if requested
        if args.validate_e57 and out_format == "e57" and os.path.exists(output_file):
            is_valid, msg, pt_count = validate_e57_checksum(output_file)
            metadata["validation"] = {
                "is_valid": is_valid,
                "message": msg,
                "point_count": pt_count
            }
            if is_valid is False:
                print(f"WARNING: E57 validation failed: {msg}")
                print("Note: CloudCompare E57 writer has checksum issues. Consider using PCD or ROS Bag format instead.")
            elif is_valid is True:
                print(f"✓ E57 validation passed. Point count: {pt_count}")

        if os.path.exists(output_file):
            print("\n✔ Finished! File saved to:")
            print(output_file)
            print(f"Metadata written to: {metadata_file}")
            
            # Save metadata
            try:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"WARNING: could not write metadata file: {e}")
            
            sys.exit(0)
        else:
            print("ERROR: CloudCompare returned success but output file not found.")
            print(f"Check CloudCompare logs. Metadata: {metadata_file}")
            try:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            except:
                pass
            sys.exit(5)
    else:
        print("Dry run complete (no output saved).")
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except:
            pass
        sys.exit(0)


if __name__ == '__main__':
    main()
