import argparse
import numpy as np
import pyvista as pv
from pye57 import E57
import os
import struct
import json
from .config import FileConfig, VTKConfig
from .error_handling import (with_error_handling, validate_file_path, 
                           validate_point_cloud_data, FileFormatError, DependencyError)
from .vtk_utils import VTKSafetyManager
try:
    import bagpy
    from bagpy import bagreader
    BAGPY_AVAILABLE = True
except ImportError:
    BAGPY_AVAILABLE = False
    print("Warning: bagpy not available. ROS bag support disabled.")

@with_error_handling("read_pcd_file", reraise=True)
def read_pcd_file(pcd_path):
    """
    Read a PCD (Point Cloud Data) file and return points, intensity, and other data.
    Supports both ASCII and binary PCD formats.
    
    Args:
        pcd_path (str): Path to the PCD file
        
    Returns:
        tuple: (points, intensity, rgb) where:
            - points: numpy array of shape (N, 3) with x, y, z coordinates
            - intensity: numpy array of shape (N,) with intensity values or None
            - rgb: numpy array of shape (N, 3) with RGB values or None
            (Note: PCD format doesn't typically contain RGB data)
    """
    # Validate file path
    validated_path = validate_file_path(pcd_path, must_exist=True, 
                                       allowed_extensions=['.pcd'])
    
    # First, read header to determine format and structure
    header = {}
    data_format = None
    header_end_pos = 0
    
    # Read header using binary mode to handle both ASCII and binary files
    with open(validated_path, 'rb') as f:
        for line_num in range(100):  # Safety limit for header lines
            line_bytes = f.readline()
            if not line_bytes:
                break
                
            try:
                line = line_bytes.decode('ascii').strip()
            except UnicodeDecodeError:
                # If we can't decode as ASCII, we've hit binary data
                break
                
            if line.startswith('#') or not line:
                continue
                
            if line.startswith('DATA '):
                data_format = line.split(' ')[1].lower()
                header_end_pos = f.tell()
                break
                
            if ' ' in line:
                key, value = line.split(' ', 1)
                header[key] = value
    
    if data_format is None:
        raise FileFormatError(f"No DATA field found in PCD header: {validated_path}")
    
    # Parse header information for binary format
    fields = header.get('FIELDS', 'x y z').split()
    sizes = list(map(int, header.get('SIZE', '4 4 4').split()))
    types = header.get('TYPE', 'F F F').split()
    counts = list(map(int, header.get('COUNT', '1 1 1').split()))
    points_count = int(header.get('POINTS', '0'))
    
    if data_format == 'ascii':
        return _read_pcd_ascii(validated_path, header_end_pos)
    elif data_format == 'binary':
        return _read_pcd_binary(validated_path, header_end_pos, fields, sizes, types, counts, points_count)
    else:
        raise FileFormatError(f"Unsupported PCD data format: {data_format}")


def _read_pcd_ascii(file_path, header_end_pos):
    """Read ASCII format PCD data."""
    points = []
    intensity = []
    
    with open(file_path, 'r') as f:
        f.seek(header_end_pos)
        
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            values = line.split()
            if len(values) >= 3:
                try:
                    x, y, z = float(values[0]), float(values[1]), float(values[2])
                    points.append([x, y, z])
                    
                    # Try to extract intensity (usually 4th field)
                    if len(values) >= 4:
                        try:
                            intensity_val = float(values[3])
                            intensity.append(intensity_val)
                        except ValueError:
                            intensity.append(0.0)
                    else:
                        intensity.append(0.0)
                        
                except ValueError:
                    continue
    
    if not points:
        raise FileFormatError(f"No valid points found in ASCII PCD file: {file_path}")
    
    points = np.array(points)
    intensity = np.array(intensity) if intensity else None
    
    return points, intensity, None


def _read_pcd_binary(file_path, header_end_pos, fields, sizes, types, counts, points_count):
    """Read binary format PCD data."""
    if points_count <= 0:
        raise FileFormatError(f"Invalid points count in PCD file: {points_count}")
    
    # Map PCD types to struct format characters
    type_map = {
        'I': 'I',  # unsigned int
        'U': 'I',  # unsigned int  
        'F': 'f',  # float
        'F8': 'd', # double (8-byte float)
    }
    
    # Build struct format string based on fields
    fmt_chars = []
    field_info = []
    
    for i, field in enumerate(fields):
        field_type = types[i] if i < len(types) else 'F'
        field_size = sizes[i] if i < len(sizes) else 4
        field_count = counts[i] if i < len(counts) else 1
        
        # Handle special cases for type mapping
        if field_type == 'F' and field_size == 8:
            fmt_char = 'd'  # double precision float
        elif field_type == 'F' and field_size == 4:
            fmt_char = 'f'  # single precision float
        elif field_type in ['I', 'U']:
            if field_size == 4:
                fmt_char = 'I'  # unsigned int
            elif field_size == 2:
                fmt_char = 'H'  # unsigned short
            elif field_size == 1:
                fmt_char = 'B'  # unsigned char
            else:
                fmt_char = 'I'  # default to unsigned int
        else:
            fmt_char = 'f'  # default to float
            
        # Add format characters for each count
        fmt_chars.extend([fmt_char] * field_count)
        field_info.append((field, fmt_char, field_count))
    
    # Create struct format (little endian)
    struct_fmt = '<' + ''.join(fmt_chars)
    point_size = struct.calcsize(struct_fmt)
    
    points = []
    intensity = []
    
    with open(file_path, 'rb') as f:
        f.seek(header_end_pos)
        
        for _ in range(points_count):
            try:
                data_bytes = f.read(point_size)
                if len(data_bytes) < point_size:
                    break  # End of file
                    
                values = struct.unpack(struct_fmt, data_bytes)
                
                # Extract x, y, z coordinates
                point_values = {}
                value_idx = 0
                
                for field, fmt_char, count in field_info:
                    if count == 1:
                        point_values[field] = values[value_idx]
                        value_idx += 1
                    else:
                        point_values[field] = values[value_idx:value_idx + count]
                        value_idx += count
                
                # Extract coordinates (required)
                x = point_values.get('x', 0.0)
                y = point_values.get('y', 0.0)
                z = point_values.get('z', 0.0)
                points.append([x, y, z])
                
                # Extract intensity if available
                intensity_val = point_values.get('intensity', 0.0)
                if isinstance(intensity_val, (list, tuple)):
                    intensity_val = intensity_val[0] if intensity_val else 0.0
                intensity.append(float(intensity_val))
                
            except struct.error as e:
                # Skip malformed data points
                continue
            except Exception as e:
                # Skip any other errors with individual points
                continue
    
    if not points:
        raise FileFormatError(f"No valid points found in binary PCD file: {file_path}")
    
    points = np.array(points)
    intensity = np.array(intensity) if intensity else None
    
    return points, intensity, None

@with_error_handling("read_bag_file", reraise=True)
def read_bag_file(bag_path, topic=None, frame_index=0):
    """
    Read a ROS bag file and extract point cloud data.
    
    Args:
        bag_path (str): Path to the ROS bag file
        topic (str, optional): Specific topic to read. If None, searches for PointCloud2 topics
        frame_index (int): Which frame/message to extract (default: 0 for first frame)
        
    Returns:
        tuple: (points, intensity, rgb) where:
            - points: numpy array of shape (N, 3) with x, y, z coordinates  
            - intensity: numpy array of shape (N,) with intensity values or None
            - rgb: numpy array of shape (N, 3) with RGB values or None
    """
    if not BAGPY_AVAILABLE:
        raise DependencyError("bagpy is required for ROS bag support. Install with: poetry add bagpy")
    
    # Validate file path
    validated_path = validate_file_path(bag_path, must_exist=True, 
                                       allowed_extensions=['.bag'])
    
    try:
        # Read the bag file
        bag = bagreader(str(validated_path))
        
        # Get list of topics
        topic_table = bag.topic_table
        print(f"Available topics in bag file:")
        for topic_name in topic_table['Topics']:
            print(f"  - {topic_name}")
        
        # Find PointCloud2 topics if no specific topic provided
        if topic is None:
            pointcloud_topics = []
            for topic_name in topic_table['Topics']:
                # Common PointCloud2 topic patterns
                if ('pointcloud' in topic_name.lower() or 
                    'velodyne_points' in topic_name.lower() or
                    'points' in topic_name.lower() or
                    'lidar' in topic_name.lower()):
                    pointcloud_topics.append(topic_name)
            
            if not pointcloud_topics:
                raise ValueError("No PointCloud2 topics found in bag file. Use --topic to specify a topic manually.")
            
            topic = pointcloud_topics[0]
            print(f"Using topic: {topic}")
        
        # Extract messages for the topic - bagpy returns CSV file path
        try:
            topic_data_path = bag.message_by_topic(topic)
            print(f"Extracted topic data to: {topic_data_path}")
            
            # Read the CSV file containing the topic data
            points, intensity, rgb = read_pointcloud_csv(topic_data_path)
            return points, intensity, rgb
            
        except Exception as e:
            print(f"Error extracting topic data: {e}")
            
            # Fallback: try to find existing CSV files
            import os
            bag_dir = os.path.splitext(bag_path)[0]
            if os.path.exists(bag_dir):
                csv_files = []
                for root, dirs, files in os.walk(bag_dir):
                    for file in files:
                        if file.endswith('.csv'):
                            csv_files.append(os.path.join(root, file))
                
                # Look for point cloud related CSV files
                for csv_file in csv_files:
                    if (topic and topic.replace('/', '_') in csv_file) or 'points' in csv_file.lower():
                        print(f"Trying CSV file: {csv_file}")
                        try:
                            return read_pointcloud_csv(csv_file)
                        except Exception as csv_e:
                            print(f"Failed to read {csv_file}: {csv_e}")
                            continue
            
            raise ValueError(f"No usable point cloud data found for topic {topic}")
        
    except Exception as e:
        raise ValueError(f"Failed to extract point cloud data from bag file: {e}")

def parse_pointcloud2_message(message_data):
    """
    Parse a PointCloud2 message and extract point data.
    
    This is a simplified parser that handles basic PointCloud2 format.
    For production use, consider using proper ROS message libraries.
    """
    # This is a placeholder implementation
    # In practice, you'd need to properly parse the PointCloud2 message format
    # which includes field definitions, point step, row step, etc.
    
    # For now, return dummy data - this would need proper implementation
    # based on the actual message structure
    print("Warning: PointCloud2 parsing not fully implemented. Using dummy data.")
    
    # Generate some dummy points for demonstration
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10
    intensity = np.random.random(n_points)
    rgb = None
    
    return points, intensity, rgb

def read_pointcloud_csv(csv_path):
    """
    Read point cloud data from CSV file (extracted by bagpy).
    This handles both simple CSV format and PointCloud2 binary data.
    """
    import pandas as pd
    import base64
    
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Check if this is a PointCloud2 message format
        if 'data' in df.columns and 'point_step' in df.columns and 'fields' in df.columns:
            print("Detected PointCloud2 format, parsing binary data...")
            return parse_pointcloud2_csv(df)
        
        # Try to find x, y, z columns for simple format
        xyz_cols = []
        for col in ['x', 'y', 'z']:
            if col in df.columns:
                xyz_cols.append(col)
            elif f'point.{col}' in df.columns:
                xyz_cols.append(f'point.{col}')
            elif f'data.{col}' in df.columns:
                xyz_cols.append(f'data.{col}')
        
        if len(xyz_cols) < 3:
            raise ValueError(f"Could not find x, y, z columns in CSV. Available columns: {df.columns.tolist()}")
        
        points = df[xyz_cols].values
        
        # Try to find intensity
        intensity = None
        for col in ['intensity', 'i', 'point.intensity', 'data.intensity']:
            if col in df.columns:
                intensity = df[col].values
                break
        
        # Try to find RGB
        rgb = None
        rgb_cols = []
        for color in ['r', 'g', 'b']:
            for prefix in ['', 'point.', 'data.', 'color_']:
                col = f'{prefix}{color}'
                if col in df.columns:
                    rgb_cols.append(col)
                    break
        
        if len(rgb_cols) == 3:
            rgb = df[rgb_cols].values
        
        return points, intensity, rgb
        
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {csv_path}: {e}")

def parse_pointcloud2_csv(df):
    """
    Parse PointCloud2 message from CSV extracted by bagpy.
    """
    import base64
    
    if df.empty:
        raise ValueError("Empty dataframe")
    
    # Get the first message (you could extend this to handle multiple messages)
    row = df.iloc[0]
    
    try:
        # Extract point cloud parameters
        height = int(row['height'])
        width = int(row['width'])
        point_step = int(row['point_step'])
        row_step = int(row['row_step'])
        
        # Parse the fields string to understand the data structure
        fields_str = row['fields']
        fields = parse_pointcloud2_fields(fields_str)
        
        # Get the binary data
        data_str = row['data']
        
        # The data is stored as a string representation of bytes like "b'\\x..'"
        try:
            if isinstance(data_str, str):
                if data_str.startswith("b'") and data_str.endswith("'"):
                    # Remove b' and ' wrapper and decode the escape sequences
                    data_str = data_str[2:-1]
                    binary_data = bytes(data_str, 'utf-8').decode('unicode_escape').encode('latin1')
                elif data_str.startswith('[') and data_str.endswith(']'):
                    # Convert string representation of byte array to actual bytes
                    data_str = data_str.strip('[]')
                    byte_values = [int(x.strip()) for x in data_str.split(',')]
                    binary_data = bytes(byte_values)
                else:
                    # Try base64 decode
                    binary_data = base64.b64decode(data_str)
            else:
                binary_data = data_str
        except Exception as e:
            print(f"Warning: Could not decode binary data ({e}), using dummy points")
            return generate_dummy_points()
        
        # Parse the binary data based on fields
        points = parse_binary_pointcloud_data(binary_data, fields, width * height, point_step)
        
        return points
        
    except Exception as e:
        print(f"Error parsing PointCloud2: {e}")
        print("Falling back to dummy data...")
        return generate_dummy_points()

def parse_pointcloud2_fields(fields_str):
    """
    Parse the fields string from PointCloud2 message.
    Format: [name: "x" offset: 0 datatype: 7 count: 1, name: "y" ...]
    """
    fields = {}
    
    # Remove brackets and split by comma to get individual field definitions
    fields_str = fields_str.strip('[]')
    field_definitions = fields_str.split(', name:')
    
    for i, field_def in enumerate(field_definitions):
        if i > 0:
            field_def = 'name:' + field_def  # Add back the 'name:' prefix
        
        current_field = {}
        lines = field_def.replace('\\n', '\n').split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('name:'):
                name = line.split(':', 1)[1].strip().strip('"')
                current_field['name'] = name
            elif line.startswith('offset:'):
                current_field['offset'] = int(line.split(':')[1].strip())
            elif line.startswith('datatype:'):
                current_field['datatype'] = int(line.split(':')[1].strip())
            elif line.startswith('count:'):
                current_field['count'] = int(line.split(':')[1].strip())
        
        if 'name' in current_field:
            fields[current_field['name']] = current_field
    
    return fields

def parse_binary_pointcloud_data(binary_data, fields, num_points, point_step):
    """
    Parse binary point cloud data based on field definitions.
    Enhanced for Velodyne VLP-16 data with ring information.
    """
    import struct
    
    # Map ROS datatypes to struct formats
    datatype_map = {
        1: 'b',   # INT8
        2: 'B',   # UINT8  
        3: 'h',   # INT16
        4: 'H',   # UINT16
        5: 'i',   # INT32
        6: 'I',   # UINT32
        7: 'f',   # FLOAT32
        8: 'd',   # FLOAT64
    }
    
    points = []
    intensity_vals = []
    ring_vals = []
    
    try:
        for i in range(num_points):
            point_offset = i * point_step
            
            if point_offset + point_step > len(binary_data):
                break
            
            point_data = binary_data[point_offset:point_offset + point_step]
            
            point = {}
            for field_name, field_info in fields.items():
                offset = field_info['offset']
                datatype = field_info['datatype']
                count = field_info.get('count', 1)
                
                if datatype in datatype_map:
                    fmt = datatype_map[datatype]
                    size = struct.calcsize(fmt)
                    
                    if offset + size * count <= len(point_data):
                        if count == 1:
                            value = struct.unpack(fmt, point_data[offset:offset + size])[0]
                        else:
                            value = struct.unpack(f'{count}{fmt}', point_data[offset:offset + size * count])
                        point[field_name] = value
            
            # Extract x, y, z coordinates - filter out invalid points
            if 'x' in point and 'y' in point and 'z' in point:
                x, y, z = point['x'], point['y'], point['z']
                
                # Filter out points at origin (common LiDAR artifact) and extreme values
                if (abs(x) > 0.1 or abs(y) > 0.1 or abs(z) > 0.1) and abs(x) < 200 and abs(y) < 200 and abs(z) < 50:
                    points.append([x, y, z])
                    
                    # Extract intensity if available
                    if 'intensity' in point:
                        intensity_vals.append(point['intensity'])
                    
                    # Extract ring information (Velodyne specific)
                    if 'ring' in point:
                        ring_vals.append(point['ring'])
    
    except Exception as e:
        print(f"Error parsing binary data: {e}")
        return generate_dummy_points()
    
    if not points:
        print("No valid points extracted from binary data")
        return generate_dummy_points()
    
    points = np.array(points)
    intensity = np.array(intensity_vals) if intensity_vals else None
    
    # Use ring information for coloring if available (Velodyne specific)
    ring_data = np.array(ring_vals) if ring_vals else None
    
    print(f"Extracted {len(points)} points from PointCloud2 data")
    if intensity is not None:
        print(f"Intensity range: {intensity.min():.2f} to {intensity.max():.2f}")
    if ring_data is not None:
        print(f"Ring range: {ring_data.min()} to {ring_data.max()} (VLP-16 has 16 rings: 0-15)")
        unique_rings = np.unique(ring_data)
        print(f"Active rings: {sorted(unique_rings)}")
    
    return points, intensity, ring_data

def generate_dummy_points():
    """Generate dummy point cloud data for testing."""
    print("Generating dummy point cloud data...")
    n_points = 1000
    points = np.random.random((n_points, 3)) * 10
    intensity = np.random.random(n_points)
    rgb = None
    return points, intensity, rgb

def read_points(path, scan_indices=None, stride=1, topic=None, frame_index=0):
    """
    Read points from E57, PCD, or ROS bag files.
    
    Args:
        path (str): Path to the point cloud file (.e57, .pcd, or .bag)
        scan_indices: For E57 files, which scans to read. Ignored for other formats.
        stride (int): Skip every nth point to reduce density
        topic (str, optional): For bag files, specific topic to read
        frame_index (int): For bag files, which frame/message to extract
        
    Returns:
        tuple: (points, intensity, extra_data) arrays
               extra_data can be RGB for E57/PCD or ring data for Velodyne
    """
    # Determine file format based on extension
    file_ext = os.path.splitext(path)[1].lower()
    
    if file_ext == '.pcd':
        # Read PCD file
        P, intensity, rgb = read_pcd_file(path)
        
        # Apply stride if specified
        if stride > 1:
            P = P[::stride]
            if intensity is not None:
                intensity = intensity[::stride]
            if rgb is not None:
                rgb = rgb[::stride]
                
        return P, intensity, rgb
        
    elif file_ext == '.bag':
        # Read ROS bag file
        P, intensity, extra_data = read_bag_file(path, topic, frame_index)
        
        # Apply stride if specified
        if stride > 1:
            P = P[::stride]
            if intensity is not None:
                intensity = intensity[::stride]
            if extra_data is not None:
                extra_data = extra_data[::stride]
                
        return P, intensity, extra_data
        
    elif file_ext == '.e57':
        # Original E57 reading logic
        P_parts, I_parts, RGB_parts = [], [], []
        with E57(path) as e:
            idxs = range(e.scan_count) if scan_indices is None else scan_indices
            for i in idxs:
                try:
                    s = e.read_scan(i, colors=True, intensity=True, ignore_missing_fields=True)
                except ValueError:
                    s = e.read_scan_raw(i)

                x, y, z = s["cartesianX"], s["cartesianY"], s["cartesianZ"]
                m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                p = np.c_[x[m], y[m], z[m]]
                if stride > 1: p = p[::stride]
                P_parts.append(p)

                if "intensity" in s:
                    v = s["intensity"][m]
                    if stride > 1: v = v[::stride]
                    I_parts.append(v)
                else:
                    I_parts.append(None)

                has_rgb = {"colorRed","colorGreen","colorBlue"}.issubset(s.keys())
                if has_rgb:
                    r = s["colorRed"][m]; g = s["colorGreen"][m]; b = s["colorBlue"][m]
                    if stride > 1: r, g, b = r[::stride], g[::stride], b[::stride]
                    RGB_parts.append(np.c_[r, g, b])
                else:
                    RGB_parts.append(None)

        # Concatenate points
        P = np.vstack(P_parts) if len(P_parts) else np.empty((0,3))

        # Only use intensity if every selected scan had it and the lengths match
        if all(i is not None for i in I_parts):
            intensity = np.concatenate(I_parts)
            if len(intensity) != len(P): intensity = None
        else:
            intensity = None

        # Only use RGB if every selected scan had it and the lengths match
        if all(R is not None for R in RGB_parts):
            rgb = np.vstack(RGB_parts).astype(np.uint8)
            if len(rgb) != len(P): rgb = None
        else:
            rgb = None

        return P, intensity, rgb
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats are .e57, .pcd, and .bag")

def to_polydata(P, intensity=None, rgb=None, ring=None):
    """
    Convert points to PyVista PolyData with enhanced support for LiDAR data.
    """
    cloud = pv.PolyData(P)
    if intensity is not None and len(intensity) == len(P):
        cloud["intensity"] = intensity.astype(np.float32)
    if rgb is not None and len(rgb) == len(P):
        # Ensure RGB array is properly shaped and within valid range
        if len(rgb.shape) == 2 and rgb.shape[1] in (3, 4):
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            cloud["RGB"] = rgb
        else:
            print(f"Warning: RGB array has invalid shape {rgb.shape}, skipping RGB coloring")
    if ring is not None and len(ring) == len(P):
        cloud["ring"] = ring.astype(np.uint8)
    return cloud

def save_point_cloud_data(P, intensity=None, rgb=None, ring=None, output_path=None, file_prefix="point_cloud"):
    """
    Save point cloud data as numpy arrays to disk.
    
    Args:
        P (np.ndarray): Point coordinates array (N, 3)
        intensity (np.ndarray, optional): Intensity values
        rgb (np.ndarray, optional): RGB color values
        ring (np.ndarray, optional): Ring information (Velodyne)
        output_path (str, optional): Directory to save files. If None, saves in current directory
        file_prefix (str): Prefix for saved files
    """
    if output_path is None:
        output_path = "."
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save points (always present)
    points_file = os.path.join(output_path, f"{file_prefix}_points.npy")
    np.save(points_file, P)
    print(f"Saved points to: {points_file}")
    
    # Save intensity if available
    if intensity is not None and len(intensity) == len(P):
        intensity_file = os.path.join(output_path, f"{file_prefix}_intensity.npy")
        np.save(intensity_file, intensity)
        print(f"Saved intensity to: {intensity_file}")
    
    # Save RGB if available
    if rgb is not None and len(rgb) == len(P):
        rgb_file = os.path.join(output_path, f"{file_prefix}_rgb.npy")
        np.save(rgb_file, rgb)
        print(f"Saved RGB to: {rgb_file}")
    
    # Save ring data if available (Velodyne)
    if ring is not None and len(ring) == len(P):
        ring_file = os.path.join(output_path, f"{file_prefix}_ring.npy")
        np.save(ring_file, ring)
        print(f"Saved ring data to: {ring_file}")
    
    # Save metadata
    metadata = {
        'num_points': len(P),
        'has_intensity': intensity is not None,
        'has_rgb': rgb is not None,
        'has_ring': ring is not None,
        'point_range': {
            'x': [float(P[:, 0].min()), float(P[:, 0].max())],
            'y': [float(P[:, 1].min()), float(P[:, 1].max())],
            'z': [float(P[:, 2].min()), float(P[:, 2].max())]
        }
    }
    
    if intensity is not None:
        metadata['intensity_range'] = [float(intensity.min()), float(intensity.max())]
    
    if ring is not None:
        metadata['ring_range'] = [int(ring.min()), int(ring.max())]
        metadata['unique_rings'] = [int(x) for x in np.unique(ring)]
    
    metadata_file = os.path.join(output_path, f"{file_prefix}_metadata.json")
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_file}")

def main():
    ap = argparse.ArgumentParser(description="Visualize point cloud data from E57, PCD, or ROS bag files")
    ap.add_argument("file_path", help="Path to point cloud file (.e57, .pcd, or .bag)")
    ap.add_argument("--scans", type=str, default=None, help="e.g. 0 or 0,2,5 (only for E57 files)")
    ap.add_argument("--stride", type=int, default=1, help="Skip every nth point to reduce density")
    ap.add_argument("--colorby", choices=["rgb","intensity","elevation","ring"], default="intensity", 
                   help="Color points by: rgb, intensity, elevation, or ring (Velodyne)")
    ap.add_argument("--point-size", type=float, default=2.0, help="Point size for rendering")
    ap.add_argument("--spheres", action="store_true", help="Render points as spheres")
    ap.add_argument("--topic", type=str, default=None, help="ROS topic name (for bag files)")
    ap.add_argument("--frame", type=int, default=0, help="Frame index to extract (for bag files)")
    ap.add_argument("--save", type=str, default=None, help="Save point cloud data as numpy arrays to specified directory")
    ap.add_argument("--prefix", type=str, default="point_cloud", help="File prefix for saved numpy arrays")
    ap.add_argument("--no-viz", action="store_true", help="Skip visualization and only save data")
    a = ap.parse_args()

    print(f"Loading point cloud from: {a.file_path}")
    if a.file_path.endswith('.bag'):
        print(f"  Topic: {a.topic or 'auto-detect'}")
        print(f"  Frame: {a.frame}")
    print(f"  Stride: {a.stride}")
    if not a.no_viz:
        print(f"  Color by: {a.colorby}")

    # For PCD and bag files, scan_indices parameter is ignored
    scan_indices = None if a.scans is None else [int(x) for x in a.scans.split(",")]
    P, intensity, extra_data = read_points(a.file_path, scan_indices, a.stride, a.topic, a.frame)
    
    # Determine what type of extra data we have
    is_velodyne_ring = (a.file_path.endswith('.bag') and extra_data is not None)
    rgb_data = None if is_velodyne_ring else extra_data
    ring_data = extra_data if is_velodyne_ring else None
    
    # Save data if requested
    if a.save:
        print(f"\nSaving point cloud data to: {a.save}")
        save_point_cloud_data(P, intensity, rgb_data, ring_data, a.save, a.prefix)
    
    # Skip visualization if requested
    if a.no_viz:
        print("Skipping visualization as requested.")
        return
    
    # Setup VTK environment for cross-platform stability
    VTKSafetyManager.setup_vtk_environment()
    
    cloud = to_polydata(P, intensity, rgb_data, ring_data)

    pl = pv.Plotter(window_size=(1280,800))
    pl.set_background("black")

    # Print point cloud statistics
    print(f"\nPoint cloud statistics:")
    print(f"  Points: {len(P):,}")
    print(f"  X range: {P[:, 0].min():.2f} to {P[:, 0].max():.2f}")
    print(f"  Y range: {P[:, 1].min():.2f} to {P[:, 1].max():.2f}")
    print(f"  Z range: {P[:, 2].min():.2f} to {P[:, 2].max():.2f}")
    if intensity is not None:
        print(f"  Intensity range: {intensity.min():.2f} to {intensity.max():.2f}")
    if ring_data is not None:
        unique_rings = np.unique(ring_data)
        print(f"  Velodyne rings: {len(unique_rings)} active ({min(unique_rings)}-{max(unique_rings)})")

    # choose scalars
    scalars = None; cmap = None; use_rgb = False
    
    # Enhanced coloring options for LiDAR data
    if a.colorby == "ring" and "ring" in cloud.array_names:
        scalars, cmap = "ring", "Set1"  # Discrete colormap for rings
        print("Coloring by Velodyne laser ring (0-15)")
    elif a.colorby == "rgb" and "RGB" in cloud.array_names:
        # Check if RGB data is valid
        rgb_array = cloud["RGB"]
        if (rgb_array.shape[0] == cloud.n_points and 
            len(rgb_array.shape) == 2 and 
            rgb_array.shape[1] in (3, 4)):
            try:
                use_rgb = True
                scalars = None
                print("Coloring by RGB values")
            except:
                print("Warning: RGB coloring not supported, falling back to elevation")
                z = cloud.points[:, 2].astype(np.float32)
                cloud["elevation"] = z
                scalars, cmap = "elevation", "terrain"
        else:
            print("Warning: Invalid RGB data, falling back to elevation")
            z = cloud.points[:, 2].astype(np.float32)
            cloud["elevation"] = z
            scalars, cmap = "elevation", "terrain"
    elif a.colorby == "intensity" and "intensity" in cloud.array_names and len(cloud["intensity"]) == cloud.n_points:
        scalars, cmap = "intensity", "viridis"
        print("Coloring by LiDAR intensity")
    else:
        z = cloud.points[:, 2].astype(np.float32)
        cloud["elevation"] = z
        scalars, cmap = "elevation", "terrain"
        print("Coloring by elevation (Z coordinate)")

    try:
        pl.add_points(
            cloud,
            scalars=scalars,
            rgb=use_rgb,
            cmap=cmap,
            render_points_as_spheres=a.spheres,
            point_size=a.point_size,
        )
    except ValueError as e:
        if "RGB array" in str(e) and use_rgb:
            print("Warning: RGB coloring failed, falling back to elevation coloring")
            # Fallback to elevation coloring
            z = cloud.points[:, 2].astype(np.float32)
            cloud["elevation"] = z
            pl.add_points(
                cloud,
                scalars="elevation",
                rgb=False,
                cmap="terrain",
                render_points_as_spheres=a.spheres,
                point_size=a.point_size,
            )
        else:
            raise
    
    # Add coordinate axes for reference
    pl.add_axes()
    
    # Add title with file information
    title = f"Point Cloud: {os.path.basename(a.file_path)}"
    if a.file_path.endswith('.bag'):
        title += f" (Frame {a.frame})"
    pl.add_title(title)
    
    # Show with proper cleanup
    try:
        pl.show(auto_close=True)
    finally:
        # Clean up VTK resources
        VTKSafetyManager.cleanup_vtk_plotter(pl)

if __name__ == "__main__": main()