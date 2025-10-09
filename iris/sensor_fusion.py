#!/usr/bin/env python3
"""
Sensor Fusion System for RCooper Dataset
Combines YOLO car detection in images with LIDAR point cloud data.

This system:
1. Loads synchronized camera-LIDAR data from correspondence file
2. Uses YOLO to detect cars in camera images
3. Projects LIDAR points to camera coordinates
4. Finds LIDAR points corresponding to detected car bounding boxes

Author: Claude
Date: September 2025
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import argparse

# YOLO and point cloud processing
from ultralytics import YOLO
import open3d as o3d


class CameraLidarFusion:
    """Sensor fusion system for camera and LIDAR data."""

    def __init__(self, correspondence_file: str, base_data_path: str, calib_path: str = None):
        """
        Initialize the sensor fusion system.

        Args:
            correspondence_file: Path to camera-LIDAR correspondence JSON file
            base_data_path: Base path for the data directory
            calib_path: Path to calibration directory (defaults to base_data_path/calib)
        """
        self.base_data_path = Path(base_data_path)
        self.calib_path = Path(calib_path) if calib_path else self.base_data_path / "calib"
        self.correspondence_data = self._load_correspondence_file(correspondence_file)

        # Initialize YOLO model for car detection
        print("Loading YOLO model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Use nano version for speed

    def _load_correspondence_file(self, file_path: str) -> dict:
        """Load the camera-LIDAR correspondence file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _load_device_calibration(self, device_id: str) -> dict:
        """
        Load calibration parameters for a specific device from calibration files.

        Args:
            device_id: Device ID (e.g., '105') to load calibration for

        Returns:
            Dictionary with calibration parameters for both cameras
        """
        calib_file = self.calib_path / "lidar2cam" / f"{device_id}.json"

        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")

        with open(calib_file, 'r') as f:
            calib_data = json.load(f)

        calibration = {}

        # Process each camera
        for cam_key in ['cam_0', 'cam_1']:
            if cam_key not in calib_data:
                continue

            cam_data = calib_data[cam_key]

            # Extract intrinsic matrix
            intrinsic_matrix = np.array(cam_data['intrinsic'])

            # Extract extrinsic matrix (4x4 transformation matrix)
            extrinsic_matrix = np.array(cam_data['extrinsic'])

            # Extract rotation and translation from extrinsic matrix
            rotation_matrix = extrinsic_matrix[:3, :3]
            translation_vector = extrinsic_matrix[:3, 3]

            # Store calibration data
            cam_name = cam_key.replace('_', '')  # 'cam_0' -> 'cam0'
            calibration[cam_name] = {
                'camera_matrix': intrinsic_matrix,
                'distortion': np.zeros(5),  # No distortion coefficients in this format
                'image_size': [1920, 1200]  # Standard image size for RCooper dataset
            }

            calibration[f'lidar_to_{cam_name}'] = {
                'rotation': rotation_matrix,
                'translation': translation_vector
            }

        return calibration

    def get_frame_data(self, frame_id: int) -> Optional[dict]:
        """Get data for a specific frame ID."""
        for frame in self.correspondence_data['correspondence_frames']:
            if frame['frame_id'] == frame_id:
                return frame
        return None

    def load_lidar_points(self, pcd_file_path: str) -> np.ndarray:
        """Load LIDAR point cloud from PCD file."""
        full_path = self.base_data_path / pcd_file_path.lstrip('./')

        # Try to load with open3d
        try:
            pcd = o3d.io.read_point_cloud(str(full_path))
            points = np.asarray(pcd.points)
            print(f"Loaded {points.shape[0]} LIDAR points from {full_path}")
            return points
        except Exception as e:
            print(f"Error loading point cloud with open3d: {e}")
            # Fallback to manual PCD parsing if needed
            return self._parse_pcd_file(full_path)

    def _parse_pcd_file(self, file_path: Path) -> np.ndarray:
        """Manually parse ASCII PCD file as fallback."""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find data start
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('DATA'):
                data_start = i + 1
                break

        # Parse point data
        points = []
        for line in lines[data_start:]:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])

        return np.array(points)

    def project_lidar_to_camera(self, lidar_points: np.ndarray, camera: str, calibration: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project LIDAR points to camera image coordinates.

        Args:
            lidar_points: Nx3 array of LIDAR points in LIDAR coordinate system
            camera: 'cam0' or 'cam1'
            calibration: Calibration parameters dictionary

        Returns:
            image_points: Nx2 array of projected 2D points
            depths: N array of depth values (z coordinates in camera frame)
        """
        # Get calibration parameters
        camera_matrix = calibration[camera]['camera_matrix']
        distortion = calibration[camera]['distortion']
        extrinsics = calibration[f'lidar_to_{camera}']

        # Transform LIDAR points to camera coordinate system
        R = extrinsics['rotation']
        t = extrinsics['translation']

        # Apply transformation: P_cam = R * P_lidar + t
        lidar_points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
        transformation_matrix = np.hstack([R, t.reshape(-1, 1)])
        camera_points_homo = lidar_points_homo @ transformation_matrix.T
        camera_points = camera_points_homo[:, :3]

        # Filter points behind the camera (negative Z)
        valid_mask = camera_points[:, 2] > 0
        camera_points = camera_points[valid_mask]

        if camera_points.shape[0] == 0:
            return np.array([]), np.array([]), valid_mask

        # Project to image plane
        image_points, _ = cv2.projectPoints(
            camera_points,
            np.zeros(3), np.zeros(3),  # No additional rotation/translation
            camera_matrix,
            distortion
        )

        image_points = image_points.reshape(-1, 2).astype(np.float32)
        depths = camera_points[:, 2]

        return image_points, depths, valid_mask

    def detect_cars_yolo(self, image: np.ndarray) -> List[Dict]:
        """
        Detect cars in image using YOLO.

        Args:
            image: Input image

        Returns:
            List of detection dictionaries with bbox, confidence, class
        """
        results = self.yolo_model(image)

        car_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # YOLO class 2 is 'car'
                    if int(box.cls) == 2:
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        confidence = float(box.conf[0])

                        car_detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class': 'car'
                        })

        return car_detections

    def find_lidar_points_in_bbox(self, image_points: np.ndarray, depths: np.ndarray,
                                  bbox: np.ndarray, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find LIDAR points that fall within a bounding box.

        Args:
            image_points: Nx2 array of projected points
            depths: N array of corresponding depths
            bbox: [x1, y1, x2, y2] bounding box
            image_size: (width, height) of image

        Returns:
            points_in_bbox: Indices of points within bbox
            bbox_depths: Corresponding depth values
        """
        # Filter points within image bounds
        valid_x = (image_points[:, 0] >= 0) & (image_points[:, 0] < image_size[0])
        valid_y = (image_points[:, 1] >= 0) & (image_points[:, 1] < image_size[1])
        valid_image = valid_x & valid_y

        # Filter points within bounding box
        x1, y1, x2, y2 = bbox
        in_bbox_x = (image_points[:, 0] >= x1) & (image_points[:, 0] <= x2)
        in_bbox_y = (image_points[:, 1] >= y1) & (image_points[:, 1] <= y2)
        in_bbox = in_bbox_x & in_bbox_y & valid_image

        return np.where(in_bbox)[0], depths[in_bbox]

    def visualize_results(self, image: np.ndarray, car_detections: List[Dict],
                         projected_points: np.ndarray, lidar_points_indices: List[np.ndarray],
                         output_path: str = None) -> np.ndarray:
        """
        Visualize detection results on image.

        Args:
            image: Original image
            car_detections: List of car detection results
            projected_points: All projected LIDAR points
            lidar_points_indices: Indices of LIDAR points for each detection
            output_path: Optional path to save visualization

        Returns:
            Annotated image
        """
        vis_image = image.copy()

        # Draw all projected LIDAR points in light blue
        if len(projected_points) > 0:
            for point in projected_points:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                    cv2.circle(vis_image, (x, y), 2, (255, 255, 0), -1)

        # Draw car detections and associated LIDAR points
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        for i, (detection, point_indices) in enumerate(zip(car_detections, lidar_points_indices)):
            color = colors[i % len(colors)]
            bbox = detection['bbox']
            confidence = detection['confidence']

            # Draw bounding box
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"Car {confidence:.2f} ({len(point_indices)} pts)"
            cv2.putText(vis_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Highlight LIDAR points in this detection
            for idx in point_indices:
                if idx < len(projected_points):
                    point = projected_points[idx]
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < vis_image.shape[1] and 0 <= y < vis_image.shape[0]:
                        cv2.circle(vis_image, (x, y), 3, color, -1)

        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"Visualization saved to {output_path}")

        return vis_image

    def process_frame(self, frame_id: int, camera: str = 'cam0', visualize: bool = True) -> Dict:
        """
        Process a specific frame for sensor fusion.

        Args:
            frame_id: Frame ID to process
            camera: Camera to use ('cam0' or 'cam1')
            visualize: Whether to create visualization

        Returns:
            Dictionary with fusion results
        """
        print(f"Processing frame {frame_id} with {camera}")

        # Get frame data
        frame_data = self.get_frame_data(frame_id)
        if not frame_data:
            raise ValueError(f"Frame {frame_id} not found in correspondence data")

        # Load calibration for this specific device
        device_id = frame_data['device_id']
        print(f"Loading calibration for device {device_id}")
        calibration = self._load_device_calibration(device_id)

        # Load image
        image_path = self.base_data_path / frame_data[camera]['file_path'].lstrip('./')
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))
        print(f"Loaded image: {image.shape}")

        # Load LIDAR points
        lidar_points = self.load_lidar_points(frame_data['lidar']['file_path'])

        # Detect cars using YOLO
        print("Detecting cars with YOLO...")
        car_detections = self.detect_cars_yolo(image)
        print(f"Found {len(car_detections)} car detections")

        # Project LIDAR points to camera
        print("Projecting LIDAR points to camera...")
        projected_points, depths, valid_mask = self.project_lidar_to_camera(lidar_points, camera, calibration)
        print(f"Projected {len(projected_points)} LIDAR points to image plane")

        # Find LIDAR points for each car detection
        fusion_results = []
        lidar_points_per_detection = []

        image_size = tuple(calibration[camera]['image_size'])

        for i, detection in enumerate(car_detections):
            bbox = detection['bbox']
            point_indices, _ = self.find_lidar_points_in_bbox(
                projected_points, depths, bbox, image_size
            )

            # Get original 3D LIDAR points for this detection
            original_indices = np.where(valid_mask)[0][point_indices]
            detection_lidar_points = lidar_points[original_indices]

            result = {
                'detection_id': i,
                'bbox': bbox,
                'confidence': detection['confidence'],
                'num_lidar_points': len(point_indices),
                'lidar_points_3d': detection_lidar_points,
                'depth_range': (depths[point_indices].min(), depths[point_indices].max()) if len(depths[point_indices]) > 0 else (0, 0),
                'projected_points': projected_points[point_indices]
            }

            fusion_results.append(result)
            lidar_points_per_detection.append(point_indices)

            print(f"Detection {i}: {len(point_indices)} LIDAR points, "
                  f"depth range: {result['depth_range'][0]:.2f}-{result['depth_range'][1]:.2f}m")

        # Create visualization
        if visualize:
            # Create sequence-based output directory with camera grouping
            sequence_id = frame_data['sequence_id']
            output_dir = Path("sensor_fusion_output") / f"{device_id}_{sequence_id}" / camera
            output_dir.mkdir(parents=True, exist_ok=True)

            # Include timestamp in filename for proper ordering
            timestamp = frame_data[camera]['timestamp']
            output_path = output_dir / f"frame_{frame_id:03d}_{timestamp:.6f}.jpg"
            _ = self.visualize_results(
                image, car_detections, projected_points,
                lidar_points_per_detection, str(output_path)
            )

        return {
            'frame_id': frame_id,
            'camera': camera,
            'image_shape': image.shape,
            'total_lidar_points': len(lidar_points),
            'projected_points': len(projected_points),
            'car_detections': len(car_detections),
            'fusion_results': fusion_results,
            'timestamp': frame_data[camera]['timestamp']
        }

    def process_frame_data(self, frame_data: Dict, camera: str = 'cam0', visualize: bool = True) -> Dict:
        """
        Process sensor fusion using specific frame data.

        Args:
            frame_data: Frame data dictionary from correspondence file
            camera: Camera to use ('cam0' or 'cam1')
            visualize: Whether to create visualization

        Returns:
            Dictionary with fusion results
        """
        frame_id = frame_data['frame_id']
        print(f"Processing frame {frame_id} with {camera}")

        # Load calibration for this specific device
        device_id = frame_data['device_id']
        print(f"Loading calibration for device {device_id}")
        calibration = self._load_device_calibration(device_id)

        # Load image
        image_path = self.base_data_path / frame_data[camera]['file_path'].lstrip('./')
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path))
        print(f"Loaded image: {image.shape}")

        # Load LIDAR points
        lidar_points = self.load_lidar_points(frame_data['lidar']['file_path'])

        # Detect cars using YOLO
        print("Detecting cars with YOLO...")
        car_detections = self.detect_cars_yolo(image)
        print(f"Found {len(car_detections)} car detections")

        # Project LIDAR points to camera
        print("Projecting LIDAR points to camera...")
        projected_points, depths, valid_mask = self.project_lidar_to_camera(lidar_points, camera, calibration)
        print(f"Projected {len(projected_points)} LIDAR points to image plane")

        # Find LIDAR points for each car detection
        fusion_results = []
        lidar_points_per_detection = []

        image_size = tuple(calibration[camera]['image_size'])

        for i, detection in enumerate(car_detections):
            bbox = detection['bbox']
            point_indices, _ = self.find_lidar_points_in_bbox(
                projected_points, depths, bbox, image_size
            )

            # Get original 3D LIDAR points for this detection
            original_indices = np.where(valid_mask)[0][point_indices]
            detection_lidar_points = lidar_points[original_indices]

            result = {
                'detection_id': i,
                'bbox': bbox,
                'confidence': detection['confidence'],
                'num_lidar_points': len(point_indices),
                'lidar_points_3d': detection_lidar_points,
                'depth_range': (depths[point_indices].min(), depths[point_indices].max()) if len(depths[point_indices]) > 0 else (0, 0),
                'projected_points': projected_points[point_indices]
            }

            fusion_results.append(result)
            lidar_points_per_detection.append(point_indices)

            print(f"Detection {i}: {len(point_indices)} LIDAR points, "
                  f"depth range: {result['depth_range'][0]:.2f}-{result['depth_range'][1]:.2f}m")

        # Create visualization
        if visualize:
            # Create sequence-based output directory with camera grouping
            sequence_id = frame_data['sequence_id']
            output_dir = Path("sensor_fusion_output") / f"{device_id}_{sequence_id}" / camera
            output_dir.mkdir(parents=True, exist_ok=True)

            # Include timestamp in filename for proper ordering
            timestamp = frame_data[camera]['timestamp']
            output_path = output_dir / f"frame_{frame_id:03d}_{timestamp:.6f}.jpg"
            _ = self.visualize_results(
                image, car_detections, projected_points,
                lidar_points_per_detection, str(output_path)
            )

        return {
            'frame_id': frame_id,
            'camera': camera,
            'image_shape': image.shape,
            'total_lidar_points': len(lidar_points),
            'projected_points': len(projected_points),
            'car_detections': len(car_detections),
            'fusion_results': fusion_results,
            'timestamp': frame_data[camera]['timestamp']
        }


def main():
    """Main function to run sensor fusion on specific frame."""
    parser = argparse.ArgumentParser(description='Camera-LIDAR Sensor Fusion for RCooper Dataset')
    parser.add_argument('--frame_id', type=int, default=2,
                       help='Frame ID to process (default: 2)')
    parser.add_argument('--camera', type=str, default='cam0', choices=['cam0', 'cam1'],
                       help='Camera to use (default: cam0)')
    parser.add_argument('--correspondence_file', type=str,
                       default='data/RCooper/corespondence_camera_lidar.json',
                       help='Path to correspondence file')
    parser.add_argument('--data_path', type=str, default='data/RCooper',
                       help='Base data directory path')
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable visualization output')

    args = parser.parse_args()

    try:
        # Initialize fusion system
        print("Initializing sensor fusion system...")
        fusion_system = CameraLidarFusion(
            correspondence_file=args.correspondence_file,
            base_data_path=args.data_path
        )

        # Process the specified frame
        results = fusion_system.process_frame(
            frame_id=args.frame_id,
            camera=args.camera,
            visualize=not args.no_viz
        )

        # Print summary
        print("\n=== SENSOR FUSION RESULTS ===")
        print(f"Frame ID: {results['frame_id']}")
        print(f"Camera: {results['camera']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Image shape: {results['image_shape']}")
        print(f"Total LIDAR points: {results['total_lidar_points']}")
        print(f"Projected LIDAR points: {results['projected_points']}")
        print(f"Car detections: {results['car_detections']}")
        print()

        for i, result in enumerate(results['fusion_results']):
            print(f"Car Detection {i}:")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Bounding box: {result['bbox']}")
            print(f"  LIDAR points: {result['num_lidar_points']}")
            print(f"  Depth range: {result['depth_range'][0]:.2f}m - {result['depth_range'][1]:.2f}m")
            print()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())