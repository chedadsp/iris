#!/usr/bin/env python3
"""
Sensor Fusion WITHOUT YOLO Detection

This script performs camera-LIDAR sensor fusion without running YOLO detection.
Instead, it uses one of these methods:
1. Pre-computed bounding boxes from previous YOLO runs
2. Manual bounding box specification
3. Full scene projection (no detection filtering)

Perfect for when you want to:
- Skip expensive YOLO inference
- Use existing detection results
- Process full LIDAR scenes
- Debug calibration without detection

Usage:
    # Use existing detections from tracking data
    python sensor_fusion_no_yolo.py --sequence seq-6 --device 105 --mode tracking

    # Manual bounding boxes
    python sensor_fusion_no_yolo.py --sequence seq-6 --device 105 --mode manual --bbox 100,100,500,400

    # Full scene (no filtering)
    python sensor_fusion_no_yolo.py --sequence seq-6 --device 105 --mode full
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import open3d as o3d


class SensorFusionNoYOLO:
    """Sensor fusion without YOLO detection."""

    def __init__(self, base_data_path: str, correspondence_file: str = None):
        """
        Initialize sensor fusion system without YOLO.

        Args:
            base_data_path: Path to data directory
            correspondence_file: Optional path to correspondence JSON
        """
        self.base_data_path = Path(base_data_path)
        self.calib_path = self.base_data_path / "calib"

        if correspondence_file:
            with open(correspondence_file, 'r') as f:
                self.correspondence_data = json.load(f)
        else:
            self.correspondence_data = None

        print("✅ Sensor fusion initialized (NO YOLO)")

    def load_calibration(self, device_id: str) -> dict:
        """Load calibration for device."""
        calib_file = self.calib_path / "lidar2cam" / f"{device_id}.json"

        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")

        with open(calib_file, 'r') as f:
            calib_data = json.load(f)

        calibration = {}

        for cam_key in ['cam_0', 'cam_1']:
            if cam_key not in calib_data:
                continue

            cam_data = calib_data[cam_key]
            intrinsic = np.array(cam_data['intrinsic'])
            extrinsic = np.array(cam_data['extrinsic'])

            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]

            cam_name = cam_key.replace('_', '')
            calibration[cam_name] = {
                'camera_matrix': intrinsic,
                'distortion': np.zeros(5),
                'image_size': [1920, 1200]
            }

            calibration[f'lidar_to_{cam_name}'] = {
                'rotation': rotation,
                'translation': translation
            }

        return calibration

    def load_lidar_points(self, pcd_file: Path) -> np.ndarray:
        """Load LIDAR points from PCD file."""
        try:
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            points = np.asarray(pcd.points)
            print(f"  Loaded {points.shape[0]} LIDAR points")
            return points
        except Exception as e:
            print(f"  Error loading point cloud: {e}")
            return np.array([])

    def project_lidar_to_camera(self, lidar_points: np.ndarray, camera: str,
                               calibration: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project LIDAR points to camera coordinates."""
        camera_matrix = calibration[camera]['camera_matrix']
        distortion = calibration[camera]['distortion']
        extrinsics = calibration[f'lidar_to_{camera}']

        # Transform to camera frame
        R = extrinsics['rotation']
        t = extrinsics['translation']

        lidar_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
        transform = np.hstack([R, t.reshape(-1, 1)])
        cam_points_homo = lidar_homo @ transform.T
        cam_points = cam_points_homo[:, :3]

        # Filter points behind camera
        valid_mask = cam_points[:, 2] > 0
        cam_points_valid = cam_points[valid_mask]

        if cam_points_valid.shape[0] == 0:
            return np.array([]), np.array([]), valid_mask

        # Project to image
        img_points, _ = cv2.projectPoints(
            cam_points_valid,
            np.zeros(3), np.zeros(3),
            camera_matrix,
            distortion
        )

        img_points = img_points.reshape(-1, 2).astype(np.float32)
        depths = cam_points_valid[:, 2]

        return img_points, depths, valid_mask

    def extract_bbox_points(self, lidar_points: np.ndarray, image_points: np.ndarray,
                           depths: np.ndarray, bbox: List[int],
                           image_size: Tuple[int, int] = (1920, 1200)) -> np.ndarray:
        """
        Extract LIDAR points within a bounding box.

        Args:
            lidar_points: Original LIDAR points (after valid_mask applied)
            image_points: Projected 2D points
            depths: Depth values
            bbox: [x1, y1, x2, y2] bounding box
            image_size: (width, height) of image

        Returns:
            LIDAR points within bounding box
        """
        x1, y1, x2, y2 = bbox

        # Find points within bbox and image bounds
        mask = (
            (image_points[:, 0] >= x1) & (image_points[:, 0] <= x2) &
            (image_points[:, 1] >= y1) & (image_points[:, 1] <= y2) &
            (image_points[:, 0] >= 0) & (image_points[:, 0] < image_size[0]) &
            (image_points[:, 1] >= 0) & (image_points[:, 1] < image_size[1])
        )

        return lidar_points[mask], image_points[mask], depths[mask]

    def process_with_tracking_data(self, sequence: str, device: str,
                                   output_dir: Path) -> Dict:
        """
        Process using existing tracking data (pre-computed bounding boxes).

        Args:
            sequence: Sequence ID (e.g., 'seq-6')
            device: Device ID (e.g., '105')
            output_dir: Output directory

        Returns:
            Processing results
        """
        print(f"\n📊 Processing {device}_{sequence} using TRACKING DATA")
        print("=" * 60)

        # Look for tracking data
        tracking_file = Path(f"sensor_fusion_output/enhanced_tracking/{device}_{sequence}/{device}_{sequence}_enhanced_tracking.json")

        if not tracking_file.exists():
            print(f"❌ Tracking data not found: {tracking_file}")
            return None

        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)

        print(f"✅ Loaded tracking data for {tracking_data.get('total_tracked_cars', 0)} cars")

        # Load calibration
        calibration = self.load_calibration(device)

        # This approach uses the summary data - would need frame-by-frame detections
        # for actual processing. For now, show what's available:

        results = {
            'sequence': f"{device}_{sequence}",
            'method': 'tracking_data',
            'tracking_summary': tracking_data.get('tracking_summary', {})
        }

        print("\n📋 Available tracking data:")
        for car_id, car_data in tracking_data.get('tracking_summary', {}).items():
            print(f"\n  Car {car_id}:")
            print(f"    Detections: {car_data['total_detections']}")
            print(f"    Points: {car_data['total_points']}")
            print(f"    Cameras: {car_data['cameras_used']}")
            print(f"    Centroid: {car_data['centroid_3d']}")

        return results

    def process_with_manual_bbox(self, sequence: str, device: str, camera: str,
                                 frame_id: int, bbox: List[int], output_dir: Path) -> Dict:
        """
        Process using manually specified bounding box.

        Args:
            sequence: Sequence ID
            device: Device ID
            camera: Camera ID ('cam0' or 'cam1')
            frame_id: Frame number to process
            bbox: [x1, y1, x2, y2] bounding box
            output_dir: Output directory

        Returns:
            Processing results
        """
        print(f"\n📊 Processing {device}_{sequence} frame {frame_id} with MANUAL BBOX")
        print(f"   Camera: {camera}")
        print(f"   BBox: {bbox}")
        print("=" * 60)

        # Load calibration
        calibration = self.load_calibration(device)

        # Find corresponding frame
        if not self.correspondence_data:
            print("❌ No correspondence data available")
            return None

        target_frame = None
        for frame in self.correspondence_data['correspondence_frames']:
            if (frame['device_id'] == device and
                frame['sequence_id'] == sequence and
                frame['frame_id'] == frame_id):
                target_frame = frame
                break

        if not target_frame:
            print(f"❌ Frame {frame_id} not found in correspondence data")
            return None

        # Load LIDAR points
        pcd_path = self.base_data_path / target_frame['lidar']['file_path'].lstrip('./')
        lidar_points = self.load_lidar_points(pcd_path)

        if lidar_points.shape[0] == 0:
            print("❌ No LIDAR points loaded")
            return None

        # Project to camera
        print(f"\n🔄 Projecting to {camera}...")
        img_points, depths, valid_mask = self.project_lidar_to_camera(
            lidar_points, camera, calibration
        )

        lidar_valid = lidar_points[valid_mask]

        print(f"  Projected {img_points.shape[0]} points")

        # Extract points in bbox
        bbox_points, bbox_img_points, bbox_depths = self.extract_bbox_points(
            lidar_valid, img_points, depths, bbox
        )

        print(f"  Extracted {bbox_points.shape[0]} points in bounding box")

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as PCD
        pcd_output = output_dir / f"frame_{frame_id:03d}_{camera}_bbox.pcd"
        if bbox_points.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(bbox_points)
            o3d.io.write_point_cloud(str(pcd_output), pcd)
            print(f"\n✅ Saved: {pcd_output}")

        # Load and visualize camera image
        cam_data = target_frame.get(camera, {})
        if cam_data and 'file_path' in cam_data:
            img_path = self.base_data_path / cam_data['file_path'].lstrip('./')
            if img_path.exists():
                img = cv2.imread(str(img_path))

                # Draw bbox
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw projected points in bbox
                for pt in bbox_img_points:
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)

                # Save visualization
                viz_output = output_dir / f"frame_{frame_id:03d}_{camera}_fusion.jpg"
                cv2.imwrite(str(viz_output), img)
                print(f"✅ Saved: {viz_output}")

        return {
            'sequence': f"{device}_{sequence}",
            'frame_id': frame_id,
            'camera': camera,
            'method': 'manual_bbox',
            'bbox': bbox,
            'total_points': bbox_points.shape[0],
            'output_pcd': str(pcd_output)
        }

    def process_full_scene(self, sequence: str, device: str, camera: str,
                          frame_id: int, output_dir: Path) -> Dict:
        """
        Process full scene without bounding box filtering.

        Args:
            sequence: Sequence ID
            device: Device ID
            camera: Camera ID
            frame_id: Frame number
            output_dir: Output directory

        Returns:
            Processing results
        """
        print(f"\n📊 Processing {device}_{sequence} frame {frame_id} FULL SCENE")
        print(f"   Camera: {camera}")
        print("=" * 60)

        # Load calibration
        calibration = self.load_calibration(device)

        # Find frame
        if not self.correspondence_data:
            print("❌ No correspondence data available")
            return None

        target_frame = None
        for frame in self.correspondence_data['correspondence_frames']:
            if (frame['device_id'] == device and
                frame['sequence_id'] == sequence and
                frame['frame_id'] == frame_id):
                target_frame = frame
                break

        if not target_frame:
            print(f"❌ Frame {frame_id} not found")
            return None

        # Load LIDAR
        pcd_path = self.base_data_path / target_frame['lidar']['file_path'].lstrip('./')
        lidar_points = self.load_lidar_points(pcd_path)

        if lidar_points.shape[0] == 0:
            return None

        # Project
        print(f"\n🔄 Projecting to {camera}...")
        img_points, depths, valid_mask = self.project_lidar_to_camera(
            lidar_points, camera, calibration
        )

        lidar_valid = lidar_points[valid_mask]

        print(f"  Projected {img_points.shape[0]} points visible in camera")

        # Filter to image bounds
        img_width, img_height = 1920, 1200
        in_bounds = (
            (img_points[:, 0] >= 0) & (img_points[:, 0] < img_width) &
            (img_points[:, 1] >= 0) & (img_points[:, 1] < img_height)
        )

        lidar_in_view = lidar_valid[in_bounds]
        img_points_in_view = img_points[in_bounds]
        depths_in_view = depths[in_bounds]

        print(f"  {lidar_in_view.shape[0]} points within image bounds")

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save PCD
        pcd_output = output_dir / f"frame_{frame_id:03d}_{camera}_full.pcd"
        if lidar_in_view.shape[0] > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_in_view)
            o3d.io.write_point_cloud(str(pcd_output), pcd)
            print(f"\n✅ Saved: {pcd_output}")

        # Visualize
        cam_data = target_frame.get(camera, {})
        if cam_data and 'file_path' in cam_data:
            img_path = self.base_data_path / cam_data['file_path'].lstrip('./')
            if img_path.exists():
                img = cv2.imread(str(img_path))

                # Draw all projected points colored by depth
                for pt, depth in zip(img_points_in_view, depths_in_view):
                    # Color by depth (closer = red, farther = blue)
                    max_depth = depths_in_view.max()
                    min_depth = depths_in_view.min()
                    if max_depth > min_depth:
                        norm_depth = (depth - min_depth) / (max_depth - min_depth)
                        color = (int(255 * (1 - norm_depth)), 0, int(255 * norm_depth))
                    else:
                        color = (0, 255, 0)
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 1, color, -1)

                viz_output = output_dir / f"frame_{frame_id:03d}_{camera}_full_fusion.jpg"
                cv2.imwrite(str(viz_output), img)
                print(f"✅ Saved: {viz_output}")

        return {
            'sequence': f"{device}_{sequence}",
            'frame_id': frame_id,
            'camera': camera,
            'method': 'full_scene',
            'total_points': lidar_in_view.shape[0],
            'output_pcd': str(pcd_output)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sensor Fusion without YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use tracking data
  python sensor_fusion_no_yolo.py --sequence seq-6 --device 105 --mode tracking

  # Manual bounding box
  python sensor_fusion_no_yolo.py --sequence seq-6 --device 105 --mode manual \\
      --frame 0 --camera cam1 --bbox 100,100,500,400

  # Full scene projection
  python sensor_fusion_no_yolo.py --sequence seq-6 --device 105 --mode full \\
      --frame 0 --camera cam1
        """
    )

    parser.add_argument('--sequence', required=True, help='Sequence ID (e.g., seq-6)')
    parser.add_argument('--device', required=True, help='Device ID (e.g., 105)')
    parser.add_argument('--mode', required=True,
                       choices=['tracking', 'manual', 'full'],
                       help='Processing mode')
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame ID to process (for manual/full modes)')
    parser.add_argument('--camera', default='cam1',
                       help='Camera ID (cam0 or cam1)')
    parser.add_argument('--bbox', type=str,
                       help='Bounding box as x1,y1,x2,y2 (for manual mode)')
    parser.add_argument('--correspondence',
                       default='data/RCooper/corespondence_camera_lidar.json',
                       help='Path to correspondence file')
    parser.add_argument('--data-path',
                       default='data/RCooper',
                       help='Base data path')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: sensor_fusion_output_no_yolo/DEVICE_SEQUENCE)')

    args = parser.parse_args()

    print("🚀 Sensor Fusion WITHOUT YOLO")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Sequence: {args.device}_{args.sequence}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"sensor_fusion_output_no_yolo/{args.device}_{args.sequence}")

    # Initialize fusion
    fusion = SensorFusionNoYOLO(
        base_data_path=args.data_path,
        correspondence_file=args.correspondence if args.mode in ['manual', 'full'] else None
    )

    # Process based on mode
    result = None

    if args.mode == 'tracking':
        result = fusion.process_with_tracking_data(args.sequence, args.device, output_dir)

    elif args.mode == 'manual':
        if not args.bbox:
            print("❌ --bbox required for manual mode (format: x1,y1,x2,y2)")
            return 1

        bbox = [int(x) for x in args.bbox.split(',')]
        if len(bbox) != 4:
            print("❌ Invalid bbox format. Use: x1,y1,x2,y2")
            return 1

        result = fusion.process_with_manual_bbox(
            args.sequence, args.device, args.camera, args.frame, bbox, output_dir
        )

    elif args.mode == 'full':
        result = fusion.process_full_scene(
            args.sequence, args.device, args.camera, args.frame, output_dir
        )

    if result:
        print("\n" + "=" * 60)
        print("✅ Processing complete!")
        print(f"📁 Output: {output_dir}")
        print("=" * 60)
        return 0
    else:
        print("\n❌ Processing failed")
        return 1


if __name__ == "__main__":
    exit(main())
