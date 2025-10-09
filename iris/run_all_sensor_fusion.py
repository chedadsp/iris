#!/usr/bin/env python3
"""
Batch Sensor Fusion Runner for RCooper Dataset

Runs sensor fusion on multiple frames from the RCooper dataset.
Supports processing ranges of frames, specific devices, and different cameras.

Author: Claude
Date: September 2025
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
import csv

from sensor_fusion import CameraLidarFusion


class BatchSensorFusion:
    """Batch processor for sensor fusion on multiple frames."""

    def __init__(self, correspondence_file: str, base_data_path: str):
        """Initialize batch processor."""
        self.correspondence_file = correspondence_file
        self.base_data_path = base_data_path
        self.output_dir = Path("sensor_fusion_output")
        self.output_dir.mkdir(exist_ok=True)

        # Load correspondence data to get frame information
        with open(correspondence_file, 'r') as f:
            self.correspondence_data = json.load(f)

        self.frames = self.correspondence_data['correspondence_frames']
        print(f"Loaded {len(self.frames)} frames from correspondence file")

    def get_frame_ranges_by_device(self) -> Dict[str, List[int]]:
        """Get frame ranges organized by device ID."""
        device_frames = {}
        for frame in self.frames:
            device_id = frame['device_id']
            frame_id = frame['frame_id']

            if device_id not in device_frames:
                device_frames[device_id] = []
            device_frames[device_id].append(frame_id)

        # Sort frame IDs for each device
        for device_id in device_frames:
            device_frames[device_id].sort()

        return device_frames

    def filter_frames_by_sync_quality(self, min_quality: str = 'excellent') -> List[int]:
        """Filter frames by synchronization quality."""
        quality_order = {'poor': 0, 'acceptable': 1, 'good': 2, 'excellent': 3}
        min_quality_level = quality_order.get(min_quality, 3)

        filtered_frames = []
        for frame in self.frames:
            sync_quality = frame['synchronization']['sync_quality']
            if quality_order.get(sync_quality, 0) >= min_quality_level:
                filtered_frames.append(frame['frame_id'])

        return sorted(filtered_frames)

    def process_frame_range(self, start_frame: int, end_frame: int,
                           cameras: List[str] = ['cam0', 'cam1'],
                           device_filter: Optional[str] = None,
                           sync_quality_filter: Optional[str] = None) -> List[Dict]:
        """
        Process a range of frames with sensor fusion.

        Args:
            start_frame: Starting frame ID
            end_frame: Ending frame ID (inclusive)
            cameras: List of cameras to process ('cam0', 'cam1')
            device_filter: Only process frames from specific device
            sync_quality_filter: Only process frames with minimum sync quality

        Returns:
            List of processing results
        """
        # Initialize fusion system
        print("Initializing sensor fusion system...")
        fusion_system = CameraLidarFusion(
            correspondence_file=self.correspondence_file,
            base_data_path=self.base_data_path
        )

        # Filter frames
        target_frames = []
        for frame in self.frames:
            frame_id = frame['frame_id']

            # Check frame range
            if not (start_frame <= frame_id <= end_frame):
                continue

            # Check device filter
            if device_filter and frame['device_id'] != device_filter:
                continue

            # Check sync quality filter
            if sync_quality_filter:
                sync_quality = frame['synchronization']['sync_quality']
                quality_order = {'poor': 0, 'acceptable': 1, 'good': 2, 'excellent': 3}
                min_level = quality_order.get(sync_quality_filter, 3)
                current_level = quality_order.get(sync_quality, 0)
                if current_level < min_level:
                    continue

            target_frames.append(frame)

        print(f"Processing {len(target_frames)} frames (Frame {start_frame}-{end_frame})")
        print(f"Cameras: {cameras}")
        if device_filter:
            print(f"Device filter: {device_filter}")
        if sync_quality_filter:
            print(f"Sync quality filter: {sync_quality_filter}+")

        results = []
        total_combinations = len(target_frames) * len(cameras)
        processed = 0

        start_time = time.time()

        for frame in target_frames:
            frame_id = frame['frame_id']
            device_id = frame['device_id']

            for camera in cameras:
                try:
                    print(f"\n[{processed+1}/{total_combinations}] Processing Frame {frame_id}, {camera.upper()}, Device {device_id}")

                    # Check if files exist before processing
                    frame_data = fusion_system.get_frame_data(frame_id)
                    if not frame_data:
                        print(f"  ⚠️  Frame data not found, skipping")
                        continue

                    image_path = fusion_system.base_data_path / frame_data[camera]['file_path'].lstrip('./')
                    lidar_path = fusion_system.base_data_path / frame_data['lidar']['file_path'].lstrip('./')

                    if not image_path.exists():
                        print(f"  ⚠️  Image file not found: {image_path}, skipping")
                        continue

                    if not lidar_path.exists():
                        print(f"  ⚠️  LIDAR file not found: {lidar_path}, skipping")
                        continue

                    # Process frame
                    result = fusion_system.process_frame(
                        frame_id=frame_id,
                        camera=camera,
                        visualize=True
                    )

                    # Summarize results
                    total_detections = result['car_detections']
                    total_fusion_points = sum(f['num_lidar_points'] for f in result['fusion_results'])

                    print(f"  ✅ Cars: {total_detections}, LIDAR points: {total_fusion_points}")

                    # Store result summary
                    result_summary = {
                        'frame_id': frame_id,
                        'camera': camera,
                        'device_id': device_id,
                        'sequence_id': frame['sequence_id'],
                        'sync_quality': frame['synchronization']['sync_quality'],
                        'car_detections': total_detections,
                        'total_lidar_points': result['total_lidar_points'],
                        'projected_points': result['projected_points'],
                        'fusion_points': total_fusion_points,
                        'processing_success': True,
                        'timestamp': frame[camera]['timestamp']
                    }

                    results.append(result_summary)

                except Exception as e:
                    print(f"  ❌ Error: {e}")

                    # Store error result
                    error_result = {
                        'frame_id': frame_id,
                        'camera': camera,
                        'device_id': device_id,
                        'sequence_id': frame.get('sequence_id', 'unknown'),
                        'sync_quality': frame.get('synchronization', {}).get('sync_quality', 'unknown'),
                        'car_detections': 0,
                        'total_lidar_points': 0,
                        'projected_points': 0,
                        'fusion_points': 0,
                        'processing_success': False,
                        'error': str(e),
                        'timestamp': frame.get(camera, {}).get('timestamp', 0)
                    }

                    results.append(error_result)

                processed += 1

        elapsed_time = time.time() - start_time
        print(f"\n✅ Batch processing completed!")
        print(f"⏱️  Total time: {elapsed_time:.1f}s ({elapsed_time/max(1, processed):.1f}s per frame)")
        print(f"📊 Processed: {processed} frame-camera combinations")
        print(f"💾 Results saved to: {self.output_dir}")

        return results

    def save_results_csv(self, results: List[Dict], filename: str = None):
        """Save batch processing results to CSV file."""
        if not filename:
            filename = f"batch_fusion_results_{int(time.time())}.csv"

        csv_path = self.output_dir / filename

        if not results:
            print("No results to save")
            return

        # Get all possible field names
        fieldnames = set()
        for result in results:
            fieldnames.update(result.keys())

        fieldnames = sorted(fieldnames)

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"📊 Results saved to: {csv_path}")

        # Print summary statistics
        successful = [r for r in results if r.get('processing_success', False)]
        total_cars = sum(r.get('car_detections', 0) for r in successful)
        total_fusion_points = sum(r.get('fusion_points', 0) for r in successful)

        print(f"📈 Summary:")
        print(f"   Successful processings: {len(successful)}/{len(results)}")
        print(f"   Total cars detected: {total_cars}")
        print(f"   Total LIDAR fusion points: {total_fusion_points}")

        if successful:
            avg_cars = total_cars / len(successful)
            avg_fusion = total_fusion_points / len(successful)
            print(f"   Average cars per frame: {avg_cars:.1f}")
            print(f"   Average fusion points per frame: {avg_fusion:.1f}")


def main():
    """Main function for batch sensor fusion processing."""
    parser = argparse.ArgumentParser(description='Batch Sensor Fusion for RCooper Dataset')

    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame ID (default: 0)')
    parser.add_argument('--end_frame', type=int, default=10,
                       help='Ending frame ID (default: 10)')
    parser.add_argument('--cameras', nargs='+', default=['cam0', 'cam1'],
                       choices=['cam0', 'cam1'],
                       help='Cameras to process (default: both)')
    parser.add_argument('--device', type=str, default=None,
                       help='Filter by specific device ID (e.g., 105)')
    parser.add_argument('--sync_quality', type=str, default=None,
                       choices=['poor', 'acceptable', 'good', 'excellent'],
                       help='Minimum synchronization quality')
    parser.add_argument('--correspondence_file', type=str,
                       default='data/RCooper/corespondence_camera_lidar.json',
                       help='Path to correspondence file')
    parser.add_argument('--data_path', type=str, default='data/RCooper',
                       help='Base data directory path')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with first 5 frames, cam1 only')
    parser.add_argument('--save_csv', type=str, default=None,
                       help='Save results to specific CSV filename')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        print("🚀 Quick test mode: Processing first 5 frames with cam1 only")
        args.start_frame = 0
        args.end_frame = 4
        args.cameras = ['cam1']
        args.sync_quality = 'excellent'

    try:
        # Initialize batch processor
        processor = BatchSensorFusion(
            correspondence_file=args.correspondence_file,
            base_data_path=args.data_path
        )

        # Show dataset overview
        device_frames = processor.get_frame_ranges_by_device()
        print(f"\n📋 Dataset Overview:")
        for device_id, frame_ids in device_frames.items():
            print(f"   Device {device_id}: {len(frame_ids)} frames (Frame {min(frame_ids)}-{max(frame_ids)})")

        # Process frames
        results = processor.process_frame_range(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            cameras=args.cameras,
            device_filter=args.device,
            sync_quality_filter=args.sync_quality
        )

        # Save results
        processor.save_results_csv(results, args.save_csv)

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())