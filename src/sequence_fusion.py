#!/usr/bin/env python3
"""
Sequence-Based Sensor Fusion for RCooper Dataset

Processes sensor fusion by sequence ID with frames ordered by timestamp.
Outputs are organized in folders by sequence ID.

Author: Claude
Date: September 2025
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import csv

from sensor_fusion import CameraLidarFusion


class SequenceSensorFusion:
    """Sequence-based processor for sensor fusion."""

    def __init__(self, correspondence_file: str, base_data_path: str):
        """Initialize sequence processor."""
        self.correspondence_file = correspondence_file
        self.base_data_path = base_data_path
        self.output_dir = Path("sensor_fusion_output")
        self.output_dir.mkdir(exist_ok=True)

        # Load and organize correspondence data by sequence
        with open(correspondence_file, 'r') as f:
            self.correspondence_data = json.load(f)

        self.sequences = self._organize_sequences()
        print(f"Loaded {len(self.sequences)} sequences from correspondence file")

    def _organize_sequences(self) -> Dict[str, List[Dict]]:
        """Organize frames by sequence ID and device ID."""
        sequences = defaultdict(list)

        for frame in self.correspondence_data['correspondence_frames']:
            device_id = frame['device_id']
            sequence_id = frame['sequence_id']
            seq_key = f"{device_id}_{sequence_id}"
            sequences[seq_key].append(frame)

        # Sort each sequence by timestamp for chronological order
        for seq_key in sequences:
            sequences[seq_key].sort(key=lambda x: x['lidar']['timestamp'])

        return dict(sequences)

    def list_sequences(self, device_filter: Optional[str] = None) -> None:
        """List available sequences with summary information."""
        print("📋 Available Sequences:")
        print()

        filtered_sequences = {}
        for seq_key, frames in self.sequences.items():
            device_id, sequence_id = seq_key.split('_', 1)
            if device_filter and device_id != device_filter:
                continue
            filtered_sequences[seq_key] = frames

        for seq_key, frames in filtered_sequences.items():
            device_id, sequence_id = seq_key.split('_', 1)

            # Calculate sequence statistics
            first_frame = frames[0]
            last_frame = frames[-1]
            duration = last_frame['lidar']['timestamp'] - first_frame['lidar']['timestamp']
            sync_qualities = [f['synchronization']['sync_quality'] for f in frames]
            excellent_count = sum(1 for q in sync_qualities if q == 'excellent')

            print(f"Sequence {sequence_id} (Device {device_id}):")
            print(f"  📊 {len(frames)} frames ({duration:.1f}s duration)")
            print(f"  🕐 Frames {first_frame['frame_id']}-{last_frame['frame_id']}")
            print(f"  ✅ {excellent_count}/{len(frames)} excellent sync quality")
            print(f"  📁 Output: sensor_fusion_output/{seq_key}/cam0/ and cam1/")
            print()

    def process_sequence(self, sequence_id: str, device_id: str = None,
                        cameras: List[str] = ['cam0', 'cam1'],
                        max_frames: Optional[int] = None) -> List[Dict]:
        """
        Process a complete sequence with sensor fusion.

        Args:
            sequence_id: Sequence ID to process (e.g., 'seq-53')
            device_id: Device ID (if None, processes all devices for this sequence)
            cameras: List of cameras to process
            max_frames: Maximum number of frames to process (for testing)

        Returns:
            List of processing results
        """
        # Find matching sequences
        target_sequences = []
        if device_id:
            seq_key = f"{device_id}_{sequence_id}"
            if seq_key in self.sequences:
                target_sequences = [(seq_key, self.sequences[seq_key])]
            else:
                print(f"❌ Sequence {sequence_id} not found for device {device_id}")
                return []
        else:
            # Find all devices with this sequence
            for seq_key, frames in self.sequences.items():
                if seq_key.endswith(f"_{sequence_id}"):
                    target_sequences.append((seq_key, frames))

        if not target_sequences:
            print(f"❌ No sequences found matching '{sequence_id}'")
            return []

        print(f"🚀 Processing sequence '{sequence_id}' on {len(target_sequences)} device(s)")
        print(f"📷 Cameras: {cameras}")
        if max_frames:
            print(f"🔢 Max frames: {max_frames}")

        # Initialize fusion system
        fusion_system = CameraLidarFusion(
            correspondence_file=self.correspondence_file,
            base_data_path=self.base_data_path
        )

        all_results = []

        for seq_key, frames in target_sequences:
            device_id_current = seq_key.split('_')[0]

            # Limit frames if specified
            if max_frames:
                frames = frames[:max_frames]

            print(f"\n📂 Processing {seq_key}: {len(frames)} frames")

            # Create sequence output directory
            seq_output_dir = self.output_dir / seq_key
            seq_output_dir.mkdir(exist_ok=True)

            total_combinations = len(frames) * len(cameras)
            processed = 0
            start_time = time.time()

            for i, frame in enumerate(frames):
                frame_id = frame['frame_id']
                timestamp = frame['lidar']['timestamp']

                for camera in cameras:
                    processed += 1
                    print(f"  [{processed}/{total_combinations}] Frame {frame_id:03d}, {camera.upper()} "
                          f"(t={timestamp:.3f})")

                    try:
                        # Pass the specific frame data to avoid frame_id conflicts across sequences
                        result = fusion_system.process_frame_data(
                            frame_data=frame,
                            camera=camera,
                            visualize=True
                        )

                        cars = result['car_detections']
                        fusion_points = sum(f['num_lidar_points'] for f in result['fusion_results'])

                        print(f"    ✅ {cars} cars, {fusion_points} LIDAR points")

                        # Store comprehensive result
                        result_summary = {
                            'sequence_id': sequence_id,
                            'device_id': device_id_current,
                            'frame_id': frame_id,
                            'frame_index': i,
                            'camera': camera,
                            'timestamp': timestamp,
                            'car_detections': cars,
                            'total_lidar_points': result['total_lidar_points'],
                            'projected_points': result['projected_points'],
                            'fusion_points': fusion_points,
                            'sync_quality': frame['synchronization']['sync_quality'],
                            'processing_success': True
                        }

                        all_results.append(result_summary)

                    except Exception as e:
                        print(f"    ❌ Error: {e}")

                        error_result = {
                            'sequence_id': sequence_id,
                            'device_id': device_id_current,
                            'frame_id': frame_id,
                            'frame_index': i,
                            'camera': camera,
                            'timestamp': timestamp,
                            'car_detections': 0,
                            'total_lidar_points': 0,
                            'projected_points': 0,
                            'fusion_points': 0,
                            'sync_quality': frame['synchronization']['sync_quality'],
                            'processing_success': False,
                            'error': str(e)
                        }

                        all_results.append(error_result)

            elapsed_time = time.time() - start_time
            print(f"  ⏱️  Sequence {seq_key} completed in {elapsed_time:.1f}s")

        return all_results

    def save_sequence_results(self, results: List[Dict], sequence_id: str):
        """Save sequence processing results to CSV."""
        if not results:
            print("No results to save")
            return

        csv_filename = f"sequence_{sequence_id}_results_{int(time.time())}.csv"
        csv_path = self.output_dir / csv_filename

        fieldnames = sorted(results[0].keys())

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"📊 Results saved to: {csv_path}")

        # Print summary statistics
        successful = [r for r in results if r.get('processing_success', False)]
        total_cars = sum(r.get('car_detections', 0) for r in successful)
        total_fusion_points = sum(r.get('fusion_points', 0) for r in successful)

        # Group by camera for detailed stats
        cam_stats = defaultdict(list)
        for r in successful:
            cam_stats[r['camera']].append(r)

        print(f"\n📈 Sequence '{sequence_id}' Summary:")
        print(f"   ✅ Successful processings: {len(successful)}/{len(results)}")
        print(f"   🚗 Total cars detected: {total_cars}")
        print(f"   📡 Total LIDAR fusion points: {total_fusion_points}")

        if successful:
            avg_cars = total_cars / len(successful)
            avg_fusion = total_fusion_points / len(successful)
            print(f"   📊 Average per frame: {avg_cars:.1f} cars, {avg_fusion:.1f} fusion points")

        for camera, cam_results in cam_stats.items():
            cam_cars = sum(r['car_detections'] for r in cam_results)
            cam_fusion = sum(r['fusion_points'] for r in cam_results)
            print(f"   📷 {camera.upper()}: {cam_cars} cars, {cam_fusion} fusion points ({len(cam_results)} frames)")


def main():
    """Main function for sequence-based sensor fusion."""
    parser = argparse.ArgumentParser(description='Sequence-Based Sensor Fusion for RCooper Dataset')

    parser.add_argument('--sequence', type=str, required=False,
                       help='Sequence ID to process (e.g., seq-53)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device ID filter (e.g., 105). If not specified, processes all devices for the sequence')
    parser.add_argument('--cameras', nargs='+', default=['cam0', 'cam1'],
                       choices=['cam0', 'cam1'],
                       help='Cameras to process (default: both)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process (for testing)')
    parser.add_argument('--list_sequences', action='store_true',
                       help='List all available sequences and exit')
    parser.add_argument('--list_device', type=str, default=None,
                       help='List sequences for specific device only')
    parser.add_argument('--correspondence_file', type=str,
                       default='data/RCooper/corespondence_camera_lidar.json',
                       help='Path to correspondence file')
    parser.add_argument('--data_path', type=str, default='data/RCooper',
                       help='Base data directory path')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save detailed results to CSV file')

    args = parser.parse_args()

    try:
        # Initialize sequence processor
        processor = SequenceSensorFusion(
            correspondence_file=args.correspondence_file,
            base_data_path=args.data_path
        )

        # List sequences if requested
        if args.list_sequences:
            processor.list_sequences(device_filter=args.list_device)
            return 0

        # Process specified sequence
        if not args.sequence:
            print("❌ Sequence ID is required. Use --list_sequences to see available sequences.")
            return 1

        results = processor.process_sequence(
            sequence_id=args.sequence,
            device_id=args.device,
            cameras=args.cameras,
            max_frames=args.max_frames
        )

        if not results:
            print("❌ No results generated")
            return 1

        # Save results if requested
        if args.save_csv:
            processor.save_sequence_results(results, args.sequence)

        print(f"\n✅ Sequence processing completed!")
        print(f"📁 Check output folders in: sensor_fusion_output/")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())