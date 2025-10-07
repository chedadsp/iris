#!/usr/bin/env python3
"""
Simple Batch Sensor Fusion Runner

A streamlined script to run sensor fusion on a specific range of frames.
Saves all outputs to sensor_fusion_output folder.

Author: Claude
Date: September 2025
"""

import argparse
import json
from pathlib import Path
from sensor_fusion import CameraLidarFusion


def main():
    """Simple batch processing for sensor fusion."""
    parser = argparse.ArgumentParser(description='Simple Batch Sensor Fusion')
    parser.add_argument('--start', type=int, default=0, help='Start frame ID (default: 0)')
    parser.add_argument('--end', type=int, default=4, help='End frame ID (default: 4)')
    parser.add_argument('--cameras', nargs='+', default=['cam0', 'cam1'],
                       choices=['cam0', 'cam1'], help='Cameras to process')
    parser.add_argument('--device', type=str, default='105', help='Device ID (default: 105)')

    args = parser.parse_args()

    print(f"🚀 Simple Batch Sensor Fusion")
    print(f"📝 Processing frames {args.start}-{args.end} with {args.cameras} on device {args.device}")

    # Note: Output directories are now created by sequence in the sensor fusion system
    print("📁 Outputs will be organized by sequence ID in sensor_fusion_output/device_sequence/ folders")

    # Initialize fusion system
    fusion_system = CameraLidarFusion(
        correspondence_file='data/RCooper/corespondence_camera_lidar.json',
        base_data_path='data/RCooper'
    )

    # Load correspondence data to find unique frames for the specified device
    with open('data/RCooper/corespondence_camera_lidar.json', 'r') as f:
        data = json.load(f)

    # Filter frames by device and frame range
    target_frames = []
    for frame in data['correspondence_frames']:
        if (frame['device_id'] == args.device and
            args.start <= frame['frame_id'] <= args.end):
            # Avoid duplicates by checking if frame_id is already in target_frames
            if not any(f['frame_id'] == frame['frame_id'] for f in target_frames):
                target_frames.append(frame)

    print(f"📊 Found {len(target_frames)} unique frames for device {args.device}")

    results = []
    total_combinations = len(target_frames) * len(args.cameras)
    processed = 0

    for frame in target_frames:
        frame_id = frame['frame_id']

        for camera in args.cameras:
            processed += 1
            print(f"\n[{processed}/{total_combinations}] Processing Frame {frame_id}, {camera.upper()}")

            try:
                result = fusion_system.process_frame(
                    frame_id=frame_id,
                    camera=camera,
                    visualize=True
                )

                cars = result['car_detections']
                fusion_points = sum(f['num_lidar_points'] for f in result['fusion_results'])

                print(f"  ✅ Success: {cars} cars, {fusion_points} LIDAR fusion points")

                results.append({
                    'frame_id': frame_id,
                    'camera': camera,
                    'device_id': args.device,
                    'cars': cars,
                    'fusion_points': fusion_points,
                    'success': True
                })

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'frame_id': frame_id,
                    'camera': camera,
                    'device_id': args.device,
                    'cars': 0,
                    'fusion_points': 0,
                    'success': False,
                    'error': str(e)
                })

    # Print summary
    successful = [r for r in results if r['success']]
    total_cars = sum(r['cars'] for r in successful)
    total_fusion = sum(r['fusion_points'] for r in successful)

    print(f"\n✅ Batch processing completed!")
    print(f"📊 Summary:")
    print(f"   Successful: {len(successful)}/{len(results)}")
    print(f"   Total cars detected: {total_cars}")
    print(f"   Total LIDAR fusion points: {total_fusion}")
    print(f"   Output folder: {output_dir}")

    # List generated files
    jpg_files = list(output_dir.glob("*.jpg"))
    print(f"   Generated {len(jpg_files)} visualization images")

    return 0


if __name__ == "__main__":
    exit(main())