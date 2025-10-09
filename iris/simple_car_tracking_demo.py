#!/usr/bin/env python3
"""
Simple Car Tracking Demo

This script demonstrates basic car tracking capabilities by analyzing
YOLO detections across a short sequence and showing vehicle movement
patterns and statistics.

This is a simplified version that focuses on detection analysis rather
than complex multi-frame tracking algorithms.
"""

import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Import existing sensor fusion components
from sensor_fusion import CameraLidarFusion

class SimpleCarTracker:
    """
    Simple car tracking demonstration using YOLO detections.
    Focuses on detection analysis and basic movement patterns.
    """

    def __init__(self,
                 correspondence_file: str = "data/RCooper/corespondence_camera_lidar.json",
                 data_path: str = "data/RCooper"):
        """Initialize simple car tracker."""
        self.correspondence_file = Path(correspondence_file)
        self.data_path = Path(data_path)

        # Initialize sensor fusion system
        self.fusion_system = CameraLidarFusion(
            correspondence_file=str(self.correspondence_file),
            base_data_path=str(self.data_path)
        )

        # Load correspondence data
        self._load_correspondence_data()

    def _load_correspondence_data(self) -> None:
        """Load camera-LIDAR correspondence data."""
        with open(self.correspondence_file, 'r') as f:
            self.correspondence_data = json.load(f)
        print(f"Loaded {len(self.correspondence_data['correspondence_frames'])} frames for analysis")

    def analyze_sequence_detections(self, sequence_id: str, device_id: str, max_frames: int = 10) -> Dict:
        """
        Analyze car detections across a sequence.

        Args:
            sequence_id: Sequence ID to analyze (e.g., 'seq-53')
            device_id: Device ID to filter by (e.g., '105')
            max_frames: Maximum number of frames to analyze

        Returns:
            Dictionary with detection analysis results
        """
        print(f"🚗 Analyzing car detections for sequence '{sequence_id}' on device {device_id}")

        # Get sequence frames
        sequence_frames = self._get_sequence_frames(sequence_id, device_id)
        if not sequence_frames:
            print(f"❌ No frames found for {device_id}_{sequence_id}")
            return {}

        # Limit frames for demo
        if max_frames and len(sequence_frames) > max_frames:
            sequence_frames = sequence_frames[:max_frames]

        print(f"📊 Analyzing {len(sequence_frames)} frames")

        # Analyze detections in each frame
        analysis_results = {
            'sequence_key': f"{device_id}_{sequence_id}",
            'total_frames': len(sequence_frames),
            'frame_data': [],
            'cameras': ['cam0', 'cam1'],
            'detection_stats': {
                'cam0': {'total_detections': 0, 'frames_with_cars': 0, 'avg_lidar_points': 0},
                'cam1': {'total_detections': 0, 'frames_with_cars': 0, 'avg_lidar_points': 0}
            }
        }

        all_lidar_points = {'cam0': [], 'cam1': []}

        for i, frame in enumerate(sequence_frames):
            print(f"  Analyzing frame {i+1}/{len(sequence_frames)}: {frame['frame_id']}")

            frame_result = {
                'frame_id': frame['frame_id'],
                'timestamp': frame['reference_timestamp'],
                'cameras': {}
            }

            # Process both cameras
            for camera in ['cam0', 'cam1']:
                try:
                    # Get detections for this frame and camera
                    result = self.fusion_system.process_frame_data(
                        frame_data=frame,
                        camera=camera,
                        visualize=False
                    )

                    # Extract detection info
                    detections = result.get('fusion_results', [])
                    frame_result['cameras'][camera] = {
                        'car_count': len(detections),
                        'detections': []
                    }

                    # Update statistics
                    stats = analysis_results['detection_stats'][camera]
                    stats['total_detections'] += len(detections)
                    if len(detections) > 0:
                        stats['frames_with_cars'] += 1

                    # Process each detection
                    for det in detections:
                        detection_info = {
                            'bbox': det.get('bbox', [0, 0, 0, 0]),
                            'confidence': det.get('confidence', 0.0),
                            'lidar_count': det.get('lidar_point_count', 0),
                            'depth_range': det.get('depth_range'),
                            'center_3d': self._calculate_3d_center_from_dict(det)
                        }
                        frame_result['cameras'][camera]['detections'].append(detection_info)
                        all_lidar_points[camera].append(det.get('lidar_point_count', 0))

                except Exception as e:
                    print(f"    ⚠️  Error processing frame {frame['frame_id']} {camera}: {e}")
                    frame_result['cameras'][camera] = {'car_count': 0, 'detections': []}

            analysis_results['frame_data'].append(frame_result)

        # Calculate final statistics
        for camera in ['cam0', 'cam1']:
            stats = analysis_results['detection_stats'][camera]
            if all_lidar_points[camera]:
                stats['avg_lidar_points'] = np.mean(all_lidar_points[camera])
            stats['detection_rate'] = stats['frames_with_cars'] / len(sequence_frames) if sequence_frames else 0

        return analysis_results

    def _get_sequence_frames(self, sequence_id: str, device_id: str) -> List[Dict]:
        """Get all frames for a specific sequence and device."""
        sequence_frames = []

        for frame in self.correspondence_data['correspondence_frames']:
            if (frame['sequence_id'] == sequence_id and
                frame['device_id'] == device_id):
                sequence_frames.append(frame)

        # Sort by timestamp
        sequence_frames.sort(key=lambda x: x['reference_timestamp'])
        return sequence_frames

    def _calculate_3d_center(self, lidar_points: np.ndarray) -> List[float]:
        """Calculate 3D center from LIDAR points."""
        if len(lidar_points) == 0:
            return [0, 0, 0]

        # Use median for robustness against outliers
        center = np.median(lidar_points, axis=0)
        return center.tolist()

    def _calculate_3d_center_from_dict(self, detection_dict: Dict) -> List[float]:
        """Calculate 3D center from detection dictionary."""
        # The sensor fusion results may have different structure
        # Return a default center for now - this could be enhanced
        # to extract actual 3D position from the detection data
        return [0, 0, 0]

    def create_detection_visualizations(self, analysis_results: Dict, save_output: bool = True) -> None:
        """Create visualizations of detection analysis."""
        if not analysis_results:
            print("No analysis results to visualize")
            return

        sequence_key = analysis_results['sequence_key']
        frame_data = analysis_results['frame_data']

        # Create output directory
        output_dir = Path("sensor_fusion_output") / "car_tracking" / sequence_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Detection timeline
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Extract data for plotting
        frame_ids = [f['frame_id'] for f in frame_data]
        timestamps = [f['timestamp'] for f in frame_data]
        relative_times = np.array(timestamps) - timestamps[0]

        cam0_counts = [f['cameras']['cam0']['car_count'] for f in frame_data]
        cam1_counts = [f['cameras']['cam1']['car_count'] for f in frame_data]

        # Plot 1: Car detections over time
        ax1.plot(relative_times, cam0_counts, 'o-', label='CAM0', color='blue', markersize=6)
        ax1.plot(relative_times, cam1_counts, 'o-', label='CAM1', color='red', markersize=6)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Number of Cars Detected')
        ax1.set_title(f'Car Detections Over Time - {sequence_key}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: LIDAR points per detection
        cam0_lidar = []
        cam1_lidar = []
        cam0_frame_times = []
        cam1_frame_times = []

        for i, frame in enumerate(frame_data):
            for det in frame['cameras']['cam0']['detections']:
                cam0_lidar.append(det['lidar_count'])
                cam0_frame_times.append(relative_times[i])
            for det in frame['cameras']['cam1']['detections']:
                cam1_lidar.append(det['lidar_count'])
                cam1_frame_times.append(relative_times[i])

        if cam0_lidar:
            ax2.scatter(cam0_frame_times, cam0_lidar, alpha=0.6, label='CAM0', color='blue')
        if cam1_lidar:
            ax2.scatter(cam1_frame_times, cam1_lidar, alpha=0.6, label='CAM1', color='red')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('LIDAR Points per Detection')
        ax2.set_title('LIDAR Point Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Detection statistics bar chart
        stats = analysis_results['detection_stats']
        cameras = list(stats.keys())
        total_dets = [stats[cam]['total_detections'] for cam in cameras]
        frames_with_cars = [stats[cam]['frames_with_cars'] for cam in cameras]

        x = np.arange(len(cameras))
        width = 0.35

        ax3.bar(x - width/2, total_dets, width, label='Total Detections', alpha=0.8)
        ax3.bar(x + width/2, frames_with_cars, width, label='Frames with Cars', alpha=0.8)
        ax3.set_xlabel('Camera')
        ax3.set_ylabel('Count')
        ax3.set_title('Detection Statistics by Camera')
        ax3.set_xticks(x)
        ax3.set_xticklabels(cameras)
        ax3.legend()

        # Plot 4: Detection rate comparison
        detection_rates = [stats[cam]['detection_rate'] * 100 for cam in cameras]
        avg_lidar = [stats[cam]['avg_lidar_points'] for cam in cameras]

        ax4_twin = ax4.twinx()
        bars1 = ax4.bar(x - width/2, detection_rates, width, label='Detection Rate (%)', alpha=0.8, color='green')
        bars2 = ax4_twin.bar(x + width/2, avg_lidar, width, label='Avg LIDAR Points', alpha=0.8, color='orange')

        ax4.set_xlabel('Camera')
        ax4.set_ylabel('Detection Rate (%)', color='green')
        ax4_twin.set_ylabel('Average LIDAR Points', color='orange')
        ax4.set_title('Detection Rate vs LIDAR Point Density')
        ax4.set_xticks(x)
        ax4.set_xticklabels(cameras)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()

        if save_output:
            plot_path = output_dir / f"{sequence_key}_detection_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Detection analysis plots saved: {plot_path}")
        else:
            plt.show()

        plt.close()

    def save_analysis_results(self, analysis_results: Dict) -> None:
        """Save analysis results to JSON file."""
        if not analysis_results:
            return

        sequence_key = analysis_results['sequence_key']
        output_dir = Path("sensor_fusion_output") / "car_tracking" / sequence_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_dir / f"{sequence_key}_detection_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)

        print(f"Analysis results saved: {json_path}")

    def print_summary(self, analysis_results: Dict) -> None:
        """Print summary of analysis results."""
        if not analysis_results:
            return

        sequence_key = analysis_results['sequence_key']
        stats = analysis_results['detection_stats']

        print(f"\n🎯 Car Detection Analysis Summary for {sequence_key}:")
        print(f"    Total frames analyzed: {analysis_results['total_frames']}")

        for camera in ['cam0', 'cam1']:
            cam_stats = stats[camera]
            print(f"\n  📷 {camera.upper()}:")
            print(f"    Total detections: {cam_stats['total_detections']}")
            print(f"    Frames with cars: {cam_stats['frames_with_cars']}/{analysis_results['total_frames']}")
            print(f"    Detection rate: {cam_stats['detection_rate']*100:.1f}%")
            print(f"    Avg LIDAR points per detection: {cam_stats['avg_lidar_points']:.1f}")

        # Movement analysis
        frame_data = analysis_results['frame_data']
        total_detections = sum(cam_stats['total_detections'] for cam_stats in stats.values())

        if total_detections > 0:
            print(f"\n  🚗 Movement Analysis:")
            print(f"    Total car detections across sequence: {total_detections}")
            print(f"    Average cars per frame: {total_detections / analysis_results['total_frames']:.1f}")

            # Find frames with most activity
            max_activity_frame = max(frame_data, key=lambda f: sum(f['cameras'][cam]['car_count'] for cam in ['cam0', 'cam1']))
            max_cars = sum(max_activity_frame['cameras'][cam]['car_count'] for cam in ['cam0', 'cam1'])
            print(f"    Peak activity: {max_cars} cars in frame {max_activity_frame['frame_id']}")


def main():
    """Main function for simple car tracking demo."""
    parser = argparse.ArgumentParser(description="Simple Car Tracking Demo for RCooper Dataset")
    parser.add_argument("--sequence", required=True, help="Sequence ID to analyze (e.g., seq-53)")
    parser.add_argument("--device", required=True, help="Device ID filter (e.g., 105)")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum frames to analyze")
    parser.add_argument("--correspondence_file", default="data/RCooper/corespondence_camera_lidar.json",
                       help="Path to correspondence file")
    parser.add_argument("--data_path", default="data/RCooper", help="Base data directory path")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization output")

    args = parser.parse_args()

    # Initialize tracker
    tracker = SimpleCarTracker(
        correspondence_file=args.correspondence_file,
        data_path=args.data_path
    )

    # Analyze sequence
    start_time = time.time()

    analysis_results = tracker.analyze_sequence_detections(
        args.sequence, args.device, args.max_frames
    )

    processing_time = time.time() - start_time

    if analysis_results:
        # Print summary
        tracker.print_summary(analysis_results)
        print(f"    Processing time: {processing_time:.1f}s")

        # Save results
        tracker.save_analysis_results(analysis_results)

        # Create visualizations
        if not args.no_viz:
            tracker.create_detection_visualizations(analysis_results, save_output=True)

        print(f"\n✅ Car detection analysis completed!")
        print(f"📁 Results saved in: sensor_fusion_output/car_tracking/{args.device}_{args.sequence}/")

    else:
        print(f"❌ No detection data found for {args.device}_{args.sequence}")


if __name__ == "__main__":
    main()