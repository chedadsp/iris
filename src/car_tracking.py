#!/usr/bin/env python3
"""
Car Tracking System for RCooper Dataset Sequences

This module implements multi-frame car tracking using YOLO detections and LIDAR data.
It analyzes sequences to track individual vehicles across time, providing trajectory
analysis and movement patterns.

Features:
- Multi-frame car detection and tracking
- LIDAR point association with tracked vehicles
- Trajectory analysis and visualization
- Speed and movement pattern analysis
- Integration with existing sensor fusion pipeline
"""

import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

# Import existing sensor fusion components
from sensor_fusion import CameraLidarFusion

@dataclass
class CarDetection:
    """Represents a car detection in a single frame."""
    frame_id: int
    timestamp: float
    camera: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    lidar_points: np.ndarray
    lidar_count: int
    center_3d: Optional[np.ndarray] = None  # 3D center from LIDAR points
    depth_range: Optional[Tuple[float, float]] = None

@dataclass
class CarTrack:
    """Represents a tracked car across multiple frames."""
    track_id: int
    detections: List[CarDetection]
    start_time: float
    end_time: float
    cameras: Set[str]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return len(self.detections)

    @property
    def trajectory_3d(self) -> List[np.ndarray]:
        """Get 3D trajectory points from LIDAR data."""
        return [det.center_3d for det in self.detections if det.center_3d is not None]

class CarTracker:
    """
    Multi-frame car tracking system using YOLO detections and LIDAR data.
    """

    def __init__(self,
                 correspondence_file: str = "data/RCooper/corespondence_camera_lidar.json",
                 data_path: str = "data/RCooper",
                 max_distance_2d: float = 100.0,  # pixels
                 max_distance_3d: float = 5.0,    # meters
                 max_time_gap: float = 1.0):      # seconds
        """
        Initialize car tracker.

        Args:
            correspondence_file: Path to camera-LIDAR correspondence file
            data_path: Base path to RCooper data
            max_distance_2d: Maximum 2D distance for track association (pixels)
            max_distance_3d: Maximum 3D distance for track association (meters)
            max_time_gap: Maximum time gap for track association (seconds)
        """
        self.correspondence_file = Path(correspondence_file)
        self.data_path = Path(data_path)
        self.max_distance_2d = max_distance_2d
        self.max_distance_3d = max_distance_3d
        self.max_time_gap = max_time_gap

        # Initialize sensor fusion system
        self.fusion_system = CameraLidarFusion(
            correspondence_file=str(self.correspondence_file),
            base_data_path=str(self.data_path)
        )

        # Tracking state
        self.tracks: List[CarTrack] = []
        self.next_track_id = 1

        # Load correspondence data
        self._load_correspondence_data()

    def _load_correspondence_data(self) -> None:
        """Load camera-LIDAR correspondence data."""
        with open(self.correspondence_file, 'r') as f:
            self.correspondence_data = json.load(f)
        print(f"Loaded {len(self.correspondence_data['correspondence_frames'])} frames for tracking")

    def process_sequence(self, sequence_id: str, device_id: str) -> List[CarTrack]:
        """
        Process a sequence and track cars across all frames.

        Args:
            sequence_id: Sequence ID to process (e.g., 'seq-53')
            device_id: Device ID to filter by (e.g., '105')

        Returns:
            List of car tracks found in the sequence
        """
        print(f"🚗 Starting car tracking for sequence '{sequence_id}' on device {device_id}")

        # Get sequence frames
        sequence_frames = self._get_sequence_frames(sequence_id, device_id)
        if not sequence_frames:
            print(f"❌ No frames found for {device_id}_{sequence_id}")
            return []

        print(f"📊 Processing {len(sequence_frames)} frames for tracking")

        # Reset tracking state
        self.tracks = []
        self.next_track_id = 1

        # Process each frame
        all_detections = []
        for i, frame in enumerate(sequence_frames):
            print(f"  Processing frame {i+1}/{len(sequence_frames)}: {frame['frame_id']}")

            # Process both cameras
            for camera in ['cam0', 'cam1']:
                detections = self._process_frame_for_tracking(frame, camera)
                all_detections.extend(detections)

        # Perform tracking across all detections
        self._perform_tracking(all_detections)

        print(f"✅ Found {len(self.tracks)} car tracks in sequence")

        return self.tracks

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

    def _process_frame_for_tracking(self, frame_data: Dict, camera: str) -> List[CarDetection]:
        """Process a single frame and extract car detections."""
        try:
            # Use existing sensor fusion to get detections
            result = self.fusion_system.process_frame_data(
                frame_data=frame_data,
                camera=camera,
                visualize=False  # Skip visualization for tracking
            )

            # Convert results to CarDetection objects
            detections = []
            if 'detections' in result:
                for i, detection in enumerate(result['detections']):
                    car_detection = CarDetection(
                        frame_id=frame_data['frame_id'],
                        timestamp=frame_data['reference_timestamp'],
                        camera=camera,
                        bbox=detection['bbox'],
                        confidence=detection['confidence'],
                        lidar_points=detection['lidar_points'],
                        lidar_count=len(detection['lidar_points']),
                        center_3d=self._calculate_3d_center(detection['lidar_points']),
                        depth_range=detection.get('depth_range')
                    )
                    detections.append(car_detection)

            return detections

        except Exception as e:
            print(f"    ⚠️  Error processing frame {frame_data['frame_id']} {camera}: {e}")
            return []

    def _calculate_3d_center(self, lidar_points: np.ndarray) -> Optional[np.ndarray]:
        """Calculate 3D center from LIDAR points."""
        if len(lidar_points) == 0:
            return None

        # Use median for robustness against outliers
        center = np.median(lidar_points, axis=0)
        return center

    def _perform_tracking(self, all_detections: List[CarDetection]) -> None:
        """Perform multi-frame tracking on all detections."""
        if not all_detections:
            return

        # Sort detections by timestamp
        all_detections.sort(key=lambda x: x.timestamp)

        print(f"  Performing tracking on {len(all_detections)} detections...")

        # Initialize tracks with first frame detections
        current_time = all_detections[0].timestamp
        active_tracks: List[List[CarDetection]] = []

        for detection in all_detections:
            # If too much time has passed, finalize tracks
            if detection.timestamp - current_time > self.max_time_gap:
                self._finalize_tracks(active_tracks)
                active_tracks = []

            current_time = detection.timestamp

            # Try to associate detection with existing tracks
            best_track_idx = self._find_best_track_match(detection, active_tracks)

            if best_track_idx is not None:
                # Add to existing track
                active_tracks[best_track_idx].append(detection)
            else:
                # Start new track
                active_tracks.append([detection])

        # Finalize remaining tracks
        self._finalize_tracks(active_tracks)

    def _find_best_track_match(self, detection: CarDetection,
                              active_tracks: List[List[CarDetection]]) -> Optional[int]:
        """Find the best matching track for a detection."""
        if not active_tracks:
            return None

        best_score = float('inf')
        best_idx = None

        for i, track_detections in enumerate(active_tracks):
            if not track_detections:
                continue

            # Get most recent detection in track
            last_detection = track_detections[-1]

            # Skip if cameras don't match (for now - could extend to cross-camera tracking)
            if last_detection.camera != detection.camera:
                continue

            # Check time gap
            time_gap = detection.timestamp - last_detection.timestamp
            if time_gap > self.max_time_gap:
                continue

            # Calculate similarity score
            score = self._calculate_track_similarity(last_detection, detection)

            if score < best_score:
                best_score = score
                best_idx = i

        # Only return match if score is reasonable
        if best_score < self.max_distance_2d:  # Use 2D threshold as primary
            return best_idx

        return None

    def _calculate_track_similarity(self, det1: CarDetection, det2: CarDetection) -> float:
        """Calculate similarity score between two detections."""
        # 2D bbox center distance (primary metric)
        center1 = np.array([(det1.bbox[0] + det1.bbox[2]) / 2,
                           (det1.bbox[1] + det1.bbox[3]) / 2])
        center2 = np.array([(det2.bbox[0] + det2.bbox[2]) / 2,
                           (det2.bbox[1] + det2.bbox[3]) / 2])

        distance_2d = np.linalg.norm(center1 - center2)

        # 3D distance (if available)
        distance_3d = 0.0
        if det1.center_3d is not None and det2.center_3d is not None:
            distance_3d = np.linalg.norm(det1.center_3d - det2.center_3d)

            # If 3D distance is too large, reject
            if distance_3d > self.max_distance_3d:
                return float('inf')

        # Combine scores (2D distance is primary)
        return distance_2d + distance_3d * 10  # Weight 3D distance more

    def _finalize_tracks(self, active_tracks: List[List[CarDetection]]) -> None:
        """Convert active track lists to CarTrack objects."""
        for track_detections in active_tracks:
            if len(track_detections) >= 2:  # Require at least 2 detections
                track = CarTrack(
                    track_id=self.next_track_id,
                    detections=track_detections,
                    start_time=track_detections[0].timestamp,
                    end_time=track_detections[-1].timestamp,
                    cameras={det.camera for det in track_detections}
                )
                self.tracks.append(track)
                self.next_track_id += 1

    def analyze_tracks(self, tracks: List[CarTrack]) -> Dict:
        """Analyze tracking results and provide statistics."""
        if not tracks:
            return {"total_tracks": 0}

        analysis = {
            "total_tracks": len(tracks),
            "track_durations": [track.duration for track in tracks],
            "track_lengths": [track.frame_count for track in tracks],
            "cameras_used": set(),
            "average_duration": np.mean([track.duration for track in tracks]),
            "average_length": np.mean([track.frame_count for track in tracks]),
            "tracks_by_camera": {}
        }

        # Camera statistics
        for track in tracks:
            analysis["cameras_used"].update(track.cameras)
            for camera in track.cameras:
                if camera not in analysis["tracks_by_camera"]:
                    analysis["tracks_by_camera"][camera] = 0
                analysis["tracks_by_camera"][camera] += 1

        return analysis

    def visualize_tracks(self, tracks: List[CarTrack], sequence_key: str,
                        save_output: bool = True) -> None:
        """Create visualization of car tracks."""
        if not tracks:
            print("No tracks to visualize")
            return

        # Create output directory
        output_dir = Path("sensor_fusion_output") / "car_tracking" / sequence_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Timeline visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Track timeline
        colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))
        for i, track in enumerate(tracks):
            y_pos = i
            duration = track.duration
            ax1.barh(y_pos, duration, left=track.start_time,
                    color=colors[i], alpha=0.7,
                    label=f"Track {track.track_id} ({track.frame_count} frames)")

        ax1.set_xlabel('Time (seconds from start)')
        ax1.set_ylabel('Track ID')
        ax1.set_title(f'Car Track Timeline - {sequence_key}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Detection count per track
        track_ids = [track.track_id for track in tracks]
        detection_counts = [track.frame_count for track in tracks]

        ax2.bar(track_ids, detection_counts, color=colors[:len(tracks)])
        ax2.set_xlabel('Track ID')
        ax2.set_ylabel('Number of Detections')
        ax2.set_title('Detections per Track')

        plt.tight_layout()

        if save_output:
            timeline_path = output_dir / f"{sequence_key}_track_timeline.png"
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            print(f"Track timeline saved: {timeline_path}")
        else:
            plt.show()

        plt.close()

        # 2. 3D trajectory visualization (if available)
        self._visualize_3d_trajectories(tracks, sequence_key, output_dir, save_output)

    def _visualize_3d_trajectories(self, tracks: List[CarTrack], sequence_key: str,
                                  output_dir: Path, save_output: bool) -> None:
        """Visualize 3D trajectories of tracked cars."""
        try:
            import pyvista as pv
            from point_cloud.vtk_utils import VTKSafetyManager
            from point_cloud.config import VTKConfig
        except ImportError:
            print("PyVista not available for 3D visualization")
            return

        # Collect all 3D trajectories
        trajectories_3d = []
        for track in tracks:
            traj = track.trajectory_3d
            if len(traj) >= 2:  # Need at least 2 points for trajectory
                trajectories_3d.append((track.track_id, np.array(traj)))

        if not trajectories_3d:
            print("No 3D trajectories available")
            return

        # Create 3D visualization
        VTKSafetyManager.setup_vtk_environment()
        plotter = VTKSafetyManager.create_safe_plotter(
            off_screen=save_output,
            window_size=VTKConfig.DEFAULT_WINDOW_SIZE
        )

        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories_3d)))

        for i, (track_id, trajectory) in enumerate(trajectories_3d):
            # Create trajectory line
            if len(trajectory) > 1:
                line = pv.Line(trajectory[0], trajectory[-1])
                plotter.add_mesh(line, color=colors[i][:3], line_width=3,
                               label=f"Track {track_id}")

            # Add trajectory points
            points = pv.PolyData(trajectory)
            plotter.add_mesh(points, color=colors[i][:3], point_size=8,
                           render_points_as_spheres=True)

        plotter.add_axes()
        plotter.add_legend()
        plotter.add_title(f"3D Car Trajectories - {sequence_key}")

        if save_output:
            trajectory_path = output_dir / f"{sequence_key}_3d_trajectories.png"
            plotter.screenshot(str(trajectory_path))
            print(f"3D trajectories saved: {trajectory_path}")
        else:
            plotter.show()

        VTKSafetyManager.cleanup_vtk_plotter(plotter)

    def save_tracks(self, tracks: List[CarTrack], sequence_key: str) -> None:
        """Save tracking results to files."""
        output_dir = Path("sensor_fusion_output") / "car_tracking" / sequence_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save tracks as JSON
        tracks_data = []
        for track in tracks:
            track_data = {
                "track_id": track.track_id,
                "start_time": track.start_time,
                "end_time": track.end_time,
                "duration": track.duration,
                "frame_count": track.frame_count,
                "cameras": list(track.cameras),
                "detections": []
            }

            for detection in track.detections:
                det_data = {
                    "frame_id": detection.frame_id,
                    "timestamp": detection.timestamp,
                    "camera": detection.camera,
                    "bbox": detection.bbox,
                    "confidence": detection.confidence,
                    "lidar_count": detection.lidar_count,
                    "center_3d": detection.center_3d.tolist() if detection.center_3d is not None else None,
                    "depth_range": detection.depth_range
                }
                track_data["detections"].append(det_data)

            tracks_data.append(track_data)

        # Save JSON file
        json_path = output_dir / f"{sequence_key}_tracks.json"
        with open(json_path, 'w') as f:
            json.dump(tracks_data, f, indent=2)

        print(f"Tracks saved: {json_path}")


def main():
    """Main function for car tracking."""
    parser = argparse.ArgumentParser(description="Car Tracking for RCooper Dataset Sequences")
    parser.add_argument("--sequence", required=True, help="Sequence ID to analyze (e.g., seq-53)")
    parser.add_argument("--device", required=True, help="Device ID filter (e.g., 105)")
    parser.add_argument("--correspondence_file", default="data/RCooper/corespondence_camera_lidar.json",
                       help="Path to correspondence file")
    parser.add_argument("--data_path", default="data/RCooper", help="Base data directory path")
    parser.add_argument("--max_distance_2d", type=float, default=100.0,
                       help="Maximum 2D distance for track association (pixels)")
    parser.add_argument("--max_distance_3d", type=float, default=5.0,
                       help="Maximum 3D distance for track association (meters)")
    parser.add_argument("--max_time_gap", type=float, default=1.0,
                       help="Maximum time gap for track association (seconds)")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization output")

    args = parser.parse_args()

    # Initialize tracker
    tracker = CarTracker(
        correspondence_file=args.correspondence_file,
        data_path=args.data_path,
        max_distance_2d=args.max_distance_2d,
        max_distance_3d=args.max_distance_3d,
        max_time_gap=args.max_time_gap
    )

    # Process sequence
    sequence_key = f"{args.device}_{args.sequence}"
    start_time = time.time()

    tracks = tracker.process_sequence(args.sequence, args.device)

    processing_time = time.time() - start_time

    if tracks:
        # Analyze results
        analysis = tracker.analyze_tracks(tracks)

        print(f"\n🎯 Tracking Results for {sequence_key}:")
        print(f"    Total tracks: {analysis['total_tracks']}")
        print(f"    Average duration: {analysis['average_duration']:.2f}s")
        print(f"    Average detections per track: {analysis['average_length']:.1f}")
        print(f"    Cameras used: {list(analysis['cameras_used'])}")
        print(f"    Processing time: {processing_time:.1f}s")

        # Save results
        tracker.save_tracks(tracks, sequence_key)

        # Create visualizations
        if not args.no_viz:
            tracker.visualize_tracks(tracks, sequence_key, save_output=True)

        print(f"\n✅ Car tracking completed!")
        print(f"📁 Results saved in: sensor_fusion_output/car_tracking/{sequence_key}/")

    else:
        print(f"❌ No car tracks found in {sequence_key}")


if __name__ == "__main__":
    main()