#!/usr/bin/env python3
"""
Enhanced Car Tracking with Progressive Point Cloud Building

This module implements advanced car tracking that accumulates LIDAR points over time
to build progressively better 3D models of tracked vehicles. Each detection adds
new LIDAR points to the existing car model, creating detailed point cloud structures.

Features:
- Progressive LIDAR point accumulation for each tracked car
- Image cropping for car regions
- 3D structure building over time
- Visualization of progressive model enhancement
- Tracking with improved accuracy through accumulated data

Author: Claude
Date: September 2025
"""

import json
import argparse
import time
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import open3d as o3d

# Import existing sensor fusion components
from sensor_fusion import CameraLidarFusion

@dataclass
class AccumulatedCarModel:
    """Represents an accumulated 3D model of a tracked car."""
    track_id: int
    first_detection_time: float
    last_update_time: float
    total_detections: int

    # Accumulated point cloud data
    accumulated_points: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 3))
    point_timestamps: List[float] = field(default_factory=list)
    point_confidences: List[float] = field(default_factory=list)

    # Detection history
    detection_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    detection_confidences: List[float] = field(default_factory=list)
    detection_images: List[np.ndarray] = field(default_factory=list)
    detection_frames: List[int] = field(default_factory=list)
    detection_cameras: List[str] = field(default_factory=list)

    # Model statistics
    centroid_3d: Optional[np.ndarray] = None
    bounding_box_3d: Optional[Tuple[np.ndarray, np.ndarray]] = None  # min, max
    model_quality_score: float = 0.0

    def add_detection(self, frame_id: int, timestamp: float, camera: str, bbox: Tuple[int, int, int, int],
                     confidence: float, lidar_points: np.ndarray, cropped_image: np.ndarray) -> None:
        """Add a new detection and its LIDAR points to the accumulated model."""
        self.last_update_time = timestamp
        self.total_detections += 1

        # Store detection metadata
        self.detection_bboxes.append(bbox)
        self.detection_confidences.append(confidence)
        self.detection_images.append(cropped_image)
        self.detection_frames.append(frame_id)
        self.detection_cameras.append(camera)

        # Accumulate LIDAR points
        if len(lidar_points) > 0:
            # Filter points to remove outliers (basic cleaning)
            cleaned_points = self._filter_outliers(lidar_points)

            if len(self.accumulated_points) == 0:
                self.accumulated_points = cleaned_points
            else:
                # Combine with existing points, removing near-duplicates
                self.accumulated_points = self._merge_point_clouds(
                    self.accumulated_points, cleaned_points)

            # Store metadata for new points
            num_new_points = len(cleaned_points)
            self.point_timestamps.extend([timestamp] * num_new_points)
            self.point_confidences.extend([confidence] * num_new_points)

            # Update model statistics
            self._update_model_statistics()

    def _filter_outliers(self, points: np.ndarray, std_threshold: float = 2.0) -> np.ndarray:
        """Remove outlier points using statistical filtering."""
        if len(points) < 3:
            return points

        # Remove points that are too far from the median in any dimension
        median = np.median(points, axis=0)
        distances = np.linalg.norm(points - median, axis=1)
        std_dist = np.std(distances)
        mean_dist = np.mean(distances)

        # Keep points within std_threshold standard deviations
        mask = distances <= (mean_dist + std_threshold * std_dist)
        return points[mask]

    def _merge_point_clouds(self, existing_points: np.ndarray, new_points: np.ndarray,
                           merge_threshold: float = 0.05) -> np.ndarray:
        """Merge new points with existing ones, removing near-duplicates."""
        if len(existing_points) == 0:
            return new_points
        if len(new_points) == 0:
            return existing_points

        # Use simple distance-based merging
        # For each new point, check if it's close to existing points
        unique_new_points = []

        for new_point in new_points:
            distances = np.linalg.norm(existing_points - new_point, axis=1)
            min_distance = np.min(distances)

            # Add point if it's not too close to existing points
            if min_distance > merge_threshold:
                unique_new_points.append(new_point)

        if unique_new_points:
            return np.vstack([existing_points, np.array(unique_new_points)])
        else:
            return existing_points

    def _update_model_statistics(self) -> None:
        """Update model statistics after adding points."""
        if len(self.accumulated_points) == 0:
            return

        # Update centroid
        self.centroid_3d = np.mean(self.accumulated_points, axis=0)

        # Update bounding box
        min_bounds = np.min(self.accumulated_points, axis=0)
        max_bounds = np.max(self.accumulated_points, axis=0)
        self.bounding_box_3d = (min_bounds, max_bounds)

        # Calculate model quality score based on point density and detection count
        point_count = len(self.accumulated_points)
        detection_count = self.total_detections

        # Quality increases with more points and more detections
        self.model_quality_score = min(1.0, (point_count / 100.0) * (detection_count / 5.0))

    def get_dimensions(self) -> Optional[np.ndarray]:
        """Get estimated car dimensions (length, width, height)."""
        if self.bounding_box_3d is None:
            return None

        min_bounds, max_bounds = self.bounding_box_3d
        dimensions = max_bounds - min_bounds
        return dimensions

    def save_model(self, output_path: Path) -> None:
        """Save the accumulated car model to file."""
        if len(self.accumulated_points) == 0:
            return

        # Save as PCD file
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.accumulated_points)

        # Color points by age (newer points are redder)
        if len(self.point_timestamps) == len(self.accumulated_points):
            timestamps = np.array(self.point_timestamps)
            normalized_time = (timestamps - np.min(timestamps)) / (np.max(timestamps) - np.min(timestamps) + 1e-6)
            colors = np.zeros((len(self.accumulated_points), 3))
            colors[:, 0] = normalized_time  # Red channel increases with time
            colors[:, 2] = 1.0 - normalized_time  # Blue channel decreases with time
            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(str(output_path), pcd)


class EnhancedCarTracker:
    """
    Enhanced car tracking system with progressive point cloud building.
    """

    def __init__(self,
                 correspondence_file: str = "data/RCooper/corespondence_camera_lidar.json",
                 data_path: str = "data/RCooper",
                 max_distance_2d: float = 150.0,  # pixels
                 max_distance_3d: float = 3.0,    # meters
                 max_time_gap: float = 1.5):      # seconds
        """
        Initialize enhanced car tracker.

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
        self.car_models: Dict[int, AccumulatedCarModel] = {}
        self.active_tracks: Dict[int, float] = {}  # track_id -> last_update_time
        self.next_track_id = 1

        # Load correspondence data
        self._load_correspondence_data()

    def _load_correspondence_data(self) -> None:
        """Load camera-LIDAR correspondence data."""
        with open(self.correspondence_file, 'r') as f:
            self.correspondence_data = json.load(f)
        print(f"Loaded {len(self.correspondence_data['correspondence_frames'])} frames for enhanced tracking")

    def process_sequence_with_accumulation(self, sequence_id: str, device_id: str,
                                          max_frames: Optional[int] = None) -> Dict[int, AccumulatedCarModel]:
        """
        Process a sequence with progressive car model building.

        Args:
            sequence_id: Sequence ID to process (e.g., 'seq-53')
            device_id: Device ID to filter by (e.g., '105')
            max_frames: Maximum number of frames to process

        Returns:
            Dictionary of accumulated car models by track ID
        """
        print(f"🚗 Starting enhanced car tracking with accumulation for {device_id}_{sequence_id}")

        # Get sequence frames
        sequence_frames = self._get_sequence_frames(sequence_id, device_id)
        if not sequence_frames:
            print(f"❌ No frames found for {device_id}_{sequence_id}")
            return {}

        if max_frames:
            sequence_frames = sequence_frames[:max_frames]

        print(f"📊 Processing {len(sequence_frames)} frames for enhanced tracking")

        # Reset tracking state
        self.car_models = {}
        self.active_tracks = {}
        self.next_track_id = 1

        # Process each frame
        for i, frame in enumerate(sequence_frames):
            print(f"  Processing frame {i+1}/{len(sequence_frames)}: {frame['frame_id']}")
            current_time = frame['reference_timestamp']

            # Process both cameras
            for camera in ['cam0', 'cam1']:
                self._process_frame_for_enhanced_tracking(frame, camera, current_time)

            # Clean up stale tracks
            self._cleanup_stale_tracks(current_time)

        # Final statistics
        active_models = {tid: model for tid, model in self.car_models.items()
                        if model.total_detections >= 2}

        print(f"✅ Enhanced tracking complete:")
        print(f"    Total tracks created: {len(self.car_models)}")
        print(f"    Active tracks (2+ detections): {len(active_models)}")

        return active_models

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

    def _process_frame_for_enhanced_tracking(self, frame_data: Dict, camera: str, current_time: float) -> None:
        """Process a single frame for enhanced tracking with accumulation."""
        try:
            # Use existing sensor fusion to get detections
            result = self.fusion_system.process_frame_data(
                frame_data=frame_data,
                camera=camera,
                visualize=False
            )

            # Get detections from fusion results
            fusion_results = result.get('fusion_results', [])
            if not fusion_results:
                return

            print(f"    🔍 Found {len(fusion_results)} fusion results")

            # Load the camera image for cropping
            camera_data = frame_data[camera]
            image_path = self.data_path / camera_data['file_path']
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"    ⚠️  Could not load image: {image_path}")
                return

            # Process each detection
            current_detections = []
            for detection in fusion_results:
                # Extract detection data safely
                bbox = detection.get('bbox', [0, 0, 0, 0])
                confidence = detection.get('confidence', 0.0)

                # Get LIDAR points (based on sensor fusion structure)
                lidar_points = detection.get('lidar_points_3d', np.array([]).reshape(0, 3))

                if lidar_points is None or len(lidar_points) == 0:
                    continue

                # Ensure bbox is a proper list of integers
                try:
                    if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    else:
                        bbox = [0, 0, 100, 100]  # Default bbox if invalid
                except (ValueError, TypeError):
                    bbox = [0, 0, 100, 100]  # Default bbox if conversion fails

                # Crop image around car
                cropped_image = self._crop_car_image(image, bbox)

                # Calculate 3D centroid for tracking
                centroid_3d = np.mean(lidar_points, axis=0) if len(lidar_points) > 0 else np.array([0, 0, 0])

                current_detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'lidar_points': lidar_points,
                    'centroid_3d': centroid_3d,
                    'cropped_image': cropped_image,
                    'frame_id': frame_data['frame_id'],
                    'camera': camera
                })

                print(f"    📦 Detection: bbox={bbox}, {len(lidar_points)} points, conf={confidence:.2f}")

            # Associate detections with existing tracks
            self._associate_and_update_tracks(current_detections, current_time)

        except Exception as e:
            print(f"    ⚠️  Error processing frame {frame_data['frame_id']} {camera}: {e}")

    def _crop_car_image(self, image: np.ndarray, bbox: List[int], padding: int = 20) -> np.ndarray:
        """Crop image around car bounding box with padding."""
        if len(bbox) != 4:
            return np.array([])

        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Add padding and ensure bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return image[y1:y2, x1:x2]

    def _associate_and_update_tracks(self, detections: List[Dict], current_time: float) -> None:
        """Associate detections with existing tracks and update models."""
        if not detections:
            return

        # Get active track IDs
        active_track_ids = [tid for tid, last_time in self.active_tracks.items()
                           if current_time - last_time <= self.max_time_gap]

        for detection in detections:
            best_track_id = None
            best_score = float('inf')

            # Try to match with existing tracks
            for track_id in active_track_ids:
                if track_id not in self.car_models:
                    continue

                car_model = self.car_models[track_id]
                if car_model.centroid_3d is None:
                    continue

                # Calculate similarity score
                centroid_distance = np.linalg.norm(car_model.centroid_3d - detection['centroid_3d'])
                time_gap = current_time - car_model.last_update_time

                # Skip if too far or too much time passed
                if centroid_distance > self.max_distance_3d or time_gap > self.max_time_gap:
                    continue

                # Calculate combined score (lower is better)
                score = centroid_distance + time_gap * 0.1
                if score < best_score:
                    best_score = score
                    best_track_id = track_id

            # Update existing track or create new one
            if best_track_id is not None:
                # Update existing track
                self._update_car_model(best_track_id, detection, current_time)
            else:
                # Create new track
                self._create_new_track(detection, current_time)

    def _update_car_model(self, track_id: int, detection: Dict, current_time: float) -> None:
        """Update an existing car model with new detection."""
        car_model = self.car_models[track_id]

        car_model.add_detection(
            frame_id=detection['frame_id'],
            timestamp=current_time,
            camera=detection['camera'],
            bbox=tuple(detection['bbox']),
            confidence=detection['confidence'],
            lidar_points=detection['lidar_points'],
            cropped_image=detection['cropped_image']
        )

        self.active_tracks[track_id] = current_time

    def _create_new_track(self, detection: Dict, current_time: float) -> None:
        """Create a new car track."""
        track_id = self.next_track_id
        self.next_track_id += 1

        car_model = AccumulatedCarModel(
            track_id=track_id,
            first_detection_time=current_time,
            last_update_time=current_time,
            total_detections=0
        )

        car_model.add_detection(
            frame_id=detection['frame_id'],
            timestamp=current_time,
            camera=detection['camera'],
            bbox=tuple(detection['bbox']),
            confidence=detection['confidence'],
            lidar_points=detection['lidar_points'],
            cropped_image=detection['cropped_image']
        )

        self.car_models[track_id] = car_model
        self.active_tracks[track_id] = current_time

    def _cleanup_stale_tracks(self, current_time: float) -> None:
        """Remove tracks that haven't been updated recently."""
        stale_tracks = [tid for tid, last_time in self.active_tracks.items()
                       if current_time - last_time > self.max_time_gap]

        for track_id in stale_tracks:
            del self.active_tracks[track_id]

    def visualize_progressive_building(self, car_models: Dict[int, AccumulatedCarModel],
                                     sequence_key: str, save_output: bool = True) -> None:
        """Create visualization showing progressive point cloud building."""
        print("Creating progressive building visualization...")

        # Create output directory
        output_dir = Path("sensor_fusion_output") / "enhanced_tracking" / sequence_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter models with enough detections (reduced threshold for demo)
        quality_models = {tid: model for tid, model in car_models.items()
                         if model.total_detections >= 2 and len(model.accumulated_points) >= 10}

        if not quality_models:
            print("No quality car models found for visualization")
            return

        # 1. Create progression visualization for each car
        for track_id, car_model in quality_models.items():
            self._create_single_car_progression(car_model, output_dir, save_output)

        # 2. Create summary visualization
        self._create_summary_visualization(quality_models, output_dir, sequence_key, save_output)

        # 3. Save individual car models
        for track_id, car_model in quality_models.items():
            model_path = output_dir / f"car_model_{track_id}.pcd"
            car_model.save_model(model_path)
            print(f"Car model {track_id} saved: {model_path}")

    def _create_single_car_progression(self, car_model: AccumulatedCarModel,
                                     output_dir: Path, save_output: bool) -> None:
        """Create progression visualization for a single car."""
        if car_model.total_detections < 2:
            return

        # Create progression plot showing point accumulation
        fig, axes = plt.subplots(2, min(4, car_model.total_detections), figsize=(16, 8))
        if car_model.total_detections == 1:
            axes = axes.reshape(2, 1)

        # Track cumulative points
        cumulative_points = []
        points_so_far = np.array([]).reshape(0, 3)

        detection_indices = list(range(min(4, car_model.total_detections)))

        for i, det_idx in enumerate(detection_indices):
            # Add points from this detection (simulate progressive building)
            # For visualization, we'll split the accumulated points by detection
            if i == 0:
                points_subset = car_model.accumulated_points[:len(car_model.accumulated_points)//car_model.total_detections]
            else:
                start_idx = i * len(car_model.accumulated_points) // car_model.total_detections
                end_idx = (i + 1) * len(car_model.accumulated_points) // car_model.total_detections
                points_subset = car_model.accumulated_points[start_idx:end_idx]

            if len(points_subset) > 0:
                if len(points_so_far) == 0:
                    points_so_far = points_subset
                else:
                    points_so_far = np.vstack([points_so_far, points_subset])

            cumulative_points.append(points_so_far.copy())

            # Plot 3D point cloud progression (top view)
            if len(points_so_far) > 0:
                axes[0, i].scatter(points_so_far[:, 0], points_so_far[:, 1],
                                 c=range(len(points_so_far)), cmap='viridis', s=1)
                axes[0, i].set_title(f'Detection {i+1}\n{len(points_so_far)} points')
                axes[0, i].set_xlabel('X (m)')
                axes[0, i].set_ylabel('Y (m)')
                axes[0, i].axis('equal')

            # Show cropped car image
            if i < len(car_model.detection_images):
                cropped_img = car_model.detection_images[i]
                if cropped_img.size > 0:
                    axes[1, i].imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f'Frame {car_model.detection_frames[i]}')
                axes[1, i].axis('off')

        plt.suptitle(f'Car {car_model.track_id} Progressive Building\n'
                    f'Total: {car_model.total_detections} detections, '
                    f'{len(car_model.accumulated_points)} points, '
                    f'Quality: {car_model.model_quality_score:.2f}')
        plt.tight_layout()

        if save_output:
            prog_path = output_dir / f"car_{car_model.track_id}_progression.png"
            plt.savefig(prog_path, dpi=300, bbox_inches='tight')
            print(f"Progression visualization saved: {prog_path}")
        else:
            plt.show()

        plt.close()

    def _create_summary_visualization(self, car_models: Dict[int, AccumulatedCarModel],
                                    output_dir: Path, sequence_key: str, save_output: bool) -> None:
        """Create summary visualization of all tracked cars."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 3D scatter plot of all car models
        colors = plt.cm.tab10(np.linspace(0, 1, len(car_models)))
        for i, (track_id, car_model) in enumerate(car_models.items()):
            if len(car_model.accumulated_points) > 0:
                points = car_model.accumulated_points
                ax1.scatter(points[:, 0], points[:, 1],
                          c=[colors[i]], s=2, alpha=0.7, label=f'Car {track_id}')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('All Tracked Cars (Top View)')
        ax1.legend()
        ax1.axis('equal')

        # 2. Point accumulation over detections
        for i, (track_id, car_model) in enumerate(car_models.items()):
            detection_counts = list(range(1, car_model.total_detections + 1))
            # Simulate point accumulation
            point_counts = [j * len(car_model.accumulated_points) // car_model.total_detections
                          for j in detection_counts]
            ax2.plot(detection_counts, point_counts, 'o-', color=colors[i],
                    label=f'Car {track_id}')

        ax2.set_xlabel('Detection Number')
        ax2.set_ylabel('Accumulated Points')
        ax2.set_title('Point Accumulation Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Model quality scores
        track_ids = list(car_models.keys())
        quality_scores = [car_models[tid].model_quality_score for tid in track_ids]
        point_counts = [len(car_models[tid].accumulated_points) for tid in track_ids]

        bars = ax3.bar([f'Car {tid}' for tid in track_ids], quality_scores,
                      color=colors[:len(track_ids)])
        ax3.set_ylabel('Quality Score')
        ax3.set_title('Model Quality by Car')
        ax3.set_ylim(0, 1)

        # Add point count labels on bars
        for bar, count in zip(bars, point_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{count}pts', ha='center', va='bottom', fontsize=8)

        # 4. Detection statistics
        detection_counts = [car_model.total_detections for car_model in car_models.values()]
        ax4.hist(detection_counts, bins=max(1, len(set(detection_counts))),
                color='skyblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Number of Detections')
        ax4.set_ylabel('Number of Cars')
        ax4.set_title('Detection Count Distribution')

        plt.suptitle(f'Enhanced Car Tracking Summary - {sequence_key}\n'
                    f'Tracked {len(car_models)} cars with progressive model building')
        plt.tight_layout()

        if save_output:
            summary_path = output_dir / f"{sequence_key}_tracking_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            print(f"Tracking summary saved: {summary_path}")
        else:
            plt.show()

        plt.close()

    def save_tracking_results(self, car_models: Dict[int, AccumulatedCarModel],
                            sequence_key: str) -> None:
        """Save detailed tracking results."""
        output_dir = Path("sensor_fusion_output") / "enhanced_tracking" / sequence_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary statistics
        summary_data = {
            "sequence_key": sequence_key,
            "total_tracked_cars": len(car_models),
            "tracking_summary": {}
        }

        for track_id, car_model in car_models.items():
            car_summary = {
                "track_id": track_id,
                "total_detections": car_model.total_detections,
                "total_points": len(car_model.accumulated_points),
                "tracking_duration": car_model.last_update_time - car_model.first_detection_time,
                "model_quality_score": car_model.model_quality_score,
                "cameras_used": list(set(car_model.detection_cameras)),
                "estimated_dimensions": car_model.get_dimensions().tolist() if car_model.get_dimensions() is not None else None,
                "centroid_3d": car_model.centroid_3d.tolist() if car_model.centroid_3d is not None else None
            }
            summary_data["tracking_summary"][track_id] = car_summary

        # Save JSON summary
        json_path = output_dir / f"{sequence_key}_enhanced_tracking.json"
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"Enhanced tracking results saved: {json_path}")


def main():
    """Main function for enhanced car tracking."""
    parser = argparse.ArgumentParser(description="Enhanced Car Tracking with Progressive Building")
    parser.add_argument("--sequence", required=True, help="Sequence ID to analyze (e.g., seq-53)")
    parser.add_argument("--device", required=True, help="Device ID filter (e.g., 105)")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum frames to process")
    parser.add_argument("--correspondence_file", default="data/RCooper/corespondence_camera_lidar.json",
                       help="Path to correspondence file")
    parser.add_argument("--data_path", default="data/RCooper", help="Base data directory path")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization output")

    args = parser.parse_args()

    # Initialize enhanced tracker
    tracker = EnhancedCarTracker(
        correspondence_file=args.correspondence_file,
        data_path=args.data_path
    )

    # Process sequence with accumulation
    sequence_key = f"{args.device}_{args.sequence}"
    start_time = time.time()

    print(f"🚗 Starting Enhanced Car Tracking Demo")
    print(f"Sequence: {sequence_key}")
    print(f"Max frames: {args.max_frames}")
    print()

    car_models = tracker.process_sequence_with_accumulation(
        args.sequence, args.device, args.max_frames
    )

    processing_time = time.time() - start_time

    if car_models:
        print(f"\n🎯 Enhanced Tracking Results:")
        for track_id, car_model in car_models.items():
            dimensions = car_model.get_dimensions()
            print(f"  Car {track_id}:")
            print(f"    Detections: {car_model.total_detections}")
            print(f"    Points: {len(car_model.accumulated_points):,}")
            print(f"    Quality: {car_model.model_quality_score:.2f}")
            if dimensions is not None:
                print(f"    Dimensions: {dimensions[0]:.1f}×{dimensions[1]:.1f}×{dimensions[2]:.1f}m")

        print(f"\n⏱️  Processing time: {processing_time:.1f}s")

        # Save results
        tracker.save_tracking_results(car_models, sequence_key)

        # Create visualizations
        if not args.no_viz:
            tracker.visualize_progressive_building(car_models, sequence_key, save_output=True)

        print(f"\n✅ Enhanced car tracking completed!")
        print(f"📁 Results saved in: sensor_fusion_output/enhanced_tracking/{sequence_key}/")

    else:
        print(f"❌ No car tracks found in {sequence_key}")


if __name__ == "__main__":
    main()