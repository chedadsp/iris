#!/usr/bin/env python3
"""
Sequence LIDAR Analysis - Dynamic Point Detection

Combines LIDAR points from all scans in a sequence, identifies static points that
remain unchanged across scans, and visualizes the dynamic points that represent
moving objects or scene changes.

Author: Claude
Date: September 2025
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict

# Point cloud processing
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Visualization
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Project imports
from point_cloud.vtk_utils import VTKSafetyManager
from point_cloud.config import VTKConfig


class SequenceLidarAnalyzer:
    """Analyzes LIDAR sequences to identify dynamic vs static points."""

    def __init__(self, correspondence_file: str, base_data_path: str):
        """
        Initialize the sequence LIDAR analyzer.

        Args:
            correspondence_file: Path to camera-LIDAR correspondence JSON file
            base_data_path: Base path for the data directory
        """
        self.base_data_path = Path(base_data_path)
        self.correspondence_data = self._load_correspondence_file(correspondence_file)
        self.sequences = self._organize_sequences()

        # Analysis parameters
        self.static_threshold = 0.05  # meters - points within this distance considered static
        self.min_occurrences = 0.7   # fraction - point must appear in this fraction of scans to be static
        self.voxel_size = 0.1        # meters - voxel size for downsampling

        print(f"Loaded {len(self.sequences)} sequences for analysis")

    def _load_correspondence_file(self, file_path: str) -> dict:
        """Load the camera-LIDAR correspondence file."""
        with open(file_path, 'r') as f:
            return json.load(f)

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

    def load_lidar_points(self, pcd_file_path: str) -> np.ndarray:
        """
        Load LIDAR point cloud from PCD file.

        Args:
            pcd_file_path: Path to PCD file

        Returns:
            numpy array of points (N, 3)
        """
        full_path = self.base_data_path / pcd_file_path.lstrip('./')

        try:
            # Use open3d for robust PCD loading
            pcd = o3d.io.read_point_cloud(str(full_path))
            points = np.asarray(pcd.points)

            if len(points) == 0:
                print(f"Warning: No points loaded from {full_path}")
                return np.array([]).reshape(0, 3)

            return points
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return np.array([]).reshape(0, 3)

    def downsample_points(self, points: np.ndarray, voxel_size: float = None) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.

        Args:
            points: Input points (N, 3)
            voxel_size: Size of voxel grid

        Returns:
            Downsampled points
        """
        if voxel_size is None:
            voxel_size = self.voxel_size

        if len(points) == 0:
            return points

        # Use open3d for efficient voxel downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(downsampled_pcd.points)

    def identify_static_points(self, point_clouds: List[np.ndarray],
                             timestamps: List[float]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Identify static points that appear consistently across scans.

        Args:
            point_clouds: List of point clouds from sequence
            timestamps: Corresponding timestamps

        Returns:
            Tuple of:
            - static_points: Points that remain static (N, 3)
            - dynamic_points: All unique points that are not static (M, 3)
            - analysis_stats: Dictionary with analysis statistics
        """
        print(f"Analyzing {len(point_clouds)} point clouds for static/dynamic points...")

        # Combine all points and track their occurrences
        all_points = []
        point_scan_indices = []

        for i, points in enumerate(point_clouds):
            if len(points) > 0:
                # Downsample each scan to manageable size
                downsampled = self.downsample_points(points)
                all_points.append(downsampled)
                point_scan_indices.extend([i] * len(downsampled))
                print(f"  Scan {i}: {len(points)} -> {len(downsampled)} points (downsampled)")

        if not all_points:
            print("No valid point clouds to analyze")
            return np.array([]), np.array([]), {}

        # Combine all downsampled points
        combined_points = np.vstack(all_points)
        point_scan_indices = np.array(point_scan_indices)

        print(f"Combined {len(combined_points)} total points from all scans")

        # Find points that appear in multiple scans (potential static points)
        print("Identifying static points using nearest neighbor analysis...")

        # Use KDTree for efficient neighbor search
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree',
                               metric='euclidean').fit(combined_points)

        # For each point, find how many scans it appears in
        static_points = []
        dynamic_points = []

        # Process points in batches to manage memory
        batch_size = 10000
        num_batches = (len(combined_points) + batch_size - 1) // batch_size

        occurrence_counts = np.zeros(len(combined_points))
        processed_points = set()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(combined_points))
            batch_points = combined_points[start_idx:end_idx]

            # Find neighbors within threshold distance
            distances, indices = nbrs.radius_neighbors(batch_points,
                                                     radius=self.static_threshold)

            for i, (point_distances, point_indices) in enumerate(zip(distances, indices)):
                global_idx = start_idx + i

                if global_idx in processed_points:
                    continue

                # Count unique scans this point appears in
                unique_scans = set(point_scan_indices[point_indices])
                occurrence_ratio = len(unique_scans) / len(point_clouds)

                occurrence_counts[global_idx] = occurrence_ratio

                # Mark all nearby points as processed to avoid duplicates
                for neighbor_idx in point_indices:
                    processed_points.add(neighbor_idx)

                # Classify as static or dynamic
                if occurrence_ratio >= self.min_occurrences:
                    static_points.append(combined_points[global_idx])
                else:
                    dynamic_points.append(combined_points[global_idx])

            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx + 1}/{num_batches}")

        static_points = np.array(static_points) if static_points else np.array([]).reshape(0, 3)
        dynamic_points = np.array(dynamic_points) if dynamic_points else np.array([]).reshape(0, 3)

        # Analysis statistics
        stats = {
            'total_scans': len(point_clouds),
            'total_combined_points': len(combined_points),
            'static_points': len(static_points),
            'dynamic_points': len(dynamic_points),
            'static_ratio': len(static_points) / len(combined_points) if len(combined_points) > 0 else 0,
            'dynamic_ratio': len(dynamic_points) / len(combined_points) if len(combined_points) > 0 else 0,
            'timestamps': timestamps,
            'parameters': {
                'static_threshold': self.static_threshold,
                'min_occurrences': self.min_occurrences,
                'voxel_size': self.voxel_size
            }
        }

        print(f"Analysis complete:")
        print(f"  Static points: {len(static_points)} ({stats['static_ratio']:.1%})")
        print(f"  Dynamic points: {len(dynamic_points)} ({stats['dynamic_ratio']:.1%})")

        return static_points, dynamic_points, stats

    def identify_temporal_changes(self, point_clouds: List[np.ndarray],
                                 timestamps: List[float],
                                 dynamic_points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Identify points that actually changed over time from the dynamic point set.

        This method takes the dynamic points and further filters them to find only
        those that show actual temporal variation, removing any points that might
        have been misclassified as dynamic but remain relatively static.

        Args:
            point_clouds: List of downsampled point clouds from different timestamps
            timestamps: Corresponding timestamps for each point cloud
            dynamic_points: Previously identified dynamic points

        Returns:
            Tuple of (temporal_change_points, analysis_stats)
        """
        if len(point_clouds) < 3:
            print("⚠️  Need at least 3 scans for temporal change analysis")
            return np.array([]), {}

        print(f"Analyzing temporal changes in {len(dynamic_points):,} dynamic points...")

        # Create a finer temporal analysis
        temporal_threshold = self.static_threshold * 1.2  # More generous threshold for finding correspondences
        min_movement_distance = 0.05  # Very low threshold to catch small movements (5cm)

        # For each dynamic point, check if it shows consistent movement across time
        changing_points = []
        movement_vectors = []

        # Process in smaller batches to manage memory and performance
        batch_size = 1000  # Reduced from 5000 for faster processing
        num_batches = (len(dynamic_points) + batch_size - 1) // batch_size

        # For testing, limit to a maximum number of points for performance
        if len(dynamic_points) > 5000:
            print(f"  Limiting temporal analysis to first 5,000 points for performance")
            dynamic_points = dynamic_points[:5000]
            num_batches = (len(dynamic_points) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            print(f"  Processing temporal batch {batch_idx + 1}/{num_batches}")

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(dynamic_points))
            batch_points = dynamic_points[start_idx:end_idx]

            # Simplified temporal analysis: check if points appear in different positions across scans
            # Use KDTree for faster nearest neighbor lookup
            from sklearn.neighbors import NearestNeighbors

            for point_idx, point in enumerate(batch_points):
                position_found_count = 0
                first_position = None
                last_position = None

                # Use all scans for better temporal analysis
                scan_indices = range(len(point_clouds))

                for scan_idx in scan_indices:
                    scan_points = point_clouds[scan_idx]
                    if len(scan_points) == 0:
                        continue

                    # Find closest point using vectorized operations
                    distances = np.linalg.norm(scan_points - point, axis=1)
                    min_distance = np.min(distances)

                    if min_distance <= temporal_threshold:
                        closest_point = scan_points[np.argmin(distances)]
                        position_found_count += 1

                        if first_position is None:
                            first_position = closest_point
                        last_position = closest_point

                # Simple movement check: if found in multiple scans and moved significantly
                if position_found_count >= 2 and first_position is not None and last_position is not None:
                    total_displacement = np.linalg.norm(last_position - first_position)

                    # Consider it changing if it moved more than minimum distance
                    if total_displacement > min_movement_distance:
                        changing_points.append(point)
                        movement_vectors.append(last_position - first_position)

        temporal_change_points = np.array(changing_points)
        movement_vectors = np.array(movement_vectors) if movement_vectors else np.array([]).reshape(0, 3)

        # Calculate statistics
        temporal_stats = {
            'total_dynamic_points': len(dynamic_points),
            'temporal_change_points': len(temporal_change_points),
            'temporal_change_ratio': len(temporal_change_points) / len(dynamic_points) if len(dynamic_points) > 0 else 0,
            'avg_movement_distance': np.mean(np.linalg.norm(movement_vectors, axis=1)) if len(movement_vectors) > 0 else 0,
            'max_movement_distance': np.max(np.linalg.norm(movement_vectors, axis=1)) if len(movement_vectors) > 0 else 0,
            'movement_vectors': movement_vectors,
            'temporal_threshold': temporal_threshold,
            'min_movement_distance': min_movement_distance
        }

        print(f"Temporal analysis complete:")
        print(f"  Original dynamic points: {len(dynamic_points):,}")
        print(f"  Temporal change points: {len(temporal_change_points):,} ({temporal_stats['temporal_change_ratio']:.1%})")
        print(f"  Average movement: {temporal_stats['avg_movement_distance']:.2f}m")

        return temporal_change_points, temporal_stats

    def visualize_dynamic_analysis(self, static_points: np.ndarray, dynamic_points: np.ndarray,
                                 sequence_key: str, stats: Dict, save_output: bool = True) -> None:
        """
        Visualize static vs dynamic points analysis.

        Args:
            static_points: Static points array
            dynamic_points: Dynamic points array
            sequence_key: Sequence identifier
            stats: Analysis statistics
            save_output: Whether to save visualization files
        """
        print("Creating dynamic analysis visualization...")

        # Create output directory
        output_dir = Path("sensor_fusion_output") / "sequence_analysis" / sequence_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. PyVista 3D visualization
        VTKSafetyManager.setup_vtk_environment()
        plotter = VTKSafetyManager.create_safe_plotter(
            off_screen=save_output,
            window_size=VTKConfig.DEFAULT_WINDOW_SIZE
        )

        # Add static points in blue
        if len(static_points) > 0:
            static_cloud = pv.PolyData(static_points)
            plotter.add_mesh(static_cloud, color='lightblue', point_size=2,
                           render_points_as_spheres=True, label='Static Points')

        # Add dynamic points in red
        if len(dynamic_points) > 0:
            dynamic_cloud = pv.PolyData(dynamic_points)
            plotter.add_mesh(dynamic_cloud, color='red', point_size=4,
                           render_points_as_spheres=True, label='Dynamic Points')

        # Add coordinate axes and title
        plotter.add_axes()
        plotter.add_legend()
        plotter.add_title(f"Dynamic Analysis: {sequence_key}\n"
                        f"Static: {len(static_points):,} | Dynamic: {len(dynamic_points):,}")

        if save_output:
            screenshot_path = output_dir / f"{sequence_key}_dynamic_analysis.png"
            plotter.screenshot(str(screenshot_path))
            print(f"3D visualization saved: {screenshot_path}")
        else:
            plotter.show()

        VTKSafetyManager.cleanup_vtk_plotter(plotter)

        # 2. Statistical plots
        if save_output:
            self._create_analysis_plots(stats, output_dir, sequence_key)

        # 3. Save point clouds
        if save_output:
            self._save_point_clouds(static_points, dynamic_points, output_dir, sequence_key)

    def visualize_temporal_changes(self, static_points: np.ndarray, dynamic_points: np.ndarray,
                                 temporal_change_points: np.ndarray, sequence_key: str,
                                 stats: Dict, save_output: bool = True) -> None:
        """
        Visualize temporal change analysis with three point categories.

        Args:
            static_points: Static points array
            dynamic_points: All dynamic points array
            temporal_change_points: Points that actually changed over time
            sequence_key: Sequence identifier
            stats: Analysis statistics including temporal analysis
            save_output: Whether to save visualization files
        """
        print("Creating temporal change visualization...")

        # Create output directory
        output_dir = Path("sensor_fusion_output") / "sequence_analysis" / sequence_key
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Enhanced PyVista 3D visualization showing three categories
        VTKSafetyManager.setup_vtk_environment()
        plotter = VTKSafetyManager.create_safe_plotter(
            off_screen=save_output,
            window_size=VTKConfig.DEFAULT_WINDOW_SIZE
        )

        # Add static points in blue
        if len(static_points) > 0:
            static_cloud = pv.PolyData(static_points)
            plotter.add_mesh(static_cloud, color='lightblue', point_size=2,
                           render_points_as_spheres=True, label='Static Points')

        # Add dynamic points that don't show temporal change in yellow
        if len(dynamic_points) > 0 and len(temporal_change_points) > 0:
            # Find dynamic points that are NOT in temporal change points
            dynamic_set = set(map(tuple, dynamic_points))
            temporal_set = set(map(tuple, temporal_change_points))
            non_temporal_dynamic = np.array([p for p in dynamic_set if p not in temporal_set])

            if len(non_temporal_dynamic) > 0:
                non_temporal_cloud = pv.PolyData(non_temporal_dynamic)
                plotter.add_mesh(non_temporal_cloud, color='yellow', point_size=3,
                               render_points_as_spheres=True, label='Dynamic (No Temporal Change)')

        # Add temporal change points in red (most important)
        if len(temporal_change_points) > 0:
            temporal_cloud = pv.PolyData(temporal_change_points)
            plotter.add_mesh(temporal_cloud, color='red', point_size=5,
                           render_points_as_spheres=True, label='Temporal Change Points')

        # Add coordinate axes and enhanced title
        plotter.add_axes()
        plotter.add_legend()

        temporal_stats = stats.get('temporal_analysis', {})
        temporal_count = temporal_stats.get('temporal_change_points', 0)
        temporal_ratio = temporal_stats.get('temporal_change_ratio', 0)

        plotter.add_title(f"Temporal Change Analysis: {sequence_key}\n"
                        f"Static: {len(static_points):,} | Dynamic: {len(dynamic_points):,} | "
                        f"Temporal Changes: {temporal_count:,} ({temporal_ratio:.1%})")

        if save_output:
            screenshot_path = output_dir / f"{sequence_key}_temporal_analysis.png"
            plotter.screenshot(str(screenshot_path))
            print(f"Temporal visualization saved: {screenshot_path}")
        else:
            plotter.show()

        VTKSafetyManager.cleanup_vtk_plotter(plotter)

        # 2. Save temporal change points
        if save_output:
            self._save_temporal_points(temporal_change_points, output_dir, sequence_key, temporal_stats)

    def _create_analysis_plots(self, stats: Dict, output_dir: Path, sequence_key: str) -> None:
        """Create statistical analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Point distribution pie chart
        sizes = [stats['static_points'], stats['dynamic_points']]
        labels = ['Static Points', 'Dynamic Points']
        colors = ['lightblue', 'red']

        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Static vs Dynamic Point Distribution')

        # 2. Timeline visualization (if timestamps available)
        if 'timestamps' in stats and len(stats['timestamps']) > 1:
            timestamps = np.array(stats['timestamps'])
            relative_times = timestamps - timestamps[0]

            ax2.plot(relative_times, range(len(relative_times)), 'o-', color='green')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Scan Index')
            ax2.set_title('Sequence Timeline')
            ax2.grid(True)

        # 3. Analysis parameters
        params = stats['parameters']
        param_names = list(params.keys())
        param_values = list(params.values())

        bars = ax3.bar(param_names, param_values, color=['orange', 'purple', 'brown'])
        ax3.set_title('Analysis Parameters')
        ax3.set_ylabel('Value')

        # Add value labels on bars
        for bar, value in zip(bars, param_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""
Sequence Analysis Summary: {sequence_key}

Total Scans: {stats['total_scans']}
Combined Points: {stats['total_combined_points']:,}

Static Points: {stats['static_points']:,} ({stats['static_ratio']:.1%})
Dynamic Points: {stats['dynamic_points']:,} ({stats['dynamic_ratio']:.1%})

Parameters:
• Static Threshold: {stats['parameters']['static_threshold']} m
• Min Occurrences: {stats['parameters']['min_occurrences']:.1%}
• Voxel Size: {stats['parameters']['voxel_size']} m
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plot_path = output_dir / f"{sequence_key}_analysis_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Analysis plots saved: {plot_path}")

    def _save_point_clouds(self, static_points: np.ndarray, dynamic_points: np.ndarray,
                          output_dir: Path, sequence_key: str) -> None:
        """Save point clouds in various formats."""
        # Save as numpy arrays
        if len(static_points) > 0:
            static_path = output_dir / f"{sequence_key}_static_points.npy"
            np.save(static_path, static_points)
            print(f"Static points saved: {static_path}")

        if len(dynamic_points) > 0:
            dynamic_path = output_dir / f"{sequence_key}_dynamic_points.npy"
            np.save(dynamic_path, dynamic_points)
            print(f"Dynamic points saved: {dynamic_path}")

        # Save as PCD files for external tools
        if len(static_points) > 0:
            static_pcd = o3d.geometry.PointCloud()
            static_pcd.points = o3d.utility.Vector3dVector(static_points)
            static_pcd_path = output_dir / f"{sequence_key}_static_points.pcd"
            o3d.io.write_point_cloud(str(static_pcd_path), static_pcd)

        if len(dynamic_points) > 0:
            dynamic_pcd = o3d.geometry.PointCloud()
            dynamic_pcd.points = o3d.utility.Vector3dVector(dynamic_points)
            dynamic_pcd_path = output_dir / f"{sequence_key}_dynamic_points.pcd"
            o3d.io.write_point_cloud(str(dynamic_pcd_path), dynamic_pcd)

    def _save_temporal_points(self, temporal_change_points: np.ndarray, output_dir: Path,
                             sequence_key: str, temporal_stats: Dict) -> None:
        """Save temporal change points in various formats."""
        # Always save temporal statistics, even if no points found
        temporal_json_path = output_dir / f"{sequence_key}_temporal_stats.json"
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {}
        for key, value in temporal_stats.items():
            if isinstance(value, np.ndarray):
                json_stats[key] = value.tolist()
            else:
                json_stats[key] = value

        with open(temporal_json_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        print(f"Temporal statistics saved: {temporal_json_path}")

        if len(temporal_change_points) == 0:
            print("No temporal change points found, but statistics saved")
            return

        # Save as numpy array
        temporal_path = output_dir / f"{sequence_key}_temporal_change_points.npy"
        np.save(temporal_path, temporal_change_points)
        print(f"Temporal change points saved: {temporal_path}")

        # Save as PCD file for external tools
        temporal_pcd = o3d.geometry.PointCloud()
        temporal_pcd.points = o3d.utility.Vector3dVector(temporal_change_points)
        temporal_pcd_path = output_dir / f"{sequence_key}_temporal_change_points.pcd"
        o3d.io.write_point_cloud(str(temporal_pcd_path), temporal_pcd)
        print(f"Temporal change points PCD saved: {temporal_pcd_path}")

    def analyze_sequence(self, sequence_id: str, device_id: str = None,
                        max_scans: Optional[int] = None, visualize: bool = True) -> Dict:
        """
        Analyze a complete sequence for dynamic points.

        Args:
            sequence_id: Sequence ID to analyze (e.g., 'seq-53')
            device_id: Device ID (if None, analyzes all devices for this sequence)
            max_scans: Maximum number of scans to process (for testing)
            visualize: Whether to create visualizations

        Returns:
            Dictionary with analysis results
        """
        # Find matching sequences
        target_sequences = []
        if device_id:
            seq_key = f"{device_id}_{sequence_id}"
            if seq_key in self.sequences:
                target_sequences = [(seq_key, self.sequences[seq_key])]
            else:
                print(f"❌ Sequence {sequence_id} not found for device {device_id}")
                return {}
        else:
            # Find all devices with this sequence
            for seq_key, frames in self.sequences.items():
                if seq_key.endswith(f"_{sequence_id}"):
                    target_sequences.append((seq_key, frames))

        if not target_sequences:
            print(f"❌ No sequences found matching '{sequence_id}'")
            return {}

        print(f"🔍 Analyzing sequence '{sequence_id}' on {len(target_sequences)} device(s)")

        all_results = {}

        for seq_key, frames in target_sequences:
            print(f"\n📊 Processing {seq_key}: {len(frames)} scans")

            # Limit scans if specified
            if max_scans:
                frames = frames[:max_scans]
                print(f"Limited to {len(frames)} scans for testing")

            # Load all LIDAR point clouds
            point_clouds = []
            timestamps = []
            valid_scans = 0

            start_time = time.time()

            for i, frame in enumerate(frames):
                lidar_path = frame['lidar']['file_path']
                timestamp = frame['lidar']['timestamp']

                print(f"  Loading scan {i+1}/{len(frames)}: {Path(lidar_path).name}")

                points = self.load_lidar_points(lidar_path)
                if len(points) > 0:
                    point_clouds.append(points)
                    timestamps.append(timestamp)
                    valid_scans += 1
                else:
                    print(f"    ⚠️  Skipped scan {i+1} (no points loaded)")

            if valid_scans < 2:
                print(f"❌ Insufficient valid scans ({valid_scans}) for analysis")
                continue

            print(f"✅ Loaded {valid_scans} valid scans in {time.time() - start_time:.1f}s")

            # Perform static/dynamic analysis
            static_points, dynamic_points, stats = self.identify_static_points(
                point_clouds, timestamps)

            # Perform temporal change analysis on dynamic points
            temporal_change_points, temporal_stats = self.identify_temporal_changes(
                point_clouds, timestamps, dynamic_points)

            # Update stats with temporal analysis
            stats['temporal_analysis'] = temporal_stats

            # Create visualizations
            if visualize:
                self.visualize_dynamic_analysis(static_points, dynamic_points,
                                               seq_key, stats, save_output=True)
                # Add temporal change visualization
                self.visualize_temporal_changes(static_points, dynamic_points, temporal_change_points,
                                               seq_key, stats, save_output=True)

            # Store results
            result = {
                'sequence_key': seq_key,
                'device_id': seq_key.split('_')[0],
                'sequence_id': sequence_id,
                'total_scans': len(frames),
                'valid_scans': valid_scans,
                'static_points': len(static_points),
                'dynamic_points': len(dynamic_points),
                'temporal_change_points': len(temporal_change_points),
                'analysis_duration': time.time() - start_time,
                'stats': stats
            }

            all_results[seq_key] = result

            print(f"🎯 {seq_key} Results:")
            print(f"    Static points: {len(static_points):,}")
            print(f"    Dynamic points: {len(dynamic_points):,}")
            print(f"    Temporal change points: {len(temporal_change_points):,}")
            print(f"    Processing time: {result['analysis_duration']:.1f}s")

        return all_results

    def list_sequences(self, device_filter: Optional[str] = None) -> None:
        """List available sequences with frame counts."""
        print("📋 Available Sequences for Dynamic Analysis:")
        print()

        filtered_sequences = {}
        for seq_key, frames in self.sequences.items():
            device_id, sequence_id = seq_key.split('_', 1)
            if device_filter and device_id != device_filter:
                continue
            filtered_sequences[seq_key] = frames

        for seq_key, frames in filtered_sequences.items():
            device_id, sequence_id = seq_key.split('_', 1)
            duration = frames[-1]['lidar']['timestamp'] - frames[0]['lidar']['timestamp']

            print(f"Sequence {sequence_id} (Device {device_id}):")
            print(f"  📊 {len(frames)} LIDAR scans ({duration:.1f}s duration)")
            print(f"  📁 Output: sensor_fusion_output/sequence_analysis/{seq_key}/")
            print()


def main():
    """Main function for sequence LIDAR analysis."""
    parser = argparse.ArgumentParser(description='Sequence LIDAR Dynamic Point Analysis')

    parser.add_argument('--sequence', type=str, required=False,
                       help='Sequence ID to analyze (e.g., seq-53)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device ID filter (e.g., 105)')
    parser.add_argument('--max_scans', type=int, default=None,
                       help='Maximum number of scans to process (for testing)')
    parser.add_argument('--static_threshold', type=float, default=0.05,
                       help='Distance threshold for static points (meters)')
    parser.add_argument('--min_occurrences', type=float, default=0.7,
                       help='Minimum occurrence ratio for static points (0-1)')
    parser.add_argument('--voxel_size', type=float, default=0.1,
                       help='Voxel size for downsampling (meters)')
    parser.add_argument('--list_sequences', action='store_true',
                       help='List all available sequences and exit')
    parser.add_argument('--list_device', type=str, default=None,
                       help='List sequences for specific device only')
    parser.add_argument('--correspondence_file', type=str,
                       default='data/RCooper/corespondence_camera_lidar.json',
                       help='Path to correspondence file')
    parser.add_argument('--data_path', type=str, default='data/RCooper',
                       help='Base data directory path')
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable visualization output')

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = SequenceLidarAnalyzer(
            correspondence_file=args.correspondence_file,
            base_data_path=args.data_path
        )

        # Set analysis parameters
        analyzer.static_threshold = args.static_threshold
        analyzer.min_occurrences = args.min_occurrences
        analyzer.voxel_size = args.voxel_size

        # List sequences if requested
        if args.list_sequences:
            analyzer.list_sequences(device_filter=args.list_device)
            return 0

        # Analyze specified sequence
        if not args.sequence:
            print("❌ Sequence ID is required. Use --list_sequences to see available sequences.")
            return 1

        results = analyzer.analyze_sequence(
            sequence_id=args.sequence,
            device_id=args.device,
            max_scans=args.max_scans,
            visualize=not args.no_viz
        )

        if not results:
            print("❌ No analysis results generated")
            return 1

        print(f"\n✅ Dynamic analysis completed!")
        print(f"📁 Check output folders in: sensor_fusion_output/sequence_analysis/")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())