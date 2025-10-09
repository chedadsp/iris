"""
Passenger Detection Animation System for Sales MVP
Progressive LIDAR animation showing passenger detection inside vehicles
Similar to existing progressive animation but focused on interior detection
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import open3d as o3d
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import seaborn as sns

# Import existing systems
from sensor_fusion import SensorFusion
from enhanced_car_tracking import EnhancedCarTracker

@dataclass
class PassengerDetection:
    """Represents a detected passenger within a car"""
    position: np.ndarray  # 3D position
    confidence: float
    size: float  # Estimated passenger size
    frame_detected: int
    points: np.ndarray  # LIDAR points belonging to this passenger

@dataclass
class PassengerAnimationConfig:
    """Configuration for passenger detection animation"""
    figure_size: Tuple[int, int] = (20, 12)
    figure_dpi: int = 300
    color_scheme: str = 'viridis'
    animation_duration: int = 8  # frames
    point_size: float = 0.5
    alpha_base: float = 0.6
    alpha_new: float = 1.0
    
    # Color palette for different frames
    frame_colors: List[str] = None
    
    def __post_init__(self):
        if self.frame_colors is None:
            # Professional color palette for progressive frames
            self.frame_colors = [
                '#FF6B6B',  # Coral red - initial detection
                '#4ECDC4',  # Teal - accumulation
                '#45B7D1',  # Blue - building model
                '#96CEB4',  # Mint green - refinement
                '#FFEAA7',  # Light yellow - detail
                '#DDA0DD',  # Plum - enhanced
                '#98D8C8',  # Mint - finalization
                '#F7DC6F',  # Gold - complete model
            ]

class PassengerDetectionAnimator:
    """Creates progressive animations of LIDAR detecting passengers in cars"""
    
    def __init__(self, config: PassengerAnimationConfig = None):
        self.config = config or PassengerAnimationConfig()
        self.sensor_fusion = SensorFusion()
        self.car_tracker = EnhancedCarTracker()
        
        # Passenger simulation parameters
        self.passenger_templates = self._create_passenger_templates()
        
    def _create_passenger_templates(self) -> Dict[str, np.ndarray]:
        """Create realistic passenger point cloud templates"""
        templates = {}
        
        # Driver template - typical sitting position
        driver_points = []
        # Head area
        for i in range(50):
            x = np.random.normal(0, 0.15)  # 30cm head width
            y = np.random.normal(0, 0.15) 
            z = np.random.normal(1.2, 0.1)  # 1.2m head height when sitting
            driver_points.append([x, y, z])
        
        # Torso area
        for i in range(80):
            x = np.random.normal(0, 0.25)  # 50cm shoulder width
            y = np.random.normal(0.1, 0.15)  # Slightly forward
            z = np.random.normal(0.8, 0.2)  # Torso height
            driver_points.append([x, y, z])
        
        # Arms/hands on steering wheel area
        for i in range(30):
            x = np.random.normal(0, 0.3)
            y = np.random.normal(0.3, 0.1)  # Forward for steering
            z = np.random.normal(1.0, 0.1)  # Hand height
            driver_points.append([x, y, z])
        
        templates['driver'] = np.array(driver_points)
        
        # Passenger template - relaxed sitting position
        passenger_points = []
        # Head area
        for i in range(50):
            x = np.random.normal(0.6, 0.15)  # Offset for passenger seat
            y = np.random.normal(-0.1, 0.15)  # Slightly back
            z = np.random.normal(1.2, 0.1)
            passenger_points.append([x, y, z])
        
        # Torso area
        for i in range(80):
            x = np.random.normal(0.6, 0.25)
            y = np.random.normal(0, 0.15)
            z = np.random.normal(0.8, 0.2)
            passenger_points.append([x, y, z])
        
        templates['passenger'] = np.array(passenger_points)
        
        return templates
    
    def simulate_passenger_detection(self, car_bbox: List[float], car_points: np.ndarray, 
                                   frame_idx: int) -> List[PassengerDetection]:
        """Simulate progressive passenger detection within a car"""
        passengers = []
        
        if len(car_points) < 50:  # Not enough points for passenger detection
            return passengers
        
        # Simulate passenger detection based on point density and patterns
        car_center = np.mean(car_points, axis=0)
        
        # Driver detection (higher probability)
        if len(car_points) > 100 and frame_idx >= 2:  # Need accumulation
            driver_template = self.passenger_templates['driver']
            driver_points = driver_template + car_center + np.array([-1.5, 0, 0])  # Offset for driver position
            
            # Add noise and realistic scatter
            noise = np.random.normal(0, 0.05, driver_points.shape)
            driver_points += noise
            
            # Filter points that would be inside the car
            interior_mask = self._is_interior_point(driver_points, car_center)
            driver_points = driver_points[interior_mask]
            
            if len(driver_points) > 20:
                passengers.append(PassengerDetection(
                    position=np.mean(driver_points, axis=0),
                    confidence=min(0.7 + frame_idx * 0.05, 0.95),
                    size=1.7,  # Average person height
                    frame_detected=frame_idx,
                    points=driver_points
                ))
        
        # Passenger detection (lower probability, needs more frames)
        if len(car_points) > 150 and frame_idx >= 4 and np.random.random() > 0.3:
            passenger_template = self.passenger_templates['passenger']
            passenger_points = passenger_template + car_center + np.array([-1.5, 0.6, 0])  # Passenger seat
            
            noise = np.random.normal(0, 0.05, passenger_points.shape)
            passenger_points += noise
            
            interior_mask = self._is_interior_point(passenger_points, car_center)
            passenger_points = passenger_points[interior_mask]
            
            if len(passenger_points) > 15:
                passengers.append(PassengerDetection(
                    position=np.mean(passenger_points, axis=0),
                    confidence=min(0.5 + frame_idx * 0.03, 0.85),
                    size=1.7,
                    frame_detected=frame_idx,
                    points=passenger_points
                ))
        
        return passengers
    
    def _is_interior_point(self, points: np.ndarray, car_center: np.ndarray) -> np.ndarray:
        """Check if points are within reasonable car interior bounds"""
        relative_points = points - car_center
        
        # Typical car interior bounds (rough approximation)
        x_bounds = (-2.5, 0.5)  # Car length
        y_bounds = (-1.0, 1.0)  # Car width
        z_bounds = (0.5, 2.0)   # Interior height
        
        mask = (
            (relative_points[:, 0] >= x_bounds[0]) & (relative_points[:, 0] <= x_bounds[1]) &
            (relative_points[:, 1] >= y_bounds[0]) & (relative_points[:, 1] <= y_bounds[1]) &
            (relative_points[:, 2] >= z_bounds[0]) & (relative_points[:, 2] <= z_bounds[1])
        )
        
        return mask
    
    def create_passenger_detection_animation(self, sequence_id: str, device_id: str = "105", 
                                           output_dir: str = None, max_frames: int = 8) -> str:
        """Create complete passenger detection animation for sales demo"""
        
        if output_dir is None:
            output_dir = f"sensor_fusion_output/passenger_detection/{sequence_id}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎬 Creating Passenger Detection Animation for {sequence_id}")
        print(f"📁 Output: {output_dir}")
        
        # Process frames and accumulate data
        accumulated_cars = {}
        accumulated_passengers = {}
        frame_data = []
        
        for frame_idx in range(max_frames):
            print(f"  📊 Processing frame {frame_idx + 1}/{max_frames}")
            
            # Get sensor fusion results
            fusion_results = self._get_fusion_data(sequence_id, device_id, frame_idx)
            
            frame_cars = []
            frame_passengers = []
            
            for car_id, car_data in fusion_results.items():
                # Accumulate car points
                if car_id not in accumulated_cars:
                    accumulated_cars[car_id] = {
                        'points': [],
                        'detections': [],
                        'bbox_history': []
                    }
                
                accumulated_cars[car_id]['points'].extend(car_data['points'])
                accumulated_cars[car_id]['detections'].append(car_data)
                accumulated_cars[car_id]['bbox_history'].append(car_data['bbox'])
                
                frame_cars.append(car_id)
                
                # Simulate passenger detection
                all_car_points = np.array(accumulated_cars[car_id]['points'])
                passengers = self.simulate_passenger_detection(
                    car_data['bbox'], all_car_points, frame_idx
                )
                
                for passenger in passengers:
                    passenger_key = f"{car_id}_passenger_{len(accumulated_passengers.get(car_id, []))}"
                    
                    if car_id not in accumulated_passengers:
                        accumulated_passengers[car_id] = []
                    
                    # Check if this is a new passenger or update existing
                    is_new = True
                    for existing in accumulated_passengers[car_id]:
                        if np.linalg.norm(existing.position - passenger.position) < 0.5:
                            # Update existing passenger
                            existing.confidence = max(existing.confidence, passenger.confidence)
                            existing.points = np.vstack([existing.points, passenger.points])
                            is_new = False
                            break
                    
                    if is_new:
                        accumulated_passengers[car_id].append(passenger)
                        frame_passengers.append(passenger_key)
            
            frame_data.append({
                'frame_idx': frame_idx,
                'cars': frame_cars,
                'passengers': frame_passengers,
                'car_data': dict(accumulated_cars),
                'passenger_data': dict(accumulated_passengers)
            })
        
        # Create animation frames
        print("🎨 Generating animation frames...")
        animation_frames = []
        
        for frame_idx, data in enumerate(frame_data):
            frame_path = os.path.join(output_dir, f"passenger_detection_frame_{frame_idx+1:03d}.png")
            
            self._create_frame_visualization(
                data, frame_idx, frame_path, max_frames
            )
            
            animation_frames.append(frame_path)
            print(f"  ✅ Frame {frame_idx + 1} saved")
        
        # Create summary visualization
        summary_path = os.path.join(output_dir, "passenger_detection_summary.png")
        self._create_summary_visualization(frame_data, summary_path)
        
        # Save passenger data as PCD files
        self._save_passenger_models(accumulated_passengers, output_dir)
        
        print(f"✅ Passenger detection animation complete!")
        print(f"📁 Animation frames: {len(animation_frames)}")
        print(f"📊 Summary visualization: {summary_path}")
        
        return output_dir
    
    def _get_fusion_data(self, sequence_id: str, device_id: str, frame_idx: int) -> Dict:
        """Get or simulate sensor fusion data for the frame"""
        
        # For MVP, we'll simulate realistic car detection data
        # In production, this would use actual sensor fusion results
        
        cars = {}
        
        # Simulate 1-2 cars with realistic movement
        if frame_idx == 0:
            # Initial detection
            cars['car_1'] = {
                'bbox': [1150, 100, 1250, 180],
                'points': self._generate_car_points(np.array([80, 5, 1]), 60 + frame_idx * 10),
                'confidence': 0.75,
                'position': np.array([80, 5, 1])
            }
        else:
            # Progressive accumulation with movement
            base_position = np.array([80 - frame_idx * 2, 5, 1])  # Moving closer
            cars['car_1'] = {
                'bbox': [1150 + frame_idx * 5, 100 - frame_idx * 2, 1250 + frame_idx * 5, 180 - frame_idx * 2],
                'points': self._generate_car_points(base_position, 60 + frame_idx * 15),
                'confidence': min(0.75 + frame_idx * 0.03, 0.95),
                'position': base_position
            }
            
            # Add second car later in sequence
            if frame_idx >= 3:
                cars['car_2'] = {
                    'bbox': [800, 150, 900, 220],
                    'points': self._generate_car_points(np.array([120, -8, 1]), 40 + (frame_idx-3) * 12),
                    'confidence': min(0.65 + (frame_idx-3) * 0.04, 0.90),
                    'position': np.array([120, -8, 1])
                }
        
        return cars
    
    def _generate_car_points(self, center: np.ndarray, num_points: int) -> List[np.ndarray]:
        """Generate realistic car point cloud"""
        points = []
        
        # Car body outline
        for i in range(int(num_points * 0.7)):
            # Random point within car bounds
            x = np.random.uniform(-2.5, 0.5) + center[0]
            y = np.random.uniform(-1.0, 1.0) + center[1]
            z = np.random.uniform(0.3, 2.0) + center[2]
            points.append(np.array([x, y, z]))
        
        # More dense points for car roof/structure
        for i in range(int(num_points * 0.3)):
            x = np.random.uniform(-2.0, 0) + center[0]
            y = np.random.uniform(-0.8, 0.8) + center[1]
            z = np.random.uniform(1.5, 1.8) + center[2]
            points.append(np.array([x, y, z]))
        
        return points
    
    def _create_frame_visualization(self, data: Dict, frame_idx: int, output_path: str, max_frames: int):
        """Create visualization for a single animation frame"""
        
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.figure_dpi)
        
        # Main layout: 2x3 grid
        gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[1.5, 1, 1])
        
        # Main 3D view
        ax_main = fig.add_subplot(gs[0, :2], projection='3d')
        ax_main.set_title(f"🚗 Passenger Detection Progress - Frame {frame_idx + 1}/{max_frames}", 
                         fontsize=16, fontweight='bold', pad=20)
        
        # Statistics panel
        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.set_title("📊 Detection Stats", fontsize=12, fontweight='bold')
        
        # Timeline
        ax_timeline = fig.add_subplot(gs[1, :])
        ax_timeline.set_title("🕐 Detection Timeline", fontsize=12, fontweight='bold')
        
        # Plot accumulated car and passenger data
        self._plot_3d_detection(ax_main, data, frame_idx)
        self._plot_detection_stats(ax_stats, data, frame_idx)
        self._plot_detection_timeline(ax_timeline, data, frame_idx, max_frames)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def _plot_3d_detection(self, ax, data: Dict, frame_idx: int):
        """Plot 3D car and passenger detection visualization"""
        
        car_data = data['car_data']
        passenger_data = data['passenger_data']
        
        # Plot car points with progressive colors
        for car_id, car_info in car_data.items():
            if not car_info['points']:
                continue
                
            points = np.array(car_info['points'])
            
            # Color points by detection frame
            point_colors = []
            for i, point in enumerate(points):
                frame_color_idx = min(i // (len(points) // 8 + 1), 7)
                point_colors.append(self.config.frame_colors[frame_color_idx])
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=point_colors, s=self.config.point_size * 8, alpha=0.6, label=f'Car {car_id}')
        
        # Plot passenger points with special highlighting
        for car_id, passengers in passenger_data.items():
            for i, passenger in enumerate(passengers):
                if len(passenger.points) == 0:
                    continue
                    
                # Highlight passenger points
                ax.scatter(passenger.points[:, 0], passenger.points[:, 1], passenger.points[:, 2],
                          c='red', s=self.config.point_size * 15, alpha=0.9, 
                          marker='o', edgecolors='darkred', linewidth=0.5,
                          label=f'Passenger {i+1} (conf: {passenger.confidence:.2f})')
                
                # Add passenger position marker
                pos = passenger.position
                ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          c='yellow', s=100, marker='*', 
                          edgecolors='orange', linewidth=2)
        
        # Styling
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        
        # Set reasonable view limits
        ax.set_xlim(70, 130)
        ax.set_ylim(-15, 15)
        ax.set_zlim(0, 3)
        
        # Better viewing angle for passenger detection
        ax.view_init(elev=20, azim=45)
        
        if len(car_data) > 0 or any(len(p) > 0 for p in passenger_data.values()):
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_detection_stats(self, ax, data: Dict, frame_idx: int):
        """Plot detection statistics panel"""
        ax.clear()
        ax.axis('off')
        
        car_data = data['car_data']
        passenger_data = data['passenger_data']
        
        # Calculate stats
        total_cars = len(car_data)
        total_passengers = sum(len(passengers) for passengers in passenger_data.values())
        total_points = sum(len(car_info['points']) for car_info in car_data.values())
        
        avg_confidence = 0
        if total_passengers > 0:
            confidences = []
            for passengers in passenger_data.values():
                confidences.extend([p.confidence for p in passengers])
            avg_confidence = np.mean(confidences)
        
        # Create stats text
        stats_text = f"""
🚗 Cars Detected: {total_cars}
👥 Passengers: {total_passengers}
📊 Total Points: {total_points:,}
🎯 Avg Confidence: {avg_confidence:.1%}
⏱️  Frame: {frame_idx + 1}

🔍 Detection Status:
"""
        
        for car_id, passengers in passenger_data.items():
            if passengers:
                stats_text += f"  Car {car_id}: {len(passengers)} passenger(s)\n"
            else:
                stats_text += f"  Car {car_id}: Scanning...\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
    
    def _plot_detection_timeline(self, ax, data: Dict, current_frame: int, max_frames: int):
        """Plot detection progress timeline"""
        ax.clear()
        
        # Timeline of detections
        frames = list(range(max_frames))
        car_counts = [0] * max_frames
        passenger_counts = [0] * max_frames
        
        # Fill in data up to current frame
        for i in range(current_frame + 1):
            if i < len(frames):
                car_counts[i] = len(data['car_data'])
                passenger_counts[i] = sum(len(passengers) for passengers in data['passenger_data'].values())
        
        # Plot timeline
        ax.bar(frames, car_counts, alpha=0.6, label='Cars Detected', color='lightblue')
        ax.bar(frames, passenger_counts, alpha=0.8, label='Passengers Detected', 
               color='orange', bottom=car_counts)
        
        # Highlight current frame
        ax.axvline(x=current_frame, color='red', linestyle='--', linewidth=2, label='Current Frame')
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Detections')
        ax.set_xlim(-0.5, max_frames - 0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_summary_visualization(self, frame_data: List[Dict], output_path: str):
        """Create final summary visualization showing complete detection sequence"""
        
        fig = plt.figure(figsize=(24, 12), dpi=self.config.figure_dpi)
        gs = fig.add_gridspec(2, 4, height_ratios=[2, 1])
        
        # Main summary view
        ax_main = fig.add_subplot(gs[0, :3], projection='3d')
        ax_main.set_title("🎬 Complete Passenger Detection Sequence", fontsize=18, fontweight='bold', pad=30)
        
        # Final statistics
        ax_final_stats = fig.add_subplot(gs[0, 3])
        ax_final_stats.set_title("📈 Final Results", fontsize=14, fontweight='bold')
        
        # Progress charts
        ax_progress = fig.add_subplot(gs[1, :2])
        ax_progress.set_title("📊 Detection Progress Over Time", fontsize=12, fontweight='bold')
        
        # Confidence evolution
        ax_confidence = fig.add_subplot(gs[1, 2:])
        ax_confidence.set_title("🎯 Confidence Evolution", fontsize=12, fontweight='bold')
        
        # Plot all accumulated data with color coding by frame
        final_data = frame_data[-1]  # Use final frame data
        
        self._plot_complete_detection_sequence(ax_main, frame_data)
        self._plot_final_statistics(ax_final_stats, final_data)
        self._plot_progress_charts(ax_progress, frame_data)
        self._plot_confidence_evolution(ax_confidence, frame_data)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def _plot_complete_detection_sequence(self, ax, frame_data: List[Dict]):
        """Plot the complete sequence showing all accumulated detections"""
        
        final_data = frame_data[-1]
        car_data = final_data['car_data']
        passenger_data = final_data['passenger_data']
        
        # Plot all car points with frame-based coloring
        for car_id, car_info in car_data.items():
            if not car_info['points']:
                continue
                
            points = np.array(car_info['points'])
            
            # Color by temporal progression
            num_segments = min(len(self.config.frame_colors), 8)
            points_per_segment = len(points) // num_segments + 1
            
            for i in range(num_segments):
                start_idx = i * points_per_segment
                end_idx = min((i + 1) * points_per_segment, len(points))
                
                if start_idx < len(points):
                    segment_points = points[start_idx:end_idx]
                    ax.scatter(segment_points[:, 0], segment_points[:, 1], segment_points[:, 2],
                              c=self.config.frame_colors[i], s=6, alpha=0.7)
        
        # Plot all passenger detections
        passenger_colors = ['red', 'orange', 'yellow', 'pink']
        for car_id, passengers in passenger_data.items():
            for i, passenger in enumerate(passengers):
                if len(passenger.points) == 0:
                    continue
                
                color = passenger_colors[i % len(passenger_colors)]
                ax.scatter(passenger.points[:, 0], passenger.points[:, 1], passenger.points[:, 2],
                          c=color, s=20, alpha=0.9, marker='o', edgecolors='darkred', linewidth=0.5)
                
                # Position marker
                pos = passenger.position
                ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          c='gold', s=200, marker='*', edgecolors='orange', linewidth=3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(elev=25, azim=45)
    
    def _plot_final_statistics(self, ax, final_data: Dict):
        """Plot final detection statistics"""
        ax.clear()
        ax.axis('off')
        
        car_data = final_data['car_data']
        passenger_data = final_data['passenger_data']
        
        total_cars = len(car_data)
        total_passengers = sum(len(passengers) for passengers in passenger_data.values())
        total_points = sum(len(car_info['points']) for car_info in car_data.values())
        
        # Calculate average confidence
        all_confidences = []
        for passengers in passenger_data.values():
            all_confidences.extend([p.confidence for p in passengers])
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        max_confidence = max(all_confidences) if all_confidences else 0
        
        stats_text = f"""
🎯 FINAL DETECTION RESULTS

🚗 Total Cars: {total_cars}
👥 Total Passengers: {total_passengers}
📊 Point Cloud Size: {total_points:,} points

📈 PERFORMANCE METRICS
🎯 Average Confidence: {avg_confidence:.1%}
⭐ Peak Confidence: {max_confidence:.1%}
📊 Detection Rate: {total_passengers/max(total_cars,1):.1f} passengers/car

✅ CAPABILITIES DEMONSTRATED
• Real-time LIDAR processing
• Progressive point accumulation
• Interior passenger detection
• Multi-passenger tracking
• Confidence-based filtering
• 3D position estimation
"""
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    def _plot_progress_charts(self, ax, frame_data: List[Dict]):
        """Plot detection progress over frames"""
        
        frames = [d['frame_idx'] for d in frame_data]
        car_counts = [len(d['car_data']) for d in frame_data]
        passenger_counts = [sum(len(passengers) for passengers in d['passenger_data'].values()) 
                           for d in frame_data]
        point_counts = [sum(len(car_info['points']) for car_info in d['car_data'].values()) 
                       for d in frame_data]
        
        # Normalize point counts for visualization
        point_counts_normalized = [p / 100 for p in point_counts]  # Scale down for visibility
        
        ax.plot(frames, car_counts, 'o-', label='Cars Detected', linewidth=2, markersize=8)
        ax.plot(frames, passenger_counts, 's-', label='Passengers Detected', linewidth=2, markersize=8)
        ax.plot(frames, point_counts_normalized, '^-', label='Point Cloud Size (×100)', linewidth=2, markersize=8)
        
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, len(frames) - 0.5)
    
    def _plot_confidence_evolution(self, ax, frame_data: List[Dict]):
        """Plot confidence evolution over time"""
        
        # Extract confidence data
        confidence_data = {}
        
        for frame_idx, data in enumerate(frame_data):
            for car_id, passengers in data['passenger_data'].items():
                for p_idx, passenger in enumerate(passengers):
                    key = f"Car_{car_id}_P{p_idx+1}"
                    if key not in confidence_data:
                        confidence_data[key] = {'frames': [], 'confidences': []}
                    
                    confidence_data[key]['frames'].append(frame_idx)
                    confidence_data[key]['confidences'].append(passenger.confidence)
        
        # Plot confidence evolution for each passenger
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (passenger_id, data) in enumerate(confidence_data.items()):
            color = colors[i % len(colors)]
            ax.plot(data['frames'], data['confidences'], 'o-', 
                   label=passenger_id, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        if confidence_data:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.text(0.5, 0.5, 'No passenger detections', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic')
    
    def _save_passenger_models(self, passenger_data: Dict, output_dir: str):
        """Save passenger point clouds as PCD files"""
        
        models_dir = os.path.join(output_dir, "passenger_models")
        os.makedirs(models_dir, exist_ok=True)
        
        for car_id, passengers in passenger_data.items():
            for p_idx, passenger in enumerate(passengers):
                if len(passenger.points) == 0:
                    continue
                
                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(passenger.points)
                
                # Add color based on confidence
                color_intensity = passenger.confidence
                colors = np.array([[1, 1-color_intensity, 1-color_intensity]] * len(passenger.points))
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
                # Save PCD file
                filename = f"car_{car_id}_passenger_{p_idx+1}_conf_{passenger.confidence:.2f}.pcd"
                filepath = os.path.join(models_dir, filename)
                o3d.io.write_point_cloud(filepath, pcd)
        
        print(f"💾 Passenger models saved in: {models_dir}")

def create_passenger_detection_demo(sequence_id: str = "seq-6", device_id: str = "105"):
    """Main function to create passenger detection demo animation"""
    
    print("🎬 Starting Passenger Detection Animation Demo")
    print("=" * 60)
    
    config = PassengerAnimationConfig()
    animator = PassengerDetectionAnimator(config)
    
    output_dir = animator.create_passenger_detection_animation(
        sequence_id=sequence_id,
        device_id=device_id,
        max_frames=8
    )
    
    print("\n" + "=" * 60)
    print("✅ Passenger Detection Demo Complete!")
    print(f"📁 Check output in: {output_dir}")
    print("\n🎯 Generated files:")
    print("   • passenger_detection_frame_XXX.png - Animation frames")
    print("   • passenger_detection_summary.png - Complete sequence overview")
    print("   • passenger_models/ - Individual passenger PCD files")
    print("\n💡 Use these files for your sales presentation!")
    
    return output_dir

if __name__ == "__main__":
    create_passenger_detection_demo()