#!/usr/bin/env python3
"""
Create professional animation showing sensor fusion process for sales presentations.

This script creates multiple animation types:
1. Point cloud accumulation over time
2. Sensor fusion process (camera + LIDAR)
3. Vehicle tracking progression
4. 3D model building

Perfect for sales decks and presentations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
import json
import cv2
import argparse
from typing import List, Dict, Tuple
import open3d as o3d
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Professional color scheme
COLORS = {
    'primary': '#2E86AB',
    'accent': '#A23B72',
    'success': '#F18F01',
    'dynamic': '#C73E1D',
    'static': '#6C757D',
    'bg_light': '#F8F9FA',
    'bg_dark': '#212529'
}


class SensorFusionAnimator:
    """Create animations showing sensor fusion and tracking process."""

    def __init__(self, sequence: str, device: str, output_dir: Path):
        self.sequence = sequence
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load tracking data
        self.tracking_file = Path(f"sensor_fusion_output/enhanced_tracking/{device}_{sequence}/{device}_{sequence}_enhanced_tracking.json")
        self.tracking_data = self._load_tracking_data()

        # Input data paths
        self.camera_dir = Path(f"sensor_fusion_output_3/{device}_{sequence}")

        print(f"📊 Loaded tracking data for {len(self.tracking_data.get('cars', {}))} cars")

    def _load_tracking_data(self) -> Dict:
        """Load tracking results."""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}

    def create_point_accumulation_animation(self, car_id: int, fps: int = 2) -> str:
        """
        Create animation showing progressive point cloud accumulation for a car.

        Args:
            car_id: Car ID to animate
            fps: Frames per second

        Returns:
            Path to created animation file
        """
        print(f"\n🎬 Creating point accumulation animation for Car {car_id}...")

        car_data = self.tracking_data.get('cars', {}).get(str(car_id))
        if not car_data:
            print(f"❌ No data found for car {car_id}")
            return None

        detections = car_data.get('detections', [])
        if not detections:
            print(f"❌ No detections for car {car_id}")
            return None

        # Create figure
        fig = plt.figure(figsize=(16, 9), facecolor=COLORS['bg_light'])
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Main 3D view (large, left side)
        ax_3d = fig.add_subplot(gs[:, :2], projection='3d', facecolor=COLORS['bg_light'])

        # Top view (top right)
        ax_top = fig.add_subplot(gs[0, 2], facecolor=COLORS['bg_light'])

        # Metrics panel (bottom right)
        ax_metrics = fig.add_subplot(gs[1, 2], facecolor=COLORS['bg_light'])
        ax_metrics.axis('off')

        # Animation elements
        accumulated_points = []
        frame_colors = plt.cm.viridis(np.linspace(0, 1, len(detections)))

        def init():
            """Initialize animation."""
            ax_3d.clear()
            ax_top.clear()
            return []

        def update(frame_idx):
            """Update animation for given frame."""
            ax_3d.clear()
            ax_top.clear()
            ax_metrics.clear()
            ax_metrics.axis('off')

            # Accumulate points up to current frame
            current_accumulated = []
            current_colors = []

            for i in range(frame_idx + 1):
                detection = detections[i]

                # Load car model for this detection (simplified - would load actual PCD)
                # For now, generate synthetic points based on detection
                num_points = detection.get('point_count', 100)

                # Simulate points around bounding box center
                bbox = detection.get('bbox', [0, 0, 100, 100])
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2

                # Generate points (in practice, load from PCD file)
                points = np.random.randn(num_points, 3) * 0.5
                points[:, 0] += center_x / 100  # Normalize
                points[:, 1] += center_y / 100

                current_accumulated.extend(points)
                current_colors.extend([frame_colors[i]] * num_points)

            current_accumulated = np.array(current_accumulated)
            current_colors = np.array(current_colors)

            # Plot 3D view
            if len(current_accumulated) > 0:
                ax_3d.scatter(current_accumulated[:, 0],
                            current_accumulated[:, 1],
                            current_accumulated[:, 2],
                            c=current_colors, s=2, alpha=0.6)

            ax_3d.set_xlabel('X', fontsize=10, color=COLORS['primary'])
            ax_3d.set_ylabel('Y', fontsize=10, color=COLORS['primary'])
            ax_3d.set_zlabel('Z', fontsize=10, color=COLORS['primary'])
            ax_3d.set_title(f'3D Point Cloud - Car {car_id}',
                          fontsize=14, fontweight='bold', color=COLORS['primary'])
            ax_3d.view_init(elev=20, azim=45 + frame_idx * 5)  # Rotate view

            # Plot top view
            if len(current_accumulated) > 0:
                ax_top.scatter(current_accumulated[:, 0],
                             current_accumulated[:, 1],
                             c=current_colors, s=1, alpha=0.6)
            ax_top.set_xlabel('X', fontsize=9)
            ax_top.set_ylabel('Y', fontsize=9)
            ax_top.set_title('Top View', fontsize=11, fontweight='bold')
            ax_top.set_aspect('equal')
            ax_top.grid(True, alpha=0.3)

            # Metrics panel
            detection = detections[frame_idx]
            total_points = sum(d.get('point_count', 0) for d in detections[:frame_idx + 1])

            metrics_text = f"""
            FRAME: {frame_idx + 1}/{len(detections)}

            Current Frame Points: {detection.get('point_count', 0)}
            Total Accumulated: {total_points}

            Confidence: {detection.get('confidence', 0):.2%}

            Frames Tracked: {frame_idx + 1}
            Coverage: {(frame_idx + 1) / len(detections):.1%}
            """

            ax_metrics.text(0.1, 0.5, metrics_text,
                          transform=ax_metrics.transAxes,
                          fontsize=11, verticalalignment='center',
                          fontfamily='monospace',
                          color=COLORS['primary'],
                          bbox=dict(boxstyle='round', facecolor=COLORS['bg_light'],
                                  edgecolor=COLORS['primary'], linewidth=2))

            # Progress bar
            progress = (frame_idx + 1) / len(detections)
            progress_bar = Rectangle((0.1, 0.15), 0.8 * progress, 0.05,
                                    transform=ax_metrics.transAxes,
                                    facecolor=COLORS['success'], edgecolor=COLORS['primary'],
                                    linewidth=2)
            ax_metrics.add_patch(progress_bar)

            return []

        # Create animation
        anim = animation.FuncAnimation(fig, update, init_func=init,
                                      frames=len(detections), interval=1000//fps,
                                      blit=True, repeat=True)

        # Save animation
        output_file = self.output_dir / f"car_{car_id}_point_accumulation.mp4"

        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='Nortality'), bitrate=1800)
            anim.save(str(output_file), writer=writer, dpi=150)
            print(f"✅ Saved animation: {output_file}")
            return str(output_file)
        except Exception as e:
            print(f"⚠️  Could not save as video (ffmpeg may not be installed): {e}")
            print(f"   Saving individual frames instead...")

            # Save as individual frames
            frames_dir = self.output_dir / f"car_{car_id}_frames"
            frames_dir.mkdir(exist_ok=True)

            for frame_idx in range(len(detections)):
                update(frame_idx)
                frame_file = frames_dir / f"frame_{frame_idx+1:03d}.png"
                plt.savefig(frame_file, dpi=150, bbox_inches='tight',
                          facecolor=COLORS['bg_light'])
                print(f"   Saved frame {frame_idx+1}/{len(detections)}")

            print(f"✅ Saved {len(detections)} frames to: {frames_dir}")

            # Create instructions for manual video creation
            instructions_file = frames_dir / "CREATE_VIDEO.txt"
            with open(instructions_file, 'w') as f:
                f.write(f"""To create video from frames, use ffmpeg:

cd {frames_dir}
ffmpeg -framerate {fps} -i frame_%03d.png -c:v libx264 -pix_fmt yuv420p -crf 18 ../car_{car_id}_point_accumulation.mp4

Or use online tools like:
- ezgif.com (frames to video)
- imagemagick: convert -delay {100//fps} frame_*.png animation.gif
""")

            return str(frames_dir)
        finally:
            plt.close()

    def create_sensor_fusion_explainer(self) -> str:
        """
        Create step-by-step animation explaining sensor fusion process.

        Returns:
            Path to created frames directory
        """
        print("\n🎬 Creating sensor fusion explainer animation...")

        frames_dir = self.output_dir / "fusion_explainer_frames"
        frames_dir.mkdir(exist_ok=True)

        # Get first car's first detection for demo
        cars = self.tracking_data.get('cars', {})
        if not cars:
            print("❌ No car data available")
            return None

        car_id = list(cars.keys())[0]
        car_data = cars[car_id]
        detections = car_data.get('detections', [])

        if not detections:
            print("❌ No detections available")
            return None

        detection = detections[0]

        # Load camera image
        camera_image = None
        cam_id = detection.get('camera')
        timestamp = detection.get('timestamp')

        if cam_id and timestamp:
            # Find matching camera image
            cam_images = list(self.camera_dir.glob(f"{cam_id}/frame_*.jpg"))
            if cam_images:
                camera_image = cv2.imread(str(cam_images[0]))
                camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

        # Create frames showing fusion process
        steps = [
            {
                'title': 'Step 1: Camera Detection',
                'description': 'YOLO detects vehicle in camera image\nwith bounding box and confidence score',
                'show': ['camera', 'bbox']
            },
            {
                'title': 'Step 2: Calibration Projection',
                'description': '2D bounding box projected into 3D space\nusing camera calibration matrices',
                'show': ['camera', 'bbox', 'frustum']
            },
            {
                'title': 'Step 3: LIDAR Association',
                'description': 'LIDAR points within 3D frustum\nare associated with detected vehicle',
                'show': ['camera', 'bbox', 'frustum', 'lidar']
            },
            {
                'title': 'Step 4: Fused Output',
                'description': 'Combined result: labeled 3D point cloud\nwith visual context and depth data',
                'show': ['camera', 'bbox', 'frustum', 'lidar', 'result']
            }
        ]

        for step_idx, step in enumerate(steps):
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=COLORS['bg_light'])

            # Left: Camera view
            ax_cam = axes[0]
            if camera_image is not None:
                ax_cam.imshow(camera_image)

                # Draw bounding box if in show list
                if 'bbox' in step['show']:
                    bbox = detection.get('bbox', [0, 0, 100, 100])
                    rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                   linewidth=3, edgecolor=COLORS['success'],
                                   facecolor='none', linestyle='--')
                    ax_cam.add_patch(rect)

                    # Add label
                    conf = detection.get('confidence', 0)
                    ax_cam.text(bbox[0], bbox[1]-10, f'Car {car_id}\n{conf:.1%}',
                              color='white', fontsize=12, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.8))

            ax_cam.set_title('Camera View', fontsize=14, fontweight='bold',
                           color=COLORS['primary'])
            ax_cam.axis('off')

            # Right: 3D LIDAR view
            ax_3d = fig.add_subplot(122, projection='3d', facecolor=COLORS['bg_light'])

            # Generate sample LIDAR points
            if 'lidar' in step['show']:
                points = np.random.randn(500, 3) * 2
                ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=COLORS['primary'], s=2, alpha=0.6, label='LIDAR Points')

                # Highlight associated points
                if 'result' in step['show']:
                    car_points = np.random.randn(200, 3) * 0.5
                    ax_3d.scatter(car_points[:, 0], car_points[:, 1], car_points[:, 2],
                                c=COLORS['success'], s=5, alpha=0.8, label='Car Points')

            # Draw frustum if in show list
            if 'frustum' in step['show']:
                # Simplified frustum visualization
                frustum_points = np.array([
                    [0, 0, 0], [2, 2, 5], [2, -2, 5], [-2, -2, 5], [-2, 2, 5]
                ])

                # Draw frustum edges
                for i in range(1, 5):
                    ax_3d.plot([frustum_points[0, 0], frustum_points[i, 0]],
                             [frustum_points[0, 1], frustum_points[i, 1]],
                             [frustum_points[0, 2], frustum_points[i, 2]],
                             color=COLORS['accent'], linewidth=2, alpha=0.6)

            ax_3d.set_xlabel('X', color=COLORS['primary'])
            ax_3d.set_ylabel('Y', color=COLORS['primary'])
            ax_3d.set_zlabel('Z', color=COLORS['primary'])
            ax_3d.set_title('3D LIDAR Space', fontsize=14, fontweight='bold',
                          color=COLORS['primary'])
            ax_3d.legend(loc='upper right')

            # Add step title and description
            fig.suptitle(step['title'], fontsize=18, fontweight='bold',
                        color=COLORS['primary'], y=0.98)
            fig.text(0.5, 0.92, step['description'], ha='center', fontsize=12,
                    color=COLORS['static'], style='italic')

            # Add step indicator
            progress_text = f"Step {step_idx + 1} of {len(steps)}"
            fig.text(0.95, 0.02, progress_text, ha='right', fontsize=10,
                    color=COLORS['primary'], fontweight='bold')

            # Save frame
            frame_file = frames_dir / f"step_{step_idx+1:02d}_{step['title'].replace(' ', '_').replace(':', '')}.png"
            plt.savefig(frame_file, dpi=200, bbox_inches='tight', facecolor=COLORS['bg_light'])
            print(f"   Saved: {frame_file.name}")
            plt.close()

        print(f"✅ Created {len(steps)} fusion explainer frames in: {frames_dir}")

        # Create instructions
        instructions_file = frames_dir / "README.txt"
        with open(instructions_file, 'w') as f:
            f.write("""Sensor Fusion Explainer Frames

These frames explain the sensor fusion process step-by-step.
Use them in presentations to show:

1. How camera detects vehicles
2. How 2D boxes are projected to 3D
3. How LIDAR points are associated
4. The final fused result

To create a video:
ffmpeg -framerate 0.5 -pattern_type glob -i 'step_*.png' -c:v libx264 -pix_fmt yuv420p fusion_explainer.mp4

(0.5 fps = 2 seconds per frame, adjust as needed)
""")

        return str(frames_dir)

    def create_complete_presentation_sequence(self) -> Dict[str, str]:
        """
        Create all animation types needed for complete sales presentation.

        Returns:
            Dictionary mapping animation type to file path
        """
        print("\n🎬 Creating Complete Presentation Animation Sequence")
        print("=" * 60)

        results = {}

        # 1. Sensor fusion explainer
        print("\n1️⃣  Sensor Fusion Process Explainer")
        fusion_path = self.create_sensor_fusion_explainer()
        if fusion_path:
            results['fusion_explainer'] = fusion_path

        # 2. Point accumulation for each car
        cars = self.tracking_data.get('cars', {})
        for car_id in cars.keys():
            print(f"\n2️⃣  Point Accumulation Animation - Car {car_id}")
            anim_path = self.create_point_accumulation_animation(int(car_id))
            if anim_path:
                results[f'car_{car_id}_accumulation'] = anim_path

        # 3. Create summary
        print("\n" + "=" * 60)
        print("📊 Animation Creation Summary")
        print("=" * 60)
        for anim_type, path in results.items():
            print(f"✅ {anim_type}: {path}")

        print(f"\n📁 All outputs saved to: {self.output_dir}")
        print("\n💡 Next steps:")
        print("   1. Review generated frames and videos")
        print("   2. If videos weren't created, use ffmpeg commands in README files")
        print("   3. Import into PowerPoint/Keynote presentation")
        print("   4. Add narration and transitions")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Create sensor fusion animations for sales presentations'
    )
    parser.add_argument('--sequence', default='seq-6', help='Sequence ID')
    parser.add_argument('--device', default='105', help='Device ID')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: sensor_fusion_output/animations/DEVICE_SEQUENCE)')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"sensor_fusion_output/animations/{args.device}_{args.sequence}")

    print("🎬 Nortality Sensor Fusion Animation Creator")
    print("=" * 60)
    print(f"📊 Sequence: {args.sequence}")
    print(f"🔧 Device: {args.device}")
    print(f"📁 Output: {output_dir}")
    print("=" * 60)

    # Create animator
    animator = SensorFusionAnimator(args.sequence, args.device, output_dir)

    # Generate all animations
    results = animator.create_complete_presentation_sequence()

    if results:
        print("\n✅ Animation creation completed successfully!")
        return 0
    else:
        print("\n❌ Animation creation failed")
        return 1


if __name__ == "__main__":
    exit(main())
