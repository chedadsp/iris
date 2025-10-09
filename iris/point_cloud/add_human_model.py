#!/usr/bin/env python3
"""
Human Model Generator for Interior Cockpit

This script generates a point cloud model of a human sitting in a car
and adds it to the existing interior cockpit point cloud.

Author: Dimitrije Stojanovic
Date: September 2025
"""

import numpy as np
import argparse
from pathlib import Path


class HumanModelGenerator:
    """Generates a realistic point cloud model of a seated human."""
    
    def __init__(self):
        """Initialize the human model generator."""
        self.human_points = []
        
    def generate_ellipsoid_points(self, center, radii, num_points, z_range=None):
        """Generate points on an ellipsoid surface."""
        # Generate random angles
        u = np.random.uniform(0, 2 * np.pi, num_points)
        v = np.random.uniform(0, np.pi, num_points)
        
        # Convert to cartesian coordinates
        x = radii[0] * np.sin(v) * np.cos(u) + center[0]
        y = radii[1] * np.sin(v) * np.sin(u) + center[1]
        z = radii[2] * np.cos(v) + center[2]
        
        # Filter z range if specified
        if z_range is not None:
            mask = (z >= z_range[0]) & (z <= z_range[1])
            x, y, z = x[mask], y[mask], z[mask]
        
        return np.column_stack([x, y, z])
    
    def generate_cylinder_points(self, center, radius, height, num_points):
        """Generate points on a cylinder surface."""
        # Generate random angles and heights
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        z = np.random.uniform(center[2] - height/2, center[2] + height/2, num_points)
        
        # Convert to cartesian coordinates
        x = radius * np.cos(theta) + center[0]
        y = radius * np.sin(theta) + center[1]
        
        return np.column_stack([x, y, z])
    
    def generate_seated_human(self, seat_position, density=1.0):
        """
        Generate a point cloud model of a seated human.
        
        Args:
            seat_position: [x, y, z] position of the seat (person's hip level)
            density: Point density multiplier (1.0 = normal density)
        """
        self.human_points = []
        
        # Scale point counts based on density
        base_density = int(100 * density)
        
        # Human dimensions (approximate, scaled for seated position)
        seat_x, seat_y, seat_z = seat_position
        
        # 1. Head (ellipsoid)
        head_center = [seat_x, seat_y, seat_z + 0.65]  # Head above shoulders
        head_radii = [0.10, 0.12, 0.11]  # Slightly oval head
        head_points = self.generate_ellipsoid_points(
            head_center, head_radii, base_density
        )
        self.human_points.append(head_points)
        
        # 2. Neck (small cylinder)
        neck_center = [seat_x, seat_y, seat_z + 0.55]
        neck_points = self.generate_cylinder_points(
            neck_center, 0.06, 0.08, base_density // 4
        )
        self.human_points.append(neck_points)
        
        # 3. Torso (ellipsoid, upper body)
        torso_center = [seat_x, seat_y, seat_z + 0.35]
        torso_radii = [0.18, 0.12, 0.25]  # Wider chest
        torso_points = self.generate_ellipsoid_points(
            torso_center, torso_radii, base_density * 2
        )
        self.human_points.append(torso_points)
        
        # 4. Shoulders (extend torso width)
        left_shoulder = [seat_x - 0.20, seat_y, seat_z + 0.50]
        right_shoulder = [seat_x + 0.20, seat_y, seat_z + 0.50]
        for shoulder_pos in [left_shoulder, right_shoulder]:
            shoulder_points = self.generate_ellipsoid_points(
                shoulder_pos, [0.08, 0.08, 0.08], base_density // 2
            )
            self.human_points.append(shoulder_points)
        
        # 5. Arms (cylinders from shoulders)
        arm_length = 0.30
        for side in [-1, 1]:  # Left and right
            # Upper arm
            upper_arm_center = [seat_x + side * 0.25, seat_y, seat_z + 0.35]
            upper_arm_points = self.generate_cylinder_points(
                upper_arm_center, 0.05, arm_length, base_density // 2
            )
            self.human_points.append(upper_arm_points)
            
            # Forearm (bent at elbow)
            forearm_center = [seat_x + side * 0.30, seat_y + 0.15, seat_z + 0.20]
            forearm_points = self.generate_cylinder_points(
                forearm_center, 0.04, 0.25, base_density // 2
            )
            self.human_points.append(forearm_points)
            
            # Hands
            hand_center = [seat_x + side * 0.35, seat_y + 0.25, seat_z + 0.15]
            hand_points = self.generate_ellipsoid_points(
                hand_center, [0.04, 0.08, 0.02], base_density // 4
            )
            self.human_points.append(hand_points)
        
        # 6. Seated lower torso/hips
        hip_center = [seat_x, seat_y, seat_z]
        hip_radii = [0.15, 0.20, 0.10]  # Wider hips when seated
        hip_points = self.generate_ellipsoid_points(
            hip_center, hip_radii, base_density
        )
        self.human_points.append(hip_points)
        
        # 7. Thighs (horizontal cylinders for seated position)
        thigh_length = 0.35
        for side in [-1, 1]:  # Left and right thigh
            thigh_center = [seat_x + side * 0.08, seat_y + 0.15, seat_z - 0.05]
            # Generate points along the thigh
            theta = np.random.uniform(0, 2 * np.pi, base_density)
            r = np.random.uniform(0, 0.08, base_density)  # Thigh radius
            y_offset = np.random.uniform(0, thigh_length, base_density)
            
            x = r * np.cos(theta) + thigh_center[0]
            y = y_offset + thigh_center[1]
            z = r * np.sin(theta) + thigh_center[2]
            
            thigh_points = np.column_stack([x, y, z])
            self.human_points.append(thigh_points)
        
        # 8. Knees
        for side in [-1, 1]:
            knee_center = [seat_x + side * 0.08, seat_y + 0.45, seat_z - 0.05]
            knee_points = self.generate_ellipsoid_points(
                knee_center, [0.06, 0.06, 0.06], base_density // 3
            )
            self.human_points.append(knee_points)
        
        # 9. Lower legs (vertical cylinders)
        lower_leg_height = 0.30
        for side in [-1, 1]:
            lower_leg_center = [seat_x + side * 0.08, seat_y + 0.45, seat_z - 0.20]
            lower_leg_points = self.generate_cylinder_points(
                lower_leg_center, 0.05, lower_leg_height, base_density // 2
            )
            self.human_points.append(lower_leg_points)
        
        # 10. Feet
        for side in [-1, 1]:
            foot_center = [seat_x + side * 0.08, seat_y + 0.55, seat_z - 0.35]
            foot_points = self.generate_ellipsoid_points(
                foot_center, [0.05, 0.12, 0.04], base_density // 3
            )
            self.human_points.append(foot_points)
        
        # Combine all points
        all_human_points = np.vstack(self.human_points)
        return all_human_points
    
    def add_noise_and_variation(self, points, noise_level=0.005):
        """Add realistic noise to make the human model look more natural."""
        noise = np.random.normal(0, noise_level, points.shape)
        return points + noise
    
    def scale_to_cockpit(self, points, cockpit_bounds):
        """Scale and position the human model to fit within cockpit bounds."""
        # Calculate cockpit dimensions
        x_range = cockpit_bounds['x_max'] - cockpit_bounds['x_min']
        y_range = cockpit_bounds['y_max'] - cockpit_bounds['y_min']
        z_range = cockpit_bounds['z_max'] - cockpit_bounds['z_min']
        
        # Position human in the center-front of cockpit (driver's seat)
        seat_x = cockpit_bounds['x_min'] + x_range * 0.3  # 30% from front
        seat_y = cockpit_bounds['y_min'] + y_range * 0.4  # 40% from left (driver side)
        seat_z = cockpit_bounds['z_max'] - 0.1  # Slightly below roof
        
        return [seat_x, seat_y, seat_z]


def add_human_to_cockpit(input_file, output_file, density=1.0, noise_level=0.005):
    """
    Add a human model to the existing cockpit point cloud.
    
    Args:
        input_file: Path to the input cockpit .npy file
        output_file: Path to save the combined point cloud
        density: Point density for the human model
        noise_level: Amount of noise to add for realism
    """
    # Load existing cockpit points
    print(f"Loading cockpit data from: {input_file}")
    cockpit_points = np.load(input_file)
    print(f"Loaded {len(cockpit_points)} cockpit points")
    
    # Calculate cockpit bounds
    cockpit_bounds = {
        'x_min': cockpit_points[:, 0].min(),
        'x_max': cockpit_points[:, 0].max(),
        'y_min': cockpit_points[:, 1].min(),
        'y_max': cockpit_points[:, 1].max(),
        'z_min': cockpit_points[:, 2].min(),
        'z_max': cockpit_points[:, 2].max()
    }
    
    print(f"Cockpit bounds:")
    print(f"  X: {cockpit_bounds['x_min']:.3f} to {cockpit_bounds['x_max']:.3f} ({cockpit_bounds['x_max'] - cockpit_bounds['x_min']:.3f}m)")
    print(f"  Y: {cockpit_bounds['y_min']:.3f} to {cockpit_bounds['y_max']:.3f} ({cockpit_bounds['y_max'] - cockpit_bounds['y_min']:.3f}m)")
    print(f"  Z: {cockpit_bounds['z_min']:.3f} to {cockpit_bounds['z_max']:.3f} ({cockpit_bounds['z_max'] - cockpit_bounds['z_min']:.3f}m)")
    
    # Generate human model
    print("Generating human model...")
    generator = HumanModelGenerator()
    
    # Calculate optimal seat position
    seat_position = generator.scale_to_cockpit(None, cockpit_bounds)
    print(f"Placing human at seat position: [{seat_position[0]:.3f}, {seat_position[1]:.3f}, {seat_position[2]:.3f}]")
    
    # Generate the human
    human_points = generator.generate_seated_human(seat_position, density)
    print(f"Generated {len(human_points)} human points")
    
    # Add noise for realism
    if noise_level > 0:
        human_points = generator.add_noise_and_variation(human_points, noise_level)
        print(f"Added noise with level {noise_level}")
    
    # Combine cockpit and human points
    combined_points = np.vstack([cockpit_points, human_points])
    print(f"Combined total: {len(combined_points)} points")
    
    # Save the result
    np.save(output_file, combined_points)
    print(f"Saved combined point cloud to: {output_file}")
    
    # Print statistics
    print("\nFinal statistics:")
    print(f"  Original cockpit points: {len(cockpit_points):,}")
    print(f"  Added human points: {len(human_points):,}")
    print(f"  Total points: {len(combined_points):,}")
    print(f"  Human ratio: {len(human_points)/len(combined_points)*100:.1f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Add human model to cockpit point cloud")
    parser.add_argument("--input", default="output/interior_cockpit.npy", 
                       help="Input cockpit .npy file")
    parser.add_argument("--output", default="output/interior_cockpit_with_human.npy",
                       help="Output file for combined point cloud")
    parser.add_argument("--density", type=float, default=1.0,
                       help="Point density multiplier for human model")
    parser.add_argument("--noise", type=float, default=0.005,
                       help="Noise level for realistic appearance")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite the original file instead of creating new one")
    
    args = parser.parse_args()
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist!")
        return 1
    
    # Determine output file
    if args.overwrite:
        output_path = input_path
        print("Warning: Will overwrite the original file!")
    else:
        output_path = Path(args.output)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add human to cockpit
    add_human_to_cockpit(
        input_path, 
        output_path, 
        density=args.density,
        noise_level=args.noise
    )
    
    print(f"\nSuccessfully added human model to cockpit!")
    print(f"Result saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
