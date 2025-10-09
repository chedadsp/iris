#!/usr/bin/env python3
"""
Simple launcher for passenger detection animation demo
For sales MVP demonstration
"""

import sys
import os
sys.path.append('/Users/Dimitrije.Stojanovic/work/nortality/point_cloud/src')

def run_passenger_detection_demo():
    """Run the passenger detection animation demo"""
    
    print("🎬 Passenger Detection Animation Demo")
    print("=" * 50)
    print("🎯 Creating MVP animation for sales presentation")
    print("📊 Simulating LIDAR passenger detection with progressive accumulation")
    print()
    
    try:
        # Import the passenger detection system
        from passenger_detection_animation import create_passenger_detection_demo
        
        # Run the demo
        output_dir = create_passenger_detection_demo(
            sequence_id="seq-6",
            device_id="105"
        )
        
        print(f"\n✅ Animation generation complete!")
        print(f"📁 Output directory: {output_dir}")
        
        return output_dir
        
    except Exception as e:
        print(f"❌ Error during animation generation: {e}")
        
        # Fallback to simple test
        print("\n🔄 Running simplified test...")
        test_passenger_simulation()
        
        return None

def test_passenger_simulation():
    """Test basic passenger simulation without full dependencies"""
    
    print("🧪 Testing passenger detection simulation...")
    
    # Create basic output directory
    output_dir = "sensor_fusion_output/passenger_detection_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simple demonstration file
    demo_info = """
# Passenger Detection Animation Demo

## Concept Overview
This animation demonstrates LIDAR detecting passengers inside vehicles using progressive point accumulation.

## Animation Sequence:
1. **Frame 1-2**: Initial car detection with YOLO
2. **Frame 3-4**: LIDAR point accumulation starts
3. **Frame 5-6**: Passenger shapes begin to emerge
4. **Frame 7-8**: Clear passenger silhouettes with confidence scores

## Technical Features:
- Progressive LIDAR point buildup with color coding
- Simulated passenger detection within car interior bounds
- Confidence scoring based on point density
- Multiple passenger detection (driver + passengers)
- 3D position estimation and tracking

## Sales Value Proposition:
- Interior monitoring capabilities
- Real-time passenger counting
- Safety and security applications
- Advanced sensor fusion technology

## Files that would be generated:
- passenger_detection_frame_001.png to passenger_detection_frame_008.png
- passenger_detection_summary.png
- passenger_models/ directory with PCD files
- Individual passenger point clouds with confidence scores

## Use Cases:
- Autonomous vehicle safety systems
- Public transport monitoring
- Security and surveillance
- Occupancy detection
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(demo_info)
    
    print(f"✅ Demo information created in: {output_dir}")
    print("📝 Check README.md for concept details")
    
    return output_dir

if __name__ == "__main__":
    run_passenger_detection_demo()