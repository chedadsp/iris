#!/usr/bin/env python3
"""
Sales Deck Visualization Launcher

This script generates compelling sales deck visualizations for the enhanced car tracking system.
It creates professional visualizations showing the complete pipeline from LIDAR scene overview
to detailed car detection, tracking, and 3D model building.

Usage:
    python sales_deck_launcher.py --sequence seq-53 --device 105 --max_frames 8
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from sales_deck_visualizations import SalesDeckVisualizer
    print("✅ Sales deck visualizations module loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import sales deck module: {e}")
    print("⚠️  Running with fallback enhanced tracking...")
    
    # Fallback to enhanced tracking only
    try:
        from enhanced_car_tracking import EnhancedCarTracker
        print("✅ Enhanced tracking module loaded")
        
        def run_enhanced_tracking_only(sequence_id: str, device_id: str, max_frames: int):
            """Run enhanced tracking with improved visualizations."""
            print(f"🚗 Running enhanced tracking for {device_id}_{sequence_id}")
            
            tracker = EnhancedCarTracker(
                correspondence_file="data/RCooper/corespondence_camera_lidar.json",
                data_path="data/RCooper"
            )
            
            sequence_key = f"{device_id}_{sequence_id}"
            car_models = tracker.process_sequence_with_accumulation(
                sequence_id, device_id, max_frames
            )
            
            if car_models:
                print(f"📊 Found {len(car_models)} car models")
                
                # Enhanced visualization with existing system
                tracker.visualize_progressive_building(car_models, sequence_key, save_output=True)
                tracker.save_tracking_results(car_models, sequence_key)
                
                print(f"✅ Enhanced tracking visualization complete!")
                print(f"📁 Results saved in: sensor_fusion_output/enhanced_tracking/{sequence_key}/")
            else:
                print("❌ No car models found")
        
        # Set the fallback function
        main_function = run_enhanced_tracking_only
        
    except ImportError as e2:
        print(f"❌ Failed to import enhanced tracking: {e2}")
        sys.exit(1)
else:
    # Use the full sales deck system
    def run_sales_deck_generation(sequence_id: str, device_id: str, max_frames: int):
        """Run the complete sales deck generation."""
        print(f"🎯 Generating sales deck for {device_id}_{sequence_id}")
        
        # Initialize visualizer
        visualizer = SalesDeckVisualizer()
        
        try:
            visualizer.initialize_systems(
                correspondence_file="data/RCooper/corespondence_camera_lidar.json",
                data_path="data/RCooper"
            )
            
            # Generate complete sales deck
            output_dir = visualizer.generate_complete_sales_deck(
                sequence_id, device_id, max_frames
            )
            
            print(f"\n🎯 Sales deck generation complete!")
            print(f"📁 Output: {output_dir}")
            
        except Exception as e:
            print(f"❌ Error in sales deck generation: {e}")
            print("🔄 Falling back to enhanced tracking...")
            
            # Fallback to enhanced tracking
            from enhanced_car_tracking import EnhancedCarTracker
            
            tracker = EnhancedCarTracker(
                correspondence_file="data/RCooper/corespondence_camera_lidar.json",
                data_path="data/RCooper"
            )
            
            sequence_key = f"{device_id}_{sequence_id}"
            car_models = tracker.process_sequence_with_accumulation(
                sequence_id, device_id, max_frames
            )
            
            if car_models:
                tracker.visualize_progressive_building(car_models, sequence_key, save_output=True)
                tracker.save_tracking_results(car_models, sequence_key)
                print(f"✅ Fallback enhanced tracking complete!")
    
    main_function = run_sales_deck_generation


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate Sales Deck Visualizations")
    parser.add_argument("--sequence", default="seq-53", help="Sequence ID (e.g., seq-53)")
    parser.add_argument("--device", default="105", help="Device ID (e.g., 105)")
    parser.add_argument("--max_frames", type=int, default=8, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    print("🚀 Starting Sales Deck Visualization System")
    print(f"Parameters: {args.device}_{args.sequence}, max_frames={args.max_frames}")
    print()
    
    # Run the appropriate function
    main_function(args.sequence, args.device, args.max_frames)
    
    print("\n✅ Process complete!")


if __name__ == "__main__":
    main()