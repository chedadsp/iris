#!/usr/bin/env python3
"""
Temporal Change Analysis Demo

This script demonstrates the enhanced sequence LIDAR analysis with temporal change detection.
It shows how to identify points that actually moved during a time sequence, filtering out
static elements that might have been misclassified as dynamic.

Usage:
    python demo_temporal_analysis.py --sequence seq-53 --device 105
"""

import argparse
from pathlib import Path
from sequence_lidar_analysis import SequenceLidarAnalyzer

def main():
    """Demonstrate temporal change analysis."""
    parser = argparse.ArgumentParser(description="Temporal Change Analysis Demo")
    parser.add_argument("--sequence", required=True, help="Sequence ID (e.g., seq-53)")
    parser.add_argument("--device", required=True, help="Device ID (e.g., 105)")
    parser.add_argument("--max_scans", type=int, default=5, help="Maximum scans to process")

    args = parser.parse_args()

    print("🚀 Starting Enhanced Sequence LIDAR Analysis Demo")
    print("=" * 60)
    print(f"Sequence: {args.sequence}")
    print(f"Device: {args.device}")
    print(f"Max scans: {args.max_scans}")
    print()

    # Initialize analyzer
    analyzer = SequenceLidarAnalyzer(
        correspondence_file="data/RCooper/corespondence_camera_lidar.json",
        base_data_path="data/RCooper"
    )

    print("📊 Step 1: Standard Static/Dynamic Analysis")
    print("-" * 40)

    # Perform analysis
    results = analyzer.analyze_sequence(
        sequence_id=args.sequence,
        device_id=args.device,
        max_scans=args.max_scans,
        visualize=True
    )

    if not results:
        print("❌ No results found")
        return

    # Get results for our sequence
    sequence_key = f"{args.device}_{args.sequence}"
    result = results.get(sequence_key)

    if not result:
        print(f"❌ No results for {sequence_key}")
        return

    print(f"\n🎯 Analysis Results Summary:")
    print(f"   Static points: {result['static_points']:,}")
    print(f"   Dynamic points: {result['dynamic_points']:,}")
    print(f"   Temporal change points: {result['temporal_change_points']:,}")

    # Calculate filtering effectiveness
    temporal_stats = result['stats'].get('temporal_analysis', {})
    if temporal_stats:
        original_dynamic = temporal_stats.get('total_dynamic_points', 0)
        temporal_changes = temporal_stats.get('temporal_change_points', 0)
        filtering_ratio = (original_dynamic - temporal_changes) / original_dynamic if original_dynamic > 0 else 0

        print(f"\n📈 Temporal Filtering Results:")
        print(f"   Original dynamic points: {original_dynamic:,}")
        print(f"   Actual temporal changes: {temporal_changes:,}")
        print(f"   Filtered out: {original_dynamic - temporal_changes:,} ({filtering_ratio:.1%})")
        print(f"   Average movement: {temporal_stats.get('avg_movement_distance', 0):.2f}m")

    print(f"\n📁 Generated Files:")
    output_dir = Path("sensor_fusion_output") / "sequence_analysis" / sequence_key
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        for file in sorted(files):
            print(f"   📄 {file.name}")

    print(f"\n🔍 Key Insights:")
    print(f"   • Static points represent fixed environment (buildings, ground)")
    print(f"   • Dynamic points include all moving and varying elements")
    print(f"   • Temporal change points are the subset that actually moved over time")
    print(f"   • This filtering removes noise and improves motion detection accuracy")

    print(f"\n✅ Demo completed! Check the visualization files for detailed analysis.")


if __name__ == "__main__":
    main()