#!/usr/bin/env python3
"""
Quick test script demonstrating the integrated sensor fusion workflow:
1. Run sequence-based sensor fusion (camera + LIDAR)
2. Run sequence LIDAR dynamic analysis (static vs dynamic points)

This showcases the complete pipeline requested by the user.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run both sensor fusion and dynamic analysis on a test sequence."""

    # Test sequence parameters
    test_sequence = "seq-53"
    test_device = "105"
    max_scans = 5  # Limit for quick testing

    print("🚀 Starting integrated sensor fusion test workflow")
    print(f"   Sequence: {test_sequence}")
    print(f"   Device: {test_device}")
    print(f"   Max scans: {max_scans}")
    print("=" * 60)

    # Step 1: Run sequence-based sensor fusion
    print("\n📷 Step 1: Running sequence-based sensor fusion...")
    try:
        result = subprocess.run([
            "poetry", "run", "python", "src/sequence_fusion.py",
            "--sequence", test_sequence,
            "--device", test_device
        ], cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("✅ Sensor fusion completed successfully")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Sensor fusion failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Sensor fusion timed out")
        return False
    except Exception as e:
        print(f"❌ Error running sensor fusion: {e}")
        return False

    # Step 2: Run sequence LIDAR dynamic analysis
    print(f"\n🔍 Step 2: Running sequence LIDAR dynamic analysis...")
    try:
        result = subprocess.run([
            "poetry", "run", "python", "src/sequence_lidar_analysis.py",
            "--sequence", test_sequence,
            "--device", test_device,
            "--max_scans", str(max_scans)
        ], cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ Dynamic analysis completed successfully")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Dynamic analysis failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Dynamic analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Error running dynamic analysis: {e}")
        return False

    # Summary
    print("\n" + "=" * 60)
    print("🎯 Integrated workflow completed successfully!")
    print("\n📁 Generated outputs:")
    print("   • Sensor fusion results: sensor_fusion_output/105_seq-53/")
    print("   • Dynamic analysis: sensor_fusion_output/sequence_analysis/105_seq-53/")
    print("\n🔍 Check the output folders for:")
    print("   • Camera images with LIDAR projections")
    print("   • 3D visualizations of static vs dynamic points")
    print("   • Statistical analysis plots")
    print("   • Point cloud data files (.npy and .pcd)")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)