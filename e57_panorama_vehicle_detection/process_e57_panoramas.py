import os
import sys
from extract_panorama import extract_panorama_from_e57
from detect_vehicles import detect_vehicles_on_panorama

def run_complete_pipeline():
    """
    Runs complete vehicle detection pipeline:
    1. Extract panoramas from E57 files
    2. Detect vehicles on extracted panoramas
    """
    
    print("="*60)
    print("E57 PANORAMA VEHICLE DETECTION PIPELINE")
    print("="*60)
    
    # Step 1: Extract panoramas
    print("\nSTEP 1: Extracting panoramas from E57 files...")
    print("-"*60)
    try:
        extract_panorama_from_e57()
    except Exception as e:
        print(f"Error during panorama extraction: {e}")
        return
    
    # Step 2: Detect vehicles
    print("\n" + "="*60)
    print("STEP 2: Detecting vehicles on panoramas...")
    print("-"*60)
    try:
        detect_vehicles_on_panorama()
    except Exception as e:
        print(f"Error during vehicle detection: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)
    print("\nResults saved in:")
    script_dir = os.path.dirname(__file__)
    print(f"  - Panoramas: {os.path.join(script_dir, 'data', 'panoramas')}")
    print(f"  - Detected: {os.path.join(script_dir, 'data', 'detected')}")

if __name__ == "__main__":
    run_complete_pipeline()
