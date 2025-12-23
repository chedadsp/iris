import os
import glob
import pye57
from datetime import datetime

def extract_panorama_from_e57():
    # Find E57 files in data/input folder
    script_dir = os.path.dirname(__file__)
    input_dir = os.path.join(script_dir, "data", "input")
    output_dir = os.path.join(script_dir, "data", "panoramas")
    
    if not os.path.exists(input_dir):
        print(f"Error: Folder {input_dir} does not exist!")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    e57_files = glob.glob(os.path.join(input_dir, "*.e57"))
    
    if not e57_files:
        print(f"Error: No E57 files found in folder {input_dir}!")
        return
    
    print(f"Found {len(e57_files)} E57 file(s)")
    
    # Iterate through all E57 files
    for e57_path in e57_files:
        print(f"\nProcessing: {os.path.basename(e57_path)}")
        
        # Get filename without extension
        e57_basename = os.path.splitext(os.path.basename(e57_path))[0]
        
        # Load E57 file
        e57 = pye57.E57(e57_path)
        
        # Access root structure
        root = e57.root
        
        # Get images2D node
        images2d = root["images2D"]
        print(f"  Images found: {len(images2d)}")
        
        # Iterate through all images
        for j in range(len(images2d)):
            image = images2d[j]
            
            # DEBUG: List all available keys in image
            print(f"  Image #{j} available keys:")
            try:
                for k in range(image.childCount()):
                    child = image.get(k)
                    print(f"    - {child.elementName()}")
            except Exception as debug_ex:
                print(f"    Could not list keys: {debug_ex}")
            
            # Try different representations
            success = False
            for key in ["visualReferenceRepresentation", "sphericalRepresentation", 
                        "pinholeRepresentation", "cylindricalRepresentation"]:
                try:
                    print(f"  Trying {key}...", end='')
                    rep = image[key]
                    print(" found!")
                    
                    # Find jpeg/png blob (try different names)
                    blob = None
                    for blob_name in ["jpegImage", "pngImage", "imageFile", "blobImage"]:
                        try:
                            blob = rep[blob_name]
                            print(f"    Using {blob_name}")
                            break
                        except:
                            continue
                    
                    if blob is None:
                        print(f"    No image blob found in {key}")
                        continue
                    
                    # Read data directly from BlobNode
                    blob_size = blob.byteCount()
                    
                    # Create buffer and read data
                    blob_data = bytearray(blob_size)
                    blob.read(blob_data, 0, blob_size)
                    
                    # Save as JPEG with e57 filename + _panorama
                    output_path = os.path.join(output_dir, f"{e57_basename}_panorama.jpg")
                    with open(output_path, 'wb') as out:
                        out.write(blob_data)
                    
                    print(f"  ✓ Saved: {os.path.basename(output_path)}")
                    
                    # Try to verify with PIL
                    try:
                        from PIL import Image
                        Image.MAX_IMAGE_PIXELS = None
                        img = Image.open(output_path)
                        print(f"    Dimensions: {img.width} x {img.height} | Mode: {img.mode}")
                    except Exception as pil_ex:
                        print(f"    PIL error: {pil_ex}")
                    
                    success = True
                    break
                    
                except KeyError:
                    print(" not found")
                    continue
                except Exception as ex:
                    print(f" error: {ex}")
                    continue
            
            if not success:
                print(f"  ✗ Could not extract panorama from image #{j}")

if __name__ == "__main__":
    extract_panorama_from_e57()
