import csv
import os
from collections import Counter
import sys

# Try importing PIL, if not check for cv2
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    try:
        import cv2
    except ImportError:
        print("Error: Neither PIL (Pillow) nor cv2 (opencv-python) could be imported.")
        sys.exit(1)

def get_image_size(path):
    if HAS_PIL:
        with Image.open(path) as img:
            return img.size
    else:
        # cv2 loads as (height, width, channels)
        # We only need shape
        img = cv2.imread(path)
        if img is None:
            raise ValueError("cv2 could not read image")
        h, w = img.shape[:2]
        return (w, h)

def main():
    csv_path = 'gtrain_pairs.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    resolutions = []
    resolution_samples = {}
    
    print(f"Reading {csv_path}...")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check if 'rainy' column exists
            if 'rainy' not in reader.fieldnames: # type: ignore
                print("Error: 'rainy' column not found in csv.")
                print(f"Available columns: {reader.fieldnames}") # type: ignore
                return

            rows = list(reader)
            total_files = len(rows)
            print(f"Found {total_files} entries. Checking resolutions...")

            for i, row in enumerate(rows):
                rainy_path = row['rainy']
                
                # Verify if file exists
                if not os.path.exists(rainy_path):
                    print(f"Warning: File not found: {rainy_path}")
                    continue
                
                try:
                    size = get_image_size(rainy_path)
                    resolutions.append(size)
                    if size not in resolution_samples:
                        resolution_samples[size] = rainy_path
                except Exception as e:
                    print(f"Error reading {rainy_path}: {e}")

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{total_files}...")

    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print("\nResolution Statistics (Width x Height):")
    if not resolutions:
        print("No resolutions found.")
        return

    counts = Counter(resolutions)
    for res, count in counts.most_common():
        sample_path = resolution_samples.get(res, "N/A")
        print(f"{res[0]}x{res[1]}: {count} images. Sample: {sample_path}")

if __name__ == "__main__":
    main()
