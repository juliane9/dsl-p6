from pathlib import Path
import os
import cv2
import numpy as np


# -----------------------
# Settings you may tweak
# -----------------------
FRAME_TO_SEGMENT = "frame_1180"

INPUT_DIR = Path("images_and_masks/maize_2/" + FRAME_TO_SEGMENT + "/images")
OUTPUT_MASKS_DIR = Path("images_and_masks/maize_2/" + FRAME_TO_SEGMENT + "_masks")
OUTPUT_OVERLAYS_DIR = Path("images_and_masks/maize_2/" + FRAME_TO_SEGMENT + "_masks_overlays_compressed")

# HSV color range for green plants (you can adjust these)
# Lower bound for green
LOWER_GREEN = np.array([25, 30, 30])
# Upper bound for green
UPPER_GREEN = np.array([90, 255, 255])

# Morphological operations kernel size
KERNEL_SIZE = 5
ITERATIONS = 2

# JPEG compression quality for overlays
OVERLAY_QUALITY = 75


def create_basic_mask(image_path):
    """
    Create mask using basic image processing techniques.
    Uses HSV color space to detect green vegetation.
    
    Args:
        image_path: Path to input image
        
    Returns:
        mask: Binary mask of detected vegetation
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ Failed to read image: {image_path}")
        return None
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for green colors
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=ITERATIONS)
    
    # Open to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=ITERATIONS // 2)
    
    # Dilate to connect nearby components
    mask = cv2.dilate(mask, kernel, iterations=ITERATIONS // 2)
    
    return mask, image


def save_mask(mask, output_path):
    """Save mask as PNG image."""
    cv2.imwrite(str(output_path), mask)


def create_overlay(image, mask, output_path, quality=OVERLAY_QUALITY):
    """
    Create and save overlay image showing mask on original image.
    
    Args:
        image: Original RGB image
        mask: Binary mask
        output_path: Path to save overlay
        quality: JPEG compression quality
    """
    # Create colored overlay
    overlay = image.copy()
    
    # Convert mask to 3-channel
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply green color to masked regions
    overlay[mask > 0] = [0, 255, 0]  # Green in BGR
    
    # Blend original image with overlay
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Save as compressed JPEG
    cv2.imwrite(str(output_path), blended, 
                [cv2.IMWRITE_JPEG_QUALITY, quality])


def main():
    """Main processing function."""
    
    # Create output directories if they don't exist
    OUTPUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_files = sorted([f for f in INPUT_DIR.glob("*.jpg") if f.is_file()])
    
    if not image_files:
        image_files = sorted([f for f in INPUT_DIR.glob("*.png") if f.is_file()])
    
    if not image_files:
        print(f"❌ No images found in: {INPUT_DIR}")
        return
    
    print(f"Found {len(image_files)} images in: {INPUT_DIR}")
    print(f"Saving outputs to: {OUTPUT_MASKS_DIR.parent}\n")
    
    processed = 0
    
    for image_path in image_files:
        image_name = image_path.stem
        print(f"Processing: {image_path.name}")
        
        # Create mask using basic techniques
        result = create_basic_mask(image_path)
        if result is None:
            continue
        
        mask, image = result
        
        # Get number of pixels in mask (to estimate detection)
        mask_pixels = np.sum(mask > 0)
        percentage = (mask_pixels / (mask.shape[0] * mask.shape[1])) * 100
        
        # Save mask (PNG)
        mask_output_path = OUTPUT_MASKS_DIR / f"{image_name}_mask_colors.png"
        save_mask(mask, mask_output_path)
        
        # Create and save overlay (Compressed JPG)
        overlay_output_path = OUTPUT_OVERLAYS_DIR / f"{image_name}_mask_colors_overlay_compressed.jpg"
        create_overlay(image, mask, overlay_output_path)
        
        print(f"  - Detected vegetation: {percentage:.1f}% of image")
        print(f"  - Saved mask: {mask_output_path.name}")
        print(f"  - Saved overlay: {overlay_output_path.name}")
        
        processed += 1
    
    print(f"\n✅ Successfully processed {processed}/{len(image_files)} images")


if __name__ == "__main__":
    main()
