from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import re


# -----------------------
# Settings
# -----------------------
BASE_DIR = Path("images_and_masks")


def parse_polygon_points(points_str: str) -> np.ndarray:
    """
    Parse polygon points from CVAT XML format: "x1,y1;x2,y2;x3,y3;..."
    Returns numpy array of shape (N, 2) with coordinates.
    """
    points = []
    for point_str in points_str.split(';'):
        x_str, y_str = point_str.split(',')
        points.append([float(x_str), float(y_str)])
    return np.array(points, dtype=np.int32)


def create_mask_from_polygon(width: int, height: int, points: np.ndarray) -> np.ndarray:
    """
    Create a binary mask from polygon points.
    Returns a binary mask (0 or 255) of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    # Reshape points for cv2.fillPoly: needs shape (1, N, 2)
    points_reshaped = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points_reshaped], 255)
    return mask


def save_overlay(rgb: np.ndarray, mask01: np.ndarray, out_path: Path):
    """
    Simple overlay: green-ish mask blended onto original.
    Saves as heavily compressed JPEG (~10% of original size).
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()

    # Put mask into green channel
    overlay[mask01 > 0, 1] = 255  # green channel

    blended = cv2.addWeighted(bgr, 0.70, overlay, 0.30, 0)
    # JPEG quality: 50 gives ~10% of original size (good for visual inspection only)
    cv2.imwrite(str(out_path), blended, [cv2.IMWRITE_JPEG_QUALITY, 50])


def process_xml_file(xml_path: Path, output_dir: Path, frame_images_dir: Path, save_overlays: bool = True):
    """
    Process a CVAT XML annotation file and create masks for each image.
    Handles XML files with minor syntax issues.
    """
    print(f"\nProcessing XML: {xml_path.name}")
    
    try:
        # Read and attempt to fix the XML
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Parse XML
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"  ⚠ XML parsing error: {e}")
        print(f"  Attempting to fix and parse...")
        
        # Try to fix common issues: add missing closing tags
        try:
            # Count opening and closing image tags
            open_tags = xml_content.count('<image ')
            close_tags = xml_content.count('</image>')
            
            if open_tags > close_tags:
                print(f"  Found {open_tags} opening <image> tags but only {close_tags} closing tags")
                print(f"  Adding {open_tags - close_tags} missing closing tag(s)...")
                
                # Find the last polygon closing tag before </annotations>
                last_polygon_end = xml_content.rfind('</polygon>')
                annotations_end = xml_content.rfind('</annotations>')
                
                if last_polygon_end > 0 and annotations_end > last_polygon_end:
                    # Insert missing </image> tags
                    missing_tags = '\n  </image>' * (open_tags - close_tags)
                    xml_content = (xml_content[:last_polygon_end + len('</polygon>')] + 
                                 missing_tags + 
                                 xml_content[last_polygon_end + len('</polygon>'):])
                    
                    root = ET.fromstring(xml_content)
                    print(f"  ✓ Successfully fixed and parsed XML")
                else:
                    print(f"  ❌ Could not auto-fix. Skipping this file.")
                    return
            else:
                print(f"  ❌ Unknown XML error. Skipping this file.")
                return
        except Exception as fix_error:
            print(f"  ❌ Could not fix XML: {fix_error}")
            return
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_overlays:
        overlay_dir = output_dir.parent / f"{output_dir.name.replace('_masks', '_masks_overlays_compressed')}"
        overlay_dir.mkdir(parents=True, exist_ok=True)
    else:
        overlay_dir = None
    
    # Process each image
    images = root.findall('.//image')
    print(f"Found {len(images)} images with annotations")
    
    for img_elem in images:
        img_name = img_elem.get('name')
        width = int(img_elem.get('width'))
        height = int(img_elem.get('height'))
        
        print(f"  - Processing: {img_name} ({width}x{height})")
        
        # Find all polygons for this image
        polygons = img_elem.findall('polygon')
        
        if not polygons:
            print(f"    ⚠ No polygons found, skipping")
            continue
        
        # Create combined mask for all polygons in this image
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for poly in polygons:
            points_str = poly.get('points')
            label = poly.get('label')
            
            # Parse and create mask from polygon
            points = parse_polygon_points(points_str)
            points_reshaped = points.reshape((-1, 1, 2))
            cv2.fillPoly(combined_mask, [points_reshaped], 255)
            
            print(f"    ✓ Added {label} polygon with {len(points)} points")
        
        # Save mask with ground truth naming convention
        img_stem = Path(img_name).stem
        mask_filename = f"{img_stem}_mask_ground_truth.png"
        mask_path = output_dir / mask_filename
        
        cv2.imwrite(str(mask_path), combined_mask)
        print(f"    ✓ Saved mask: {mask_filename}")
        
        # Create and save overlay if requested
        if save_overlays and overlay_dir:
            # Try to find and load the original image
            original_img_path = frame_images_dir / img_name
            
            if original_img_path.exists():
                # Load original image
                original_img = cv2.imread(str(original_img_path))
                if original_img is not None:
                    # Convert BGR to RGB for overlay function
                    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    
                    # Save overlay
                    overlay_filename = f"{img_stem}_mask_ground_truth_overlay_compressed.jpg"
                    overlay_path = overlay_dir / overlay_filename
                    save_overlay(original_rgb, combined_mask, overlay_path)
                    print(f"    ✓ Saved overlay: {overlay_filename}")
                else:
                    print(f"    ⚠ Could not read image: {img_name}")
            else:
                print(f"    ⚠ Original image not found: {original_img_path}")


def extract_frame_from_filename(filename: str) -> str:
    """
    Extract frame identifier from filename like 'cvat_annotations_frame_0001.xml'
    Returns frame identifier like 'frame_0001' or None if not found.
    """
    match = re.search(r'(frame_\d+)', filename)
    if match:
        return match.group(1)
    return None


def main():
    """
    Scan for CVAT annotation XML files and process them.
    """
    print("=== Ground Truth Mask Generator ===")
    print(f"Scanning: {BASE_DIR.resolve()}\n")
    
    # Find all CVAT annotation XML files
    xml_files = list(BASE_DIR.glob("**/cvat_annotations*.xml"))
    
    if not xml_files:
        print("❌ No CVAT annotation XML files found!")
        print("   Expected pattern: **/cvat_annotations*.xml")
        return
    
    print(f"Found {len(xml_files)} XML file(s):\n")
    
    for xml_file in xml_files:
        print(f"📁 {xml_file.relative_to(BASE_DIR)}")
        
        # Extract frame identifier from filename
        frame_id = extract_frame_from_filename(xml_file.name)
        
        if not frame_id:
            print(f"   ⚠ Could not extract frame ID from filename: {xml_file.name}")
            print(f"   Expected format: cvat_annotations_frame_XXXX.xml\n")
            continue
        
        # Determine output directory: same parent folder, but in frame_XXXX_masks subfolder
        output_dir = xml_file.parent / f"{frame_id}_masks"
        
        # Determine where original images are located
        frame_images_dir = xml_file.parent / frame_id / "images"
        
        if not frame_images_dir.exists():
            print(f"   ⚠ Image folder not found: {frame_images_dir.relative_to(BASE_DIR)}")
            print(f"   Skipping overlays for this frame\n")
            frame_images_dir = None
        
        print(f"   Frame: {frame_id}")
        print(f"   Output masks: {output_dir.relative_to(BASE_DIR)}")
        if frame_images_dir:
            print(f"   Images from: {frame_images_dir.relative_to(BASE_DIR)}")
        
        # Process the XML file
        process_xml_file(xml_file, output_dir, frame_images_dir, save_overlays=True)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
