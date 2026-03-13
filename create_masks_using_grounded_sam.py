from pathlib import Path
import os
import cv2
import numpy as np
import torch

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor


# -----------------------
# Settings you may tweak
# -----------------------
FRAME_TO_SEGMENT = "frame_0001"

INPUT_DIR = Path("images_and_masks/maize_2/" + FRAME_TO_SEGMENT +"/images")     # folder with your 21 images
OUTPUT_DIR = Path("images_and_masks/maize_2/")    # where masks will be saved

# Weights / configs
GDINO_CONFIG = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = Path("weights/groundingdino_swint_ogc.pth")

# CPU-friendly SAM checkpoint is vit_b:
SAM_TYPE = "vit_b"
SAM_CKPT = Path("weights/sam_vit_b_01ec64.pth")  # change if you have a different SAM ckpt

# Text prompt for GroundingDINO
# TEXT_PROMPT = "plant, leaf, foliage"
TEXT_PROMPT = "green leaves"

# Thresholds (lower if it finds nothing; raise if too many false detections)
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

# Save overlay previews (mask drawn over image)
SAVE_COMPRESSED_OVERLAYS = True


def ensure_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path.resolve()}")


def dino_boxes_to_xyxy_pixel(boxes_cxcywh_norm: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """
    GroundingDINO commonly returns boxes as normalized cx,cy,w,h in [0..1].
    Convert to pixel xyxy.
    """
    boxes = boxes_cxcywh_norm.clone()
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x0 = (cx - bw / 2.0) * w
    y0 = (cy - bh / 2.0) * h
    x1 = (cx + bw / 2.0) * w
    y1 = (cy + bh / 2.0) * h

    xyxy = torch.stack([x0, y0, x1, y1], dim=1)
    return xyxy


def pick_best_box(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Pick the highest-score box (best single plant candidate)."""
    best_idx = int(torch.argmax(scores).item())
    return boxes[best_idx:best_idx + 1], scores[best_idx].item()


def save_mask_png(mask01: np.ndarray, out_path: Path):
    """
    mask01: bool or {0,1} array HxW
    Saves as 0..255 PNG (uncompressed for later processing)
    """
    m = (mask01.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), m)


def save_overlay(rgb: np.ndarray, mask01: np.ndarray, out_path: Path):
    """
    Simple overlay: green-ish mask blended onto original.
    Saves as heavily compressed JPEG (~10% of original size).
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()

    # Put mask into green channel
    overlay[mask01, 1] = 255  # green channel

    blended = cv2.addWeighted(bgr, 0.70, overlay, 0.30, 0)
    # JPEG quality: 50 gives ~10% of original size (good for visual inspection only)
    cv2.imwrite(str(out_path), blended, [cv2.IMWRITE_JPEG_QUALITY, 50])


def main():
    device = "cpu"
    torch.set_grad_enabled(False)

    ensure_exists(INPUT_DIR, "Input folder")
    ensure_exists(Path(GDINO_CONFIG), "GroundingDINO config")
    ensure_exists(GDINO_CKPT, "GroundingDINO checkpoint")
    ensure_exists(SAM_CKPT, "SAM checkpoint")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_dir = OUTPUT_DIR / (FRAME_TO_SEGMENT + "_masks")
    overlay_dir = OUTPUT_DIR / (FRAME_TO_SEGMENT + "_masks_overlays_compressed")
    mask_dir.mkdir(exist_ok=True)
    if SAVE_COMPRESSED_OVERLAYS:
        overlay_dir.mkdir(exist_ok=True)

    # Load models once
    print("Loading GroundingDINO (CPU)...")
    gdino = load_model(GDINO_CONFIG, str(GDINO_CKPT), device=device)

    print("Loading SAM (CPU)...")
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT))
    sam.to(device)
    predictor = SamPredictor(sam)

    # Gather images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts]
    images.sort()

    if not images:
        print(f"No images found in: {INPUT_DIR.resolve()}")
        return

    print(f"Found {len(images)} images in: {INPUT_DIR.resolve()}")
    print(f"Saving outputs to: {OUTPUT_DIR.resolve()}")

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        # GroundingDINO load (returns RGB image_source and a preprocessed tensor)
        image_source, image_tensor = load_image(str(img_path))
        h, w = image_source.shape[:2]

        # Predict boxes from text prompt
        boxes, scores, phrases = predict(
            model=gdino,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device,
        )

        if boxes is None or len(boxes) == 0:
            print("  - No boxes found. Try lowering BOX_THRESHOLD/TEXT_THRESHOLD or adjusting TEXT_PROMPT.")
            continue

        # Choose best single detection
        best_box_norm, best_score = pick_best_box(boxes, scores)
        print(f"  - Best detection score: {best_score:.3f} | phrase: {phrases[int(torch.argmax(scores).item())]}")

        # Convert to pixel xyxy
        best_box_xyxy = dino_boxes_to_xyxy_pixel(best_box_norm, w=w, h=h)

        # Run SAM with that box
        predictor.set_image(image_source)  # expects RGB uint8
        sam_box = predictor.transform.apply_boxes_torch(best_box_xyxy, image_source.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=sam_box,
            multimask_output=False,
        )

        # masks: (1,1,H,W) boolean
        mask01 = masks[0, 0].detach().cpu().numpy().astype(bool)

        # Save outputs
        out_mask = mask_dir / f"{img_path.stem}_mask_grounded_sam_vit_b.png"
        save_mask_png(mask01, out_mask)
        print(f"  - Saved mask: {out_mask.name}")

        if SAVE_COMPRESSED_OVERLAYS:
            out_ov = overlay_dir / f"{img_path.stem}_mask_grounded_sam_vit_b_overlay_compressed.jpg"
            save_overlay(image_source, mask01, out_ov)
            print(f"  - Saved overlay: {out_ov.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()