from pathlib import Path
import cv2
import numpy as np
import torch

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor


# -----------------------
# Settings you may tweak
# -----------------------
FRAME_TO_SEGMENT = "frame_1180"

INPUT_DIR = Path("images_and_masks/maize_2/" + FRAME_TO_SEGMENT +"/images")     # folder with your 21 images
OUTPUT_DIR = Path("images_and_masks/maize_2/")    # where masks will be saved

# Weights / configs
GDINO_CONFIG = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = Path("weights/groundingdino_swint_ogc.pth")

# SAM model selection: choose from "vit_b", "vit_l", "vit_h"
# basic, large, huge
SAM_TYPE = "vit_l"

# Optional manual checkpoint override. Set to None to use defaults below.
SAM_CKPT_OVERRIDE = None

# Default checkpoints for each model size.
SAM_CKPT_BY_TYPE = {
    "vit_b": Path("weights/sam_vit_b_01ec64.pth"),
    "vit_l": Path("weights/sam_vit_l_0b3195.pth"),
    "vit_h": Path("weights/sam_vit_h_4b8939.pth"),
}

# Text prompt for GroundingDINO
# TEXT_PROMPT = "plant, leaf, foliage"
TEXT_PROMPT = "green leaves"

# Thresholds (lower if it finds nothing; raise if too many false detections)
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

# Save overlay previews (mask drawn over image)
SAVE_COMPRESSED_OVERLAYS = True

# Save probability masks (with sigmoid applied)
SAVE_PROBABILITY_MASKS = True


def get_sam_checkpoint(sam_type: str, ckpt_override):
    """Resolve selected SAM checkpoint and output-name token from model type."""
    if sam_type not in SAM_CKPT_BY_TYPE:
        supported = ", ".join(SAM_CKPT_BY_TYPE.keys())
        raise ValueError(f"Unsupported SAM_TYPE '{sam_type}'. Supported values: {supported}")

    checkpoint = Path(ckpt_override) if ckpt_override else SAM_CKPT_BY_TYPE[sam_type]
    name_token = f"grounded_sam_{sam_type}"
    return checkpoint, name_token


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


def save_overlay(rgb: np.ndarray, mask01: np.ndarray, out_path: Path, box_xyxy: np.ndarray = None):
    """
    Simple overlay: red mask blended onto original.
    Saves as heavily compressed JPEG (~10% of original size).
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()

    # Put mask into red channel for contrast against green plants.
    overlay[mask01, 2] = 255  # red channel

    blended = cv2.addWeighted(bgr, 0.70, overlay, 0.30, 0)

    # Optionally draw detection box for quick visual debugging.
    if box_xyxy is not None:
        x0, y0, x1, y1 = [int(v) for v in box_xyxy]
        cv2.rectangle(blended, (x0, y0), (x1, y1), (0, 0, 255), 2)

    # JPEG quality: 50 gives ~10% of original size (good for visual inspection only)
    cv2.imwrite(str(out_path), blended, [cv2.IMWRITE_JPEG_QUALITY, 50])


def save_probability_mask(probs: np.ndarray, out_path: Path):
    """
    probs: float array HxW with values in [0, 1] after sigmoid
    Saves as 0..255 PNG (probability map, where 0=no object, 255=high confidence object)
    """
    # Convert probability [0,1] to uint8 [0,255]
    prob_uint8 = (probs * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), prob_uint8)


def save_overlay_probability(rgb: np.ndarray, probs: np.ndarray, out_path: Path, box_xyxy: np.ndarray = None):
    """
    Probability overlay: grayscale probability map blended onto original.
    Lighter areas = higher confidence. Saves as compressed JPEG.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Create grayscale probability visualization (0=black, 1=white)
    prob_uint8 = (probs * 255).astype(np.uint8)
    prob_bgr = cv2.cvtColor(prob_uint8, cv2.COLOR_GRAY2BGR)
    
    # Blend probability map with original image
    blended = cv2.addWeighted(bgr, 0.70, prob_bgr, 0.30, 0)

    # Optionally draw detection box for quick visual debugging.
    if box_xyxy is not None:
        x0, y0, x1, y1 = [int(v) for v in box_xyxy]
        cv2.rectangle(blended, (x0, y0), (x1, y1), (0, 0, 255), 2)

    cv2.imwrite(str(out_path), blended, [cv2.IMWRITE_JPEG_QUALITY, 50])


def main():
    device = "cpu"
    torch.set_grad_enabled(False)

    sam_ckpt, sam_name_token = get_sam_checkpoint(SAM_TYPE, SAM_CKPT_OVERRIDE)

    ensure_exists(INPUT_DIR, "Input folder")
    ensure_exists(Path(GDINO_CONFIG), "GroundingDINO config")
    ensure_exists(GDINO_CKPT, "GroundingDINO checkpoint")
    ensure_exists(sam_ckpt, "SAM checkpoint")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_dir = OUTPUT_DIR / (FRAME_TO_SEGMENT + "_masks")
    overlay_dir = OUTPUT_DIR / (FRAME_TO_SEGMENT + "_masks_overlays_compressed")
    mask_dir.mkdir(exist_ok=True)
    if SAVE_COMPRESSED_OVERLAYS:
        overlay_dir.mkdir(exist_ok=True)

    # Load models once
    print("Loading GroundingDINO (CPU)...")
    gdino = load_model(GDINO_CONFIG, str(GDINO_CKPT), device=device)

    print(f"Loading SAM (CPU) | type={SAM_TYPE} | checkpoint={sam_ckpt.name}")
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(sam_ckpt))
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
        best_box_xyxy_np = best_box_xyxy[0].detach().cpu().numpy()

        # Keep box in the same coordinate frame as image_source for overlays.
        box_on_overlay = np.array([
            np.clip(best_box_xyxy_np[0], 0, w - 1),
            np.clip(best_box_xyxy_np[1], 0, h - 1),
            np.clip(best_box_xyxy_np[2], 0, w - 1),
            np.clip(best_box_xyxy_np[3], 0, h - 1),
        ], dtype=np.float32)

        # Run SAM with that box
        predictor.set_image(image_source)  # expects RGB uint8
        sam_box = predictor.transform.apply_boxes_torch(best_box_xyxy, image_source.shape[:2]).to(device)

        masks, _, low_res_logits = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=sam_box,
            multimask_output=False,
        )

        # masks: (1,1,H,W) boolean - SAM's original thresholded output (keep as is)
        mask01 = masks[0, 0].detach().cpu().numpy().astype(bool)
        
        # Build probability map with SAM's own postprocessing path.
        # Upsample logits first, then apply sigmoid to preserve edge detail.
        logits_full_res = predictor.model.postprocess_masks(
            low_res_logits,
            input_size=predictor.input_size,
            original_size=predictor.original_size,
        )
        mask_probs = torch.sigmoid(logits_full_res)[0, 0].detach().cpu().numpy()

        # Save outputs
        out_mask = mask_dir / f"{img_path.stem}_mask_{sam_name_token}.png"
        save_mask_png(mask01, out_mask)
        print(f"  - Saved mask: {out_mask.name}")

        if SAVE_PROBABILITY_MASKS:
            out_prob_mask = mask_dir / f"{img_path.stem}_mask_{sam_name_token}_probability.png"
            save_probability_mask(mask_probs, out_prob_mask)
            print(f"  - Saved probability mask: {out_prob_mask.name}")

        if SAVE_COMPRESSED_OVERLAYS:
            out_ov = overlay_dir / f"{img_path.stem}_mask_{sam_name_token}_overlay_compressed.jpg"
            save_overlay(image_source, mask01, out_ov, box_on_overlay)
            print(f"  - Saved overlay: {out_ov.name}")
            
            if SAVE_PROBABILITY_MASKS:
                out_prob_ov = overlay_dir / f"{img_path.stem}_mask_{sam_name_token}_probability_overlay_compressed.jpg"
                save_overlay_probability(image_source, mask_probs, out_prob_ov, box_on_overlay)
                print(f"  - Saved probability overlay: {out_prob_ov.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()