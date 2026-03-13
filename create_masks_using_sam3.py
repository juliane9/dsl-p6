"""
Create masks with Meta SAM 3 (Segment Anything with Concepts).

Unlike create_masks_using_grounded_sam.py, this script does NOT need
GroundingDINO  SAM 3 has built-in open-vocabulary text prompting.

Install (Miniconda base, run once):
  pip install --no-deps git+https://github.com/facebookresearch/sam3.git
  pip install einops setuptools==75.8.2

Checkpoint (auto-downloaded from HuggingFace on first run):
  You need to authenticate first:
    pip install huggingface_hub
    huggingface-cli login           # paste your HF access token
  Then request access at: https://huggingface.co/facebook/sam3

Requirements:
  - Python 3.12+, PyTorch 2.7+, CUDA 12.6+ (GPU strongly recommended)
  - On CPU the model will be very slow.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# -----------------------
# Settings you may tweak
# -----------------------
FRAME_TO_SEGMENT = "frame_1180"

INPUT_DIR = Path("images_and_masks/maize_2/" + FRAME_TO_SEGMENT + "/images")
OUTPUT_DIR = Path("images_and_masks/maize_2/")

# Optional: path to a local SAM3 checkpoint (.pt file).
# Set to None to auto-download from HuggingFace (requires hf auth + access request).
SAM3_CKPT = None  # e.g. Path("weights/sam3.pt")

# Output file name token
SAM3_NAME_TOKEN = "sam3"

# Text prompt  SAM 3 handles open-vocabulary text directly (no DINO needed).
TEXT_PROMPT = "green maize leaf"

# Confidence threshold: only keep detections above this score (0-1).
CONFIDENCE_THRESHOLD = 0.5

# Save colour overlay previews (mask drawn over image)
SAVE_COMPRESSED_OVERLAYS = True


# ------------------------------------------------
# Helper functions
# ------------------------------------------------

def save_mask_png(mask01, out_path):
    cv2.imwrite(str(out_path), (mask01.astype("uint8") * 255))


def save_overlay(rgb, mask01, out_path, box_xyxy=None):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[mask01, 2] = 255
    blended = cv2.addWeighted(bgr, 0.70, overlay, 0.30, 0)
    if box_xyxy is not None:
        x0, y0, x1, y1 = [int(v) for v in box_xyxy]
        cv2.rectangle(blended, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(str(out_path), blended, [cv2.IMWRITE_JPEG_QUALITY, 50])


def load_sam3_processor(device):
    """Build the SAM3 image model and processor (checkpoint auto-downloaded from HF)."""
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise ImportError(
            "Could not import sam3. Install it with:\n"
            "  pip install --no-deps git+https://github.com/facebookresearch/sam3.git\n"
            "  pip install einops setuptools==75.8.2\n"
            "Then authenticate with HuggingFace:\n"
            "  huggingface-cli login\n"
            "And request checkpoint access at https://huggingface.co/facebook/sam3"
        ) from exc

    checkpoint_path = str(SAM3_CKPT) if SAM3_CKPT is not None else None
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=(checkpoint_path is None),
    )
    return Sam3Processor(model, device=device, confidence_threshold=CONFIDENCE_THRESHOLD)


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA GPU detected. SAM3 will be very slow on CPU.")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_dir = OUTPUT_DIR / (FRAME_TO_SEGMENT + "_masks")
    overlay_dir = OUTPUT_DIR / (FRAME_TO_SEGMENT + "_masks_overlays_compressed")
    mask_dir.mkdir(exist_ok=True)
    if SAVE_COMPRESSED_OVERLAYS:
        overlay_dir.mkdir(exist_ok=True)

    print(f"Loading SAM3 model ({device})...")
    processor = load_sam3_processor(device=device)
    print(f"  Text prompt: '{TEXT_PROMPT}'")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts)

    if not images:
        print(f"No images found in: {INPUT_DIR.resolve()}")
        return

    print(f"Found {len(images)} images in: {INPUT_DIR.resolve()}")
    print(f"Saving outputs to: {OUTPUT_DIR.resolve()}")

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        pil_image = Image.open(img_path).convert("RGB")

        # SAM3 image inference: embed image once, then apply text prompt
        state = processor.set_image(pil_image)
        state = processor.set_text_prompt(prompt=TEXT_PROMPT, state=state)

        masks = state.get("masks")    # bool tensor [N, 1, H, W]
        boxes = state.get("boxes")    # pixel xyxy  [N, 4]
        scores = state.get("scores")  # [N]

        if masks is None or len(masks) == 0:
            print(
                f"  - No detections above threshold {CONFIDENCE_THRESHOLD}. "
                "Try lowering CONFIDENCE_THRESHOLD or adjusting TEXT_PROMPT."
            )
            continue

        print(f"  - {len(masks)} detection(s) found")
        image_np = np.array(pil_image)  # RGB uint8 for overlays

        for idx in range(len(masks)):
            score = float(scores[idx]) if scores is not None else float("nan")
            print(f"  - Detection {idx}: score={score:.3f}")

            # masks shape: [N, 1, H, W]
            mask01 = masks[idx, 0].cpu().numpy().astype(bool)

            out_mask = mask_dir / f"{img_path.stem}_mask_{SAM3_NAME_TOKEN}_{idx:02d}.png"
            save_mask_png(mask01, out_mask)
            print(f"    Saved mask: {out_mask.name}")

            if SAVE_COMPRESSED_OVERLAYS:
                box_np = boxes[idx].cpu().numpy() if boxes is not None else None
                out_overlay = (
                    overlay_dir
                    / f"{img_path.stem}_mask_{SAM3_NAME_TOKEN}_{idx:02d}_overlay_compressed.jpg"
                )
                save_overlay(image_np, mask01, out_overlay, box_np)
                print(f"    Saved overlay: {out_overlay.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
