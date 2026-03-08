# save as: make_yolosam_from_masks.py
import os, glob
import numpy as np
from PIL import Image
import torch

MASK_DIR = "images_and_masks/maize_2/frame_0001_masks"
OUT_DIR  = "images_and_masks/maize_2/frame_0001_yolosam"  # output here (safe)

os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "bboxes"), exist_ok=True)

mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, "*_mask_grounded_sam_vit_b.png")))
assert len(mask_paths) > 0, f"No masks found in {MASK_DIR}"

def to_binary_mask(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    # If RGB, collapse to grayscale
    if arr.ndim == 3:
        arr = arr[..., 0]  # works if it's white on black; otherwise use max over channels
    bin_mask = (arr > 0).astype(np.uint8)  # 0/1
    return bin_mask

for p in mask_paths:
    base = os.path.basename(p)
    stem = base.replace("_mask_grounded_sam_vit_b.png", "")  # matches image stem
    img = Image.open(p)
    m = to_binary_mask(img)

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        # empty mask -> skip or write empty bbox
        print(f"[WARN] empty mask for {stem}, skipping")
        continue

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # save mask as 0/255 PNG, instance id 000
    out_mask_path = os.path.join(OUT_DIR, "masks", f"{stem}_000.png")
    Image.fromarray((m * 255).astype(np.uint8)).save(out_mask_path)

    # save bbox as torch tensor [x1,y1,x2,y2] (float32)
    bbox = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
    out_bbox_path = os.path.join(OUT_DIR, "bboxes", f"{stem}.pt")
    torch.save(bbox, out_bbox_path)

print("Done.")
print("Wrote:", OUT_DIR)