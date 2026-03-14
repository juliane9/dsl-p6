from pathlib import Path

import cv2
import numpy as np
import pandas as pd


BASE_MASKS_ROOT = Path("images_and_masks/maize_2")
GROUND_TRUTH_SUFFIX = "_mask_ground_truth.png"
YOLOSAM_SUFFIX = "_000.png"
REPORT_PATTERN = "mask_quality_report_{frame_name}.txt"

METRIC_COLUMNS = [
    "IoU",
    "Dice",
    "Precision",
    "Recall",
    "F1",
    "AP_Foreground",
    "AP_Background",
    "mAP",
]

METHOD_LABELS = {
    "grounded_sam_vit_b": "GroundedSAM vit_b",
    "grounded_sam_vit_l": "GroundedSAM vit_l",
    "grounded_sam_vit_h": "GroundedSAM vit_h",
    "colors": "Basic Color",
    "sam3": "SAM 3",
    "yolosam": "YOLOSAM",
}


def load_mask(mask_path: Path):
    """Load mask image and convert to binary values (0 or 1)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.uint8)


def calculate_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def calculate_dice(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    intersection = np.logical_and(mask_true, mask_pred).sum()
    total = mask_true.sum() + mask_pred.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AP for a binary class with binary scores (0/1) efficiently."""
    y_true = y_true.astype(np.uint8).ravel()
    y_score = (y_score > 0).astype(np.uint8).ravel()

    positives = int(y_true.sum())
    total = int(y_true.size)
    negatives = total - positives
    if positives == 0:
        return 1.0

    tp_high = int(np.sum((y_true == 1) & (y_score == 1)))
    fp_high = int(np.sum((y_true == 0) & (y_score == 1)))

    recall_high = tp_high / positives
    precision_high = tp_high / (tp_high + fp_high) if (tp_high + fp_high) > 0 else 1.0
    precision_low = positives / total if total > 0 else 1.0

    # For binary scores, the PR curve has only two recall jumps: at score=1 and score=0.
    precision_high_envelope = max(precision_high, precision_low)
    ap = (recall_high * precision_high_envelope) + ((1.0 - recall_high) * precision_low)
    return float(ap)


def calculate_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict:
    tp = int(np.sum((prediction == 1) & (ground_truth == 1)))
    fp = int(np.sum((prediction == 1) & (ground_truth == 0)))
    fn = int(np.sum((prediction == 0) & (ground_truth == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    ap_foreground = average_precision_binary(ground_truth, prediction)
    ap_background = average_precision_binary(1 - ground_truth, 1 - prediction)
    mean_ap = 0.5 * (ap_foreground + ap_background)

    return {
        "IoU": calculate_iou(ground_truth, prediction),
        "Dice": calculate_dice(ground_truth, prediction),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1_score),
        "AP_Foreground": float(ap_foreground),
        "AP_Background": float(ap_background),
        "mAP": float(mean_ap),
    }


def method_label(method_key: str) -> str:
    if method_key in METHOD_LABELS:
        return METHOD_LABELS[method_key]
    return method_key.replace("_", " ").title()


def extract_image_id_from_gt(gt_path: Path) -> str:
    return gt_path.name[: -len(GROUND_TRUTH_SUFFIX)]


def discover_frame_mask_dirs(base_dir: Path):
    return sorted(base_dir.glob("frame_*_masks"))


def discover_methods(frame_masks_dir: Path) -> dict:
    methods = {}

    for mask_path in sorted(frame_masks_dir.glob("*_mask_*.png")):
        name = mask_path.name
        if name.endswith(GROUND_TRUTH_SUFFIX):
            continue
        if name.endswith("_probability.png"):
            continue

        marker = "_mask_"
        idx = name.rfind(marker)
        if idx == -1:
            continue

        method_key = name[idx + len(marker) : -4]
        suffix = name[idx:]
        methods[method_key] = {
            "source": "frame_masks",
            "suffix": suffix,
        }

    frame_name = frame_masks_dir.name.replace("_masks", "")
    yolosam_masks_dir = frame_masks_dir.parent / f"{frame_name}_yolosam" / "masks"
    if yolosam_masks_dir.exists() and any(yolosam_masks_dir.glob(f"*{YOLOSAM_SUFFIX}")):
        methods["yolosam"] = {
            "source": "yolosam",
            "suffix": YOLOSAM_SUFFIX,
            "dir": yolosam_masks_dir,
        }

    return methods


def evaluate_method(
    method_key: str,
    method_spec: dict,
    image_ids: list,
    gt_masks: dict,
    frame_masks_dir: Path,
):
    rows = []
    missing = 0
    load_fail = 0
    shape_mismatch = 0

    for image_id in image_ids:
        if method_spec["source"] == "frame_masks":
            mask_path = frame_masks_dir / f"{image_id}{method_spec['suffix']}"
        else:
            mask_path = method_spec["dir"] / f"{image_id}{method_spec['suffix']}"

        if not mask_path.exists():
            missing += 1
            continue

        pred_mask = load_mask(mask_path)
        if pred_mask is None:
            load_fail += 1
            continue

        gt_mask = gt_masks[image_id]
        if gt_mask.shape != pred_mask.shape:
            shape_mismatch += 1
            continue

        metrics = calculate_metrics(gt_mask, pred_mask)
        row = {
            "Image": image_id,
            "MethodKey": method_key,
            "Method": method_label(method_key),
        }
        row.update(metrics)
        rows.append(row)

    stats = {
        "evaluated": len(rows),
        "missing": missing,
        "load_fail": load_fail,
        "shape_mismatch": shape_mismatch,
    }
    return rows, stats


def build_metric_table(mean_or_std_df: pd.DataFrame) -> pd.DataFrame:
    table = mean_or_std_df[METRIC_COLUMNS].transpose().reset_index()
    table = table.rename(columns={"index": "Metric"})
    return table


def write_frame_report(frame_name: str, df: pd.DataFrame, method_stats: dict) -> Path:
    mean_df = df.groupby("Method")[METRIC_COLUMNS].mean().sort_index()
    std_df = df.groupby("Method")[METRIC_COLUMNS].std().fillna(0.0).sort_index()

    summary_table = build_metric_table(mean_df)
    std_table = build_metric_table(std_df)

    lines = []
    lines.append("=" * 120)
    lines.append("MASK QUALITY EVALUATION REPORT")
    lines.append(f"Frame: {frame_name}")
    lines.append(f"Images evaluated: {df['Image'].nunique()}")
    lines.append("=" * 120)
    lines.append("")

    lines.append("[METHOD AVAILABILITY]")
    for method_key in sorted(method_stats.keys(), key=method_label):
        stats = method_stats[method_key]
        lines.append(
            f"- {method_label(method_key)}: evaluated={stats['evaluated']}, "
            f"missing={stats['missing']}, load_fail={stats['load_fail']}, shape_mismatch={stats['shape_mismatch']}"
        )
    lines.append("")

    lines.append("[SUMMARY STATISTICS - MEAN]")
    lines.append(summary_table.to_string(index=False))
    lines.append("")

    lines.append("[STANDARD DEVIATIONS]")
    lines.append(std_table.to_string(index=False))
    lines.append("")

    lines.append("[KEY FINDINGS]")
    for metric in METRIC_COLUMNS:
        best_method = mean_df[metric].idxmax()
        best_score = mean_df[metric].max()
        lines.append(f"[+] {metric}: {best_method} achieves best score ({best_score:.4f})")
    lines.append("")

    lines.append("[METRIC DEFINITIONS]")
    lines.append("- IoU: Intersection over Union")
    lines.append("- Dice: 2 * Intersection / (Sum of masks)")
    lines.append("- Precision: TP / (TP + FP)")
    lines.append("- Recall: TP / (TP + FN)")
    lines.append("- F1: Harmonic mean of Precision and Recall")
    lines.append("- AP_Foreground: Average Precision for foreground class")
    lines.append("- AP_Background: Average Precision for background class")
    lines.append("- mAP: (AP_Foreground + AP_Background) / 2")
    lines.append("")

    lines.append("[PER-IMAGE RESULTS]")
    lines.append(df.to_string(index=False))

    report_path = Path(REPORT_PATTERN.format(frame_name=frame_name))
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def process_frame(frame_masks_dir: Path) -> bool:
    frame_name = frame_masks_dir.name.replace("_masks", "")

    gt_paths = sorted(frame_masks_dir.glob(f"*{GROUND_TRUTH_SUFFIX}"))
    if not gt_paths:
        print(f"[WARN] No ground truth masks found for {frame_name}: {frame_masks_dir}")
        return False

    gt_masks = {}
    image_ids = []
    for gt_path in gt_paths:
        image_id = extract_image_id_from_gt(gt_path)
        gt_mask = load_mask(gt_path)
        if gt_mask is None:
            print(f"[WARN] Could not load ground truth: {gt_path.name}")
            continue
        gt_masks[image_id] = gt_mask
        image_ids.append(image_id)

    if not image_ids:
        print(f"[WARN] No valid ground truth masks loaded for {frame_name}")
        return False

    methods = discover_methods(frame_masks_dir)
    if not methods:
        print(f"[WARN] No computed methods found for {frame_name}")
        return False

    all_rows = []
    method_stats = {}

    print(f"\n[INFO] Processing {frame_name}")
    print(f"[INFO] Ground truth images: {len(image_ids)}")
    print(f"[INFO] Discovered methods: {', '.join(method_label(k) for k in sorted(methods.keys(), key=method_label))}")

    for method_key in sorted(methods.keys(), key=method_label):
        rows, stats = evaluate_method(method_key, methods[method_key], image_ids, gt_masks, frame_masks_dir)
        method_stats[method_key] = stats
        all_rows.extend(rows)

    if not all_rows:
        print(f"[WARN] No valid method predictions evaluated for {frame_name}")
        return False

    df = pd.DataFrame(all_rows)
    report_path = write_frame_report(frame_name, df, method_stats)
    print(f"[OK] Report saved: {report_path}")
    return True


def main():
    frame_dirs = discover_frame_mask_dirs(BASE_MASKS_ROOT)
    if not frame_dirs:
        print(f"[ERROR] No frame mask folders found under: {BASE_MASKS_ROOT}")
        return

    print(f"Found {len(frame_dirs)} frame mask folders")

    processed = 0
    for frame_dir in frame_dirs:
        if process_frame(frame_dir):
            processed += 1

    if processed == 0:
        print("[ERROR] No frame could be evaluated")
        return

    print(f"\n[DONE] Evaluated {processed}/{len(frame_dirs)} frame folders")


if __name__ == "__main__":
    main()
