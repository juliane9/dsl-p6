from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# -----------------------
# Settings
# -----------------------
MASKS_DIR = Path("images_and_masks/maize_2/frame_0001_masks")
YOLOSAM_MASKS_DIR = Path("images_and_masks/maize_2/frame_0001_yolosam/masks")
OUTPUT_REPORT = Path("mask_quality_report_frame_0001.txt")

# Suffix patterns for different mask types
GROUND_TRUTH_SUFFIX = "_mask_ground_truth.png"
GROUNDED_SAM_B_SUFFIX = "_mask_grounded_sam_vit_b.png"
GROUNDED_SAM_L_SUFFIX = "_mask_grounded_sam_vit_l.png"
GROUNDED_SAM_H_SUFFIX = "_mask_grounded_sam_vit_h.png"
BASIC_SUFFIX = "_mask_colors.png"
YOLOSAM_SUFFIX = "_000.png"


def load_mask(mask_path):
    """Load mask image and convert to binary."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    # Convert to binary (0 or 1)
    mask = (mask > 127).astype(np.uint8)
    return mask


def calculate_iou(mask_true, mask_pred):
    """Calculate Intersection over Union (IoU)."""
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_dice(mask_true, mask_pred):
    """Calculate Dice coefficient (F1 score for binary segmentation)."""
    intersection = np.logical_and(mask_true, mask_pred)
    dice = 2 * np.sum(intersection) / (np.sum(mask_true) + np.sum(mask_pred))
    return dice


def calculate_metrics(ground_truth, prediction):
    """
    Calculate all evaluation metrics.
    
    Args:
        ground_truth: Binary ground truth mask
        prediction: Binary predicted mask
        
    Returns:
        Dictionary with all metrics
    """
    # Calculate metrics using NumPy for better performance
    tp = np.sum((prediction == 1) & (ground_truth == 1))
    tn = np.sum((prediction == 0) & (ground_truth == 0))
    fp = np.sum((prediction == 1) & (ground_truth == 0))
    fn = np.sum((prediction == 0) & (ground_truth == 1))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = {
        'IoU': calculate_iou(ground_truth, prediction),
        'Dice': calculate_dice(ground_truth, prediction),
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'Accuracy': accuracy,
    }
    
    return metrics


def get_image_id(filepath):
    """Extract base image ID from filename."""
    return filepath.stem.replace(GROUND_TRUTH_SUFFIX[:-4], "").replace(GROUNDED_SAM_B_SUFFIX[:-4], "").replace(GROUNDED_SAM_L_SUFFIX[:-4], "").replace(GROUNDED_SAM_H_SUFFIX[:-4], "").replace(BASIC_SUFFIX[:-4], "")


def main():
    """Main analysis function."""
    
    if not MASKS_DIR.exists():
        print(f"❌ Masks directory not found: {MASKS_DIR}")
        return

    if not YOLOSAM_MASKS_DIR.exists():
        print(f"❌ YOLOSAM masks directory not found: {YOLOSAM_MASKS_DIR}")
        return
    
    # Find all ground truth masks
    ground_truth_masks = sorted(MASKS_DIR.glob(f"*{GROUND_TRUTH_SUFFIX}"))
    
    if not ground_truth_masks:
        print(f"❌ No ground truth masks found in: {MASKS_DIR}")
        return
    
    print(f"Found {len(ground_truth_masks)} images to analyze\n")
    
    # Storage for results
    results = []
    
    # Process each ground truth mask
    for gt_path in ground_truth_masks:
        image_id = get_image_id(gt_path)
        
        # Construct paths for other masks
        grounded_sam_b_path = MASKS_DIR / f"{image_id}{GROUNDED_SAM_B_SUFFIX}"
        grounded_sam_l_path = MASKS_DIR / f"{image_id}{GROUNDED_SAM_L_SUFFIX}"
        grounded_sam_h_path = MASKS_DIR / f"{image_id}{GROUNDED_SAM_H_SUFFIX}"
        basic_path = MASKS_DIR / f"{image_id}{BASIC_SUFFIX}"
        yolosam_path = YOLOSAM_MASKS_DIR / f"{image_id}{YOLOSAM_SUFFIX}"
        
        # Load ground truth
        gt_mask = load_mask(gt_path)
        if gt_mask is None:
            print(f"⚠️  Skipped: Could not load ground truth {gt_path.name}")
            continue
        
        # Load GroundedSAM vit_b mask
        sam_b_mask = load_mask(grounded_sam_b_path)
        if sam_b_mask is None:
            print(f"⚠️  Skipped: Could not load GroundedSAM vit_b {grounded_sam_b_path.name}")
            continue

        # Load GroundedSAM vit_l mask
        sam_l_mask = load_mask(grounded_sam_l_path)
        if sam_l_mask is None:
            print(f"⚠️  Skipped: Could not load GroundedSAM vit_l {grounded_sam_l_path.name}")
            continue

        # Load GroundedSAM vit_h mask
        sam_h_mask = load_mask(grounded_sam_h_path)
        if sam_h_mask is None:
            print(f"⚠️  Skipped: Could not load GroundedSAM vit_h {grounded_sam_h_path.name}")
            continue
        
        # Load basic color mask
        basic_mask = load_mask(basic_path)
        if basic_mask is None:
            print(f"⚠️  Skipped: Could not load basic color {basic_path.name}")
            continue

        # Load YOLOSAM mask
        yolosam_mask = load_mask(yolosam_path)
        if yolosam_mask is None:
            print(f"⚠️  Skipped: Could not load YOLOSAM {yolosam_path.name}")
            continue
        
        # Calculate metrics for all techniques
        sam_b_metrics = calculate_metrics(gt_mask, sam_b_mask)
        sam_l_metrics = calculate_metrics(gt_mask, sam_l_mask)
        sam_h_metrics = calculate_metrics(gt_mask, sam_h_mask)
        basic_metrics = calculate_metrics(gt_mask, basic_mask)
        yolosam_metrics = calculate_metrics(gt_mask, yolosam_mask)
        
        results.append({
            'Image': image_id,
            'SAM_B_IoU': sam_b_metrics['IoU'],
            'SAM_B_Dice': sam_b_metrics['Dice'],
            'SAM_B_Precision': sam_b_metrics['Precision'],
            'SAM_B_Recall': sam_b_metrics['Recall'],
            'SAM_B_F1': sam_b_metrics['F1_Score'],
            'SAM_B_Accuracy': sam_b_metrics['Accuracy'],
            'SAM_L_IoU': sam_l_metrics['IoU'],
            'SAM_L_Dice': sam_l_metrics['Dice'],
            'SAM_L_Precision': sam_l_metrics['Precision'],
            'SAM_L_Recall': sam_l_metrics['Recall'],
            'SAM_L_F1': sam_l_metrics['F1_Score'],
            'SAM_L_Accuracy': sam_l_metrics['Accuracy'],
            'SAM_H_IoU': sam_h_metrics['IoU'],
            'SAM_H_Dice': sam_h_metrics['Dice'],
            'SAM_H_Precision': sam_h_metrics['Precision'],
            'SAM_H_Recall': sam_h_metrics['Recall'],
            'SAM_H_F1': sam_h_metrics['F1_Score'],
            'SAM_H_Accuracy': sam_h_metrics['Accuracy'],
            'Basic_IoU': basic_metrics['IoU'],
            'Basic_Dice': basic_metrics['Dice'],
            'Basic_Precision': basic_metrics['Precision'],
            'Basic_Recall': basic_metrics['Recall'],
            'Basic_F1': basic_metrics['F1_Score'],
            'Basic_Accuracy': basic_metrics['Accuracy'],
            'YOLOSAM_IoU': yolosam_metrics['IoU'],
            'YOLOSAM_Dice': yolosam_metrics['Dice'],
            'YOLOSAM_Precision': yolosam_metrics['Precision'],
            'YOLOSAM_Recall': yolosam_metrics['Recall'],
            'YOLOSAM_F1': yolosam_metrics['F1_Score'],
            'YOLOSAM_Accuracy': yolosam_metrics['Accuracy'],
        })
        
        print(f"✓ Processed: {image_id}")
    
    if not results:
        print("❌ No valid masks to analyze")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate mean metrics
    mean_metrics = {
        'Image': 'MEAN',
        'SAM_B_IoU': df['SAM_B_IoU'].mean(),
        'SAM_B_Dice': df['SAM_B_Dice'].mean(),
        'SAM_B_Precision': df['SAM_B_Precision'].mean(),
        'SAM_B_Recall': df['SAM_B_Recall'].mean(),
        'SAM_B_F1': df['SAM_B_F1'].mean(),
        'SAM_B_Accuracy': df['SAM_B_Accuracy'].mean(),
        'SAM_L_IoU': df['SAM_L_IoU'].mean(),
        'SAM_L_Dice': df['SAM_L_Dice'].mean(),
        'SAM_L_Precision': df['SAM_L_Precision'].mean(),
        'SAM_L_Recall': df['SAM_L_Recall'].mean(),
        'SAM_L_F1': df['SAM_L_F1'].mean(),
        'SAM_L_Accuracy': df['SAM_L_Accuracy'].mean(),
        'SAM_H_IoU': df['SAM_H_IoU'].mean(),
        'SAM_H_Dice': df['SAM_H_Dice'].mean(),
        'SAM_H_Precision': df['SAM_H_Precision'].mean(),
        'SAM_H_Recall': df['SAM_H_Recall'].mean(),
        'SAM_H_F1': df['SAM_H_F1'].mean(),
        'SAM_H_Accuracy': df['SAM_H_Accuracy'].mean(),
        'Basic_IoU': df['Basic_IoU'].mean(),
        'Basic_Dice': df['Basic_Dice'].mean(),
        'Basic_Precision': df['Basic_Precision'].mean(),
        'Basic_Recall': df['Basic_Recall'].mean(),
        'Basic_F1': df['Basic_F1'].mean(),
        'Basic_Accuracy': df['Basic_Accuracy'].mean(),
        'YOLOSAM_IoU': df['YOLOSAM_IoU'].mean(),
        'YOLOSAM_Dice': df['YOLOSAM_Dice'].mean(),
        'YOLOSAM_Precision': df['YOLOSAM_Precision'].mean(),
        'YOLOSAM_Recall': df['YOLOSAM_Recall'].mean(),
        'YOLOSAM_F1': df['YOLOSAM_F1'].mean(),
        'YOLOSAM_Accuracy': df['YOLOSAM_Accuracy'].mean(),
    }
    
    # Calculate std metrics
    std_metrics = {
        'Image': 'STD',
        'SAM_B_IoU': df['SAM_B_IoU'].std(),
        'SAM_B_Dice': df['SAM_B_Dice'].std(),
        'SAM_B_Precision': df['SAM_B_Precision'].std(),
        'SAM_B_Recall': df['SAM_B_Recall'].std(),
        'SAM_B_F1': df['SAM_B_F1'].std(),
        'SAM_B_Accuracy': df['SAM_B_Accuracy'].std(),
        'SAM_L_IoU': df['SAM_L_IoU'].std(),
        'SAM_L_Dice': df['SAM_L_Dice'].std(),
        'SAM_L_Precision': df['SAM_L_Precision'].std(),
        'SAM_L_Recall': df['SAM_L_Recall'].std(),
        'SAM_L_F1': df['SAM_L_F1'].std(),
        'SAM_L_Accuracy': df['SAM_L_Accuracy'].std(),
        'SAM_H_IoU': df['SAM_H_IoU'].std(),
        'SAM_H_Dice': df['SAM_H_Dice'].std(),
        'SAM_H_Precision': df['SAM_H_Precision'].std(),
        'SAM_H_Recall': df['SAM_H_Recall'].std(),
        'SAM_H_F1': df['SAM_H_F1'].std(),
        'SAM_H_Accuracy': df['SAM_H_Accuracy'].std(),
        'Basic_IoU': df['Basic_IoU'].std(),
        'Basic_Dice': df['Basic_Dice'].std(),
        'Basic_Precision': df['Basic_Precision'].std(),
        'Basic_Recall': df['Basic_Recall'].std(),
        'Basic_F1': df['Basic_F1'].std(),
        'Basic_Accuracy': df['Basic_Accuracy'].std(),
        'YOLOSAM_IoU': df['YOLOSAM_IoU'].std(),
        'YOLOSAM_Dice': df['YOLOSAM_Dice'].std(),
        'YOLOSAM_Precision': df['YOLOSAM_Precision'].std(),
        'YOLOSAM_Recall': df['YOLOSAM_Recall'].std(),
        'YOLOSAM_F1': df['YOLOSAM_F1'].std(),
        'YOLOSAM_Accuracy': df['YOLOSAM_Accuracy'].std(),
    }
    
    # Append summary rows
    df = pd.concat([df, pd.DataFrame([mean_metrics, std_metrics])], ignore_index=True)
    
    # Generate report
    print("\n" + "="*120)
    print("MASK QUALITY EVALUATION REPORT")
    print("="*120)
    
    report = []
    report.append("="*120)
    report.append("MASK QUALITY EVALUATION REPORT")
    report.append(f"Evaluating {len(ground_truth_masks)} images")
    report.append("="*120)
    report.append("")
    
    # Print summary table
    print("\n[SUMMARY STATISTICS]\n")
    report.append("[SUMMARY STATISTICS]\n")
    
    summary_data = {
        'Metric': ['IoU', 'Dice', 'Precision', 'Recall', 'F1 Score', 'Accuracy'],
        'GroundedSAM vit_b (Mean)': [
            f"{mean_metrics['SAM_B_IoU']:.4f}",
            f"{mean_metrics['SAM_B_Dice']:.4f}",
            f"{mean_metrics['SAM_B_Precision']:.4f}",
            f"{mean_metrics['SAM_B_Recall']:.4f}",
            f"{mean_metrics['SAM_B_F1']:.4f}",
            f"{mean_metrics['SAM_B_Accuracy']:.4f}",
        ],
        'GroundedSAM vit_l (Mean)': [
            f"{mean_metrics['SAM_L_IoU']:.4f}",
            f"{mean_metrics['SAM_L_Dice']:.4f}",
            f"{mean_metrics['SAM_L_Precision']:.4f}",
            f"{mean_metrics['SAM_L_Recall']:.4f}",
            f"{mean_metrics['SAM_L_F1']:.4f}",
            f"{mean_metrics['SAM_L_Accuracy']:.4f}",
        ],
        'GroundedSAM vit_h (Mean)': [
            f"{mean_metrics['SAM_H_IoU']:.4f}",
            f"{mean_metrics['SAM_H_Dice']:.4f}",
            f"{mean_metrics['SAM_H_Precision']:.4f}",
            f"{mean_metrics['SAM_H_Recall']:.4f}",
            f"{mean_metrics['SAM_H_F1']:.4f}",
            f"{mean_metrics['SAM_H_Accuracy']:.4f}",
        ],
        'Basic Color (Mean)': [
            f"{mean_metrics['Basic_IoU']:.4f}",
            f"{mean_metrics['Basic_Dice']:.4f}",
            f"{mean_metrics['Basic_Precision']:.4f}",
            f"{mean_metrics['Basic_Recall']:.4f}",
            f"{mean_metrics['Basic_F1']:.4f}",
            f"{mean_metrics['Basic_Accuracy']:.4f}",
        ],
        'YOLOSAM (Mean)': [
            f"{mean_metrics['YOLOSAM_IoU']:.4f}",
            f"{mean_metrics['YOLOSAM_Dice']:.4f}",
            f"{mean_metrics['YOLOSAM_Precision']:.4f}",
            f"{mean_metrics['YOLOSAM_Recall']:.4f}",
            f"{mean_metrics['YOLOSAM_F1']:.4f}",
            f"{mean_metrics['YOLOSAM_Accuracy']:.4f}",
        ],
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    report.append(summary_df.to_string(index=False))
    report.append("")
    
    # Standard deviations
    print("\n[STANDARD DEVIATIONS]\n")
    report.append("\n[STANDARD DEVIATIONS]\n")
    
    std_data = {
        'Metric': ['IoU', 'Dice', 'Precision', 'Recall', 'F1 Score', 'Accuracy'],
        'GroundedSAM vit_b (Std)': [
            f"{std_metrics['SAM_B_IoU']:.4f}",
            f"{std_metrics['SAM_B_Dice']:.4f}",
            f"{std_metrics['SAM_B_Precision']:.4f}",
            f"{std_metrics['SAM_B_Recall']:.4f}",
            f"{std_metrics['SAM_B_F1']:.4f}",
            f"{std_metrics['SAM_B_Accuracy']:.4f}",
        ],
        'GroundedSAM vit_l (Std)': [
            f"{std_metrics['SAM_L_IoU']:.4f}",
            f"{std_metrics['SAM_L_Dice']:.4f}",
            f"{std_metrics['SAM_L_Precision']:.4f}",
            f"{std_metrics['SAM_L_Recall']:.4f}",
            f"{std_metrics['SAM_L_F1']:.4f}",
            f"{std_metrics['SAM_L_Accuracy']:.4f}",
        ],
        'GroundedSAM vit_h (Std)': [
            f"{std_metrics['SAM_H_IoU']:.4f}",
            f"{std_metrics['SAM_H_Dice']:.4f}",
            f"{std_metrics['SAM_H_Precision']:.4f}",
            f"{std_metrics['SAM_H_Recall']:.4f}",
            f"{std_metrics['SAM_H_F1']:.4f}",
            f"{std_metrics['SAM_H_Accuracy']:.4f}",
        ],
        'Basic Color (Std)': [
            f"{std_metrics['Basic_IoU']:.4f}",
            f"{std_metrics['Basic_Dice']:.4f}",
            f"{std_metrics['Basic_Precision']:.4f}",
            f"{std_metrics['Basic_Recall']:.4f}",
            f"{std_metrics['Basic_F1']:.4f}",
            f"{std_metrics['Basic_Accuracy']:.4f}",
        ],
        'YOLOSAM (Std)': [
            f"{std_metrics['YOLOSAM_IoU']:.4f}",
            f"{std_metrics['YOLOSAM_Dice']:.4f}",
            f"{std_metrics['YOLOSAM_Precision']:.4f}",
            f"{std_metrics['YOLOSAM_Recall']:.4f}",
            f"{std_metrics['YOLOSAM_F1']:.4f}",
            f"{std_metrics['YOLOSAM_Accuracy']:.4f}",
        ],
    }
    
    std_df = pd.DataFrame(std_data)
    print(std_df.to_string(index=False))
    report.append(std_df.to_string(index=False))
    report.append("")
    
    # Key findings
    print("\n[KEY FINDINGS]\n")
    report.append("\n[KEY FINDINGS]\n")
    
    findings = []
    method_labels = {
        'SAM_B': 'GroundedSAM vit_b',
        'SAM_L': 'GroundedSAM vit_l',
        'SAM_H': 'GroundedSAM vit_h',
        'Basic': 'Basic Color',
        'YOLOSAM': 'YOLOSAM',
    }
    metric_map = {
        'IoU': ['SAM_B_IoU', 'SAM_L_IoU', 'SAM_H_IoU', 'Basic_IoU', 'YOLOSAM_IoU'],
        'Dice': ['SAM_B_Dice', 'SAM_L_Dice', 'SAM_H_Dice', 'Basic_Dice', 'YOLOSAM_Dice'],
        'Precision': ['SAM_B_Precision', 'SAM_L_Precision', 'SAM_H_Precision', 'Basic_Precision', 'YOLOSAM_Precision'],
        'Recall': ['SAM_B_Recall', 'SAM_L_Recall', 'SAM_H_Recall', 'Basic_Recall', 'YOLOSAM_Recall'],
        'Accuracy': ['SAM_B_Accuracy', 'SAM_L_Accuracy', 'SAM_H_Accuracy', 'Basic_Accuracy', 'YOLOSAM_Accuracy'],
    }

    for metric_name, keys in metric_map.items():
        scores = {}
        for key in keys:
            method_key = '_'.join(key.split('_')[:-1])
            scores[method_labels[method_key]] = mean_metrics[key]
        best_method = max(scores, key=scores.get)
        findings.append(f"[+] {metric_name}: {best_method} achieves best score ({scores[best_method]:.4f})")
    
    for finding in findings:
        print(finding)
        report.append(finding)
    
    report.append("")
    report.append("\n[METRIC DEFINITIONS]\n")
    report.append("• IoU (Intersection over Union): Ratio of intersection to union of predicted and ground truth masks")
    report.append("• Dice Coefficient: 2 * Intersection / (Sum of masks), ranges from 0 to 1")
    report.append("• Precision: True Positives / (True Positives + False Positives)")
    report.append("• Recall: True Positives / (True Positives + False Negatives)")
    report.append("• F1 Score: Harmonic mean of Precision and Recall")
    report.append("• Accuracy: Correctly classified pixels / Total pixels")
    report.append("")
    report.append("="*120)
    
    # Save report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to: {OUTPUT_REPORT}")
    
    # Print individual results table (first 10 rows for preview)
    print("\n[PER-IMAGE RESULTS (first 10 images)]\n")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
