from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# -----------------------
# Settings
# -----------------------
MASKS_DIR = Path("images_and_masks/maize_2/frame_0001_masks")
OUTPUT_REPORT = Path("mask_quality_report.txt")

# Suffix patterns for different mask types
GROUND_TRUTH_SUFFIX = "_mask_ground_truth.png"
GROUNDED_SAM_SUFFIX = "_mask_grounded_sam_vit_b.png"
BASIC_SUFFIX = "_mask_colors.png"


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
    return filepath.stem.replace(GROUND_TRUTH_SUFFIX[:-4], "").replace(GROUNDED_SAM_SUFFIX[:-4], "").replace(BASIC_SUFFIX[:-4], "")


def main():
    """Main analysis function."""
    
    if not MASKS_DIR.exists():
        print(f"❌ Masks directory not found: {MASKS_DIR}")
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
        grounded_sam_path = MASKS_DIR / f"{image_id}{GROUNDED_SAM_SUFFIX}"
        basic_path = MASKS_DIR / f"{image_id}{BASIC_SUFFIX}"
        
        # Load ground truth
        gt_mask = load_mask(gt_path)
        if gt_mask is None:
            print(f"⚠️  Skipped: Could not load ground truth {gt_path.name}")
            continue
        
        # Load GroundedSAM mask
        sam_mask = load_mask(grounded_sam_path)
        if sam_mask is None:
            print(f"⚠️  Skipped: Could not load GroundedSAM {grounded_sam_path.name}")
            continue
        
        # Load basic color mask
        basic_mask = load_mask(basic_path)
        if basic_mask is None:
            print(f"⚠️  Skipped: Could not load basic color {basic_path.name}")
            continue
        
        # Calculate metrics for both techniques
        sam_metrics = calculate_metrics(gt_mask, sam_mask)
        basic_metrics = calculate_metrics(gt_mask, basic_mask)
        
        results.append({
            'Image': image_id,
            'SAM_IoU': sam_metrics['IoU'],
            'SAM_Dice': sam_metrics['Dice'],
            'SAM_Precision': sam_metrics['Precision'],
            'SAM_Recall': sam_metrics['Recall'],
            'SAM_F1': sam_metrics['F1_Score'],
            'SAM_Accuracy': sam_metrics['Accuracy'],
            'Basic_IoU': basic_metrics['IoU'],
            'Basic_Dice': basic_metrics['Dice'],
            'Basic_Precision': basic_metrics['Precision'],
            'Basic_Recall': basic_metrics['Recall'],
            'Basic_F1': basic_metrics['F1_Score'],
            'Basic_Accuracy': basic_metrics['Accuracy'],
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
        'SAM_IoU': df['SAM_IoU'].mean(),
        'SAM_Dice': df['SAM_Dice'].mean(),
        'SAM_Precision': df['SAM_Precision'].mean(),
        'SAM_Recall': df['SAM_Recall'].mean(),
        'SAM_F1': df['SAM_F1'].mean(),
        'SAM_Accuracy': df['SAM_Accuracy'].mean(),
        'Basic_IoU': df['Basic_IoU'].mean(),
        'Basic_Dice': df['Basic_Dice'].mean(),
        'Basic_Precision': df['Basic_Precision'].mean(),
        'Basic_Recall': df['Basic_Recall'].mean(),
        'Basic_F1': df['Basic_F1'].mean(),
        'Basic_Accuracy': df['Basic_Accuracy'].mean(),
    }
    
    # Calculate std metrics
    std_metrics = {
        'Image': 'STD',
        'SAM_IoU': df['SAM_IoU'].std(),
        'SAM_Dice': df['SAM_Dice'].std(),
        'SAM_Precision': df['SAM_Precision'].std(),
        'SAM_Recall': df['SAM_Recall'].std(),
        'SAM_F1': df['SAM_F1'].std(),
        'SAM_Accuracy': df['SAM_Accuracy'].std(),
        'Basic_IoU': df['Basic_IoU'].std(),
        'Basic_Dice': df['Basic_Dice'].std(),
        'Basic_Precision': df['Basic_Precision'].std(),
        'Basic_Recall': df['Basic_Recall'].std(),
        'Basic_F1': df['Basic_F1'].std(),
        'Basic_Accuracy': df['Basic_Accuracy'].std(),
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
        'GroundedSAM (Mean)': [
            f"{mean_metrics['SAM_IoU']:.4f}",
            f"{mean_metrics['SAM_Dice']:.4f}",
            f"{mean_metrics['SAM_Precision']:.4f}",
            f"{mean_metrics['SAM_Recall']:.4f}",
            f"{mean_metrics['SAM_F1']:.4f}",
            f"{mean_metrics['SAM_Accuracy']:.4f}",
        ],
        'Basic Color (Mean)': [
            f"{mean_metrics['Basic_IoU']:.4f}",
            f"{mean_metrics['Basic_Dice']:.4f}",
            f"{mean_metrics['Basic_Precision']:.4f}",
            f"{mean_metrics['Basic_Recall']:.4f}",
            f"{mean_metrics['Basic_F1']:.4f}",
            f"{mean_metrics['Basic_Accuracy']:.4f}",
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
        'GroundedSAM (Std)': [
            f"{std_metrics['SAM_IoU']:.4f}",
            f"{std_metrics['SAM_Dice']:.4f}",
            f"{std_metrics['SAM_Precision']:.4f}",
            f"{std_metrics['SAM_Recall']:.4f}",
            f"{std_metrics['SAM_F1']:.4f}",
            f"{std_metrics['SAM_Accuracy']:.4f}",
        ],
        'Basic Color (Std)': [
            f"{std_metrics['Basic_IoU']:.4f}",
            f"{std_metrics['Basic_Dice']:.4f}",
            f"{std_metrics['Basic_Precision']:.4f}",
            f"{std_metrics['Basic_Recall']:.4f}",
            f"{std_metrics['Basic_F1']:.4f}",
            f"{std_metrics['Basic_Accuracy']:.4f}",
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
    
    # Compare IoU
    if mean_metrics['SAM_IoU'] > mean_metrics['Basic_IoU']:
        diff = (mean_metrics['SAM_IoU'] - mean_metrics['Basic_IoU']) / mean_metrics['Basic_IoU'] * 100
        findings.append(f"[+] GroundedSAM achieves {diff:.1f}% better IoU ({mean_metrics['SAM_IoU']:.4f} vs {mean_metrics['Basic_IoU']:.4f})")
    else:
        diff = (mean_metrics['Basic_IoU'] - mean_metrics['SAM_IoU']) / mean_metrics['SAM_IoU'] * 100
        findings.append(f"[+] Basic Color achieves {diff:.1f}% better IoU ({mean_metrics['Basic_IoU']:.4f} vs {mean_metrics['SAM_IoU']:.4f})")
    
    # Compare Dice
    if mean_metrics['SAM_Dice'] > mean_metrics['Basic_Dice']:
        findings.append(f"[+] GroundedSAM has better Dice coefficient ({mean_metrics['SAM_Dice']:.4f} vs {mean_metrics['Basic_Dice']:.4f})")
    else:
        findings.append(f"[+] Basic Color has better Dice coefficient ({mean_metrics['Basic_Dice']:.4f} vs {mean_metrics['SAM_Dice']:.4f})")
    
    # Compare Precision
    if mean_metrics['SAM_Precision'] > mean_metrics['Basic_Precision']:
        findings.append(f"[+] GroundedSAM has better Precision ({mean_metrics['SAM_Precision']:.4f} vs {mean_metrics['Basic_Precision']:.4f})")
    else:
        findings.append(f"[+] Basic Color has better Precision ({mean_metrics['Basic_Precision']:.4f} vs {mean_metrics['SAM_Precision']:.4f})")
    
    # Compare Recall
    if mean_metrics['SAM_Recall'] > mean_metrics['Basic_Recall']:
        findings.append(f"[+] GroundedSAM has better Recall ({mean_metrics['SAM_Recall']:.4f} vs {mean_metrics['Basic_Recall']:.4f})")
    else:
        findings.append(f"[+] Basic Color has better Recall ({mean_metrics['Basic_Recall']:.4f} vs {mean_metrics['SAM_Recall']:.4f})")
    
    # Compare Accuracy
    if mean_metrics['SAM_Accuracy'] > mean_metrics['Basic_Accuracy']:
        findings.append(f"[+] GroundedSAM has better overall Accuracy ({mean_metrics['SAM_Accuracy']:.4f} vs {mean_metrics['Basic_Accuracy']:.4f})")
    else:
        findings.append(f"[+] Basic Color has better overall Accuracy ({mean_metrics['Basic_Accuracy']:.4f} vs {mean_metrics['SAM_Accuracy']:.4f})")
    
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
