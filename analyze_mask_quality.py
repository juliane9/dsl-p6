from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import re


# -----------------------
# Settings
# -----------------------
MAIZE_DIR = Path("images_and_masks/maize_2")

# Suffix patterns for different mask types
GROUND_TRUTH_SUFFIX = "_mask_ground_truth.png"
GROUNDED_SAM_SUFFIX = "_mask_grounded_sam_vit_b.png"
GROUNDED_SAM_LARGE_SUFFIX = "_mask_grounded_sam_vit_l.png"
GROUNDED_SAM_HUGE_SUFFIX = "_mask_grounded_sam_vit_h.png"
BASIC_SUFFIX = "_mask_colors.png"

# Define all available methods
METHODS_CONFIG = {
    'Basic Color': BASIC_SUFFIX,
    'GroundedSAM vit_b': GROUNDED_SAM_SUFFIX,
    'GroundedSAM vit_l': GROUNDED_SAM_LARGE_SUFFIX,
    'GroundedSAM vit_h': GROUNDED_SAM_HUGE_SUFFIX,
}


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
    
    # Precision, Recall, F1, Balanced Accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + tnr) / 2
    
    metrics = {
        'IoU': calculate_iou(ground_truth, prediction),
        'Dice': calculate_dice(ground_truth, prediction),
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score,
        'Balanced_Accuracy': balanced_accuracy,
    }
    
    return metrics


def get_frame_id(frame_dir):
    """Extract frame ID from directory name (e.g., 'frame_0001' -> '0001')."""
    match = re.search(r'frame_(\d+)', frame_dir.name)
    return match.group(1) if match else frame_dir.name


def find_frame_directories():
    """Find all frame mask directories."""
    if not MAIZE_DIR.exists():
        print(f"❌ Maize directory not found: {MAIZE_DIR}")
        return []
    
    frame_dirs = sorted([d for d in MAIZE_DIR.iterdir() if d.is_dir() and '_masks' in d.name])
    return frame_dirs


def detect_available_methods(masks_dir):
    """Detect which mask methods are available in this frame directory."""
    available_methods = {}
    
    # Check for ground truth first
    gt_masks = list(masks_dir.glob(f"*{GROUND_TRUTH_SUFFIX}"))
    if not gt_masks:
        return None
    
    # Check which prediction methods exist
    for method_name, suffix in METHODS_CONFIG.items():
        method_masks = list(masks_dir.glob(f"*{suffix}"))
        if method_masks:
            available_methods[method_name] = suffix
    
    return available_methods if available_methods else None


def analyze_frame(masks_dir, frame_id):
    """Analyze a single frame and return results."""
    
    if not masks_dir.exists():
        print(f"❌ Frame directory not found: {masks_dir}")
        return None
    
    # Find all ground truth masks
    ground_truth_masks = sorted(masks_dir.glob(f"*{GROUND_TRUTH_SUFFIX}"))
    
    if not ground_truth_masks:
        print(f"❌ No ground truth masks found in: {masks_dir}")
        return None
    
    # Detect available methods
    available_methods = detect_available_methods(masks_dir)
    if not available_methods:
        print(f"❌ No prediction masks found in: {masks_dir}")
        return None
    
    print(f"\n📁 Frame {frame_id}: Found {len(ground_truth_masks)} images")
    print(f"   Available methods: {', '.join(available_methods.keys())}")
    
    # Storage for results
    results = []
    
    # Process each ground truth mask
    for gt_path in ground_truth_masks:
        image_id = gt_path.stem.replace(GROUND_TRUTH_SUFFIX[:-4], "")
        
        # Load ground truth
        gt_mask = load_mask(gt_path)
        if gt_mask is None:
            print(f"   ⚠️  Skipped: Could not load ground truth {gt_path.name}")
            continue
        
        # Load prediction masks for available methods
        row = {'Image': image_id}
        skip_image = False
        
        for method_name, suffix in available_methods.items():
            pred_path = masks_dir / f"{image_id}{suffix}"
            pred_mask = load_mask(pred_path)
            
            if pred_mask is None:
                print(f"   ⚠️  Skipped: Could not load {method_name} {pred_path.name}")
                skip_image = True
                break
            
            # Calculate metrics
            metrics = calculate_metrics(gt_mask, pred_mask)
            
            # Store metrics with method prefix
            method_prefix = method_name.replace(' ', '_').replace('.', '_').replace('_vit', '_vit')
            method_prefix = method_prefix.replace('GroundedSAM_vit_b', 'SAM').replace('GroundedSAM_vit_l', 'SAM_L').replace('GroundedSAM_vit_h', 'SAM_H').replace('Basic_Color', 'Basic')
            
            row[f'{method_prefix}_IoU'] = metrics['IoU']
            row[f'{method_prefix}_Dice'] = metrics['Dice']
            row[f'{method_prefix}_Precision'] = metrics['Precision']
            row[f'{method_prefix}_Recall'] = metrics['Recall']
            row[f'{method_prefix}_F1'] = metrics['F1_Score']
            row[f'{method_prefix}_Balanced_Accuracy'] = metrics['Balanced_Accuracy']
        
        if not skip_image:
            results.append(row)
            print(f"   ✓ Processed: {image_id}")
    
    if not results:
        print(f"❌ No valid masks to analyze in frame {frame_id}")
        return None
    
    return {
        'frame_id': frame_id,
        'masks_dir': masks_dir,
        'available_methods': available_methods,
        'results': results,
    }


def generate_report(analysis_data):
    """Generate report for a single frame."""
    
    frame_id = analysis_data['frame_id']
    available_methods = analysis_data['available_methods']
    results = analysis_data['results']
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Build method prefixes for column names
    method_prefixes = {}
    for method_name in available_methods.keys():
        prefix = method_name.replace(' ', '_').replace('.', '_').replace('_vit', '_vit')
        prefix = prefix.replace('GroundedSAM_vit_b', 'SAM').replace('GroundedSAM_vit_l', 'SAM_L').replace('GroundedSAM_vit_h', 'SAM_H').replace('Basic_Color', 'Basic')
        method_prefixes[method_name] = prefix
    
    # Calculate mean metrics
    mean_metrics = {'Image': 'MEAN'}
    for method_name, prefix in method_prefixes.items():
        mean_metrics[f'{prefix}_IoU'] = df[f'{prefix}_IoU'].mean()
        mean_metrics[f'{prefix}_Dice'] = df[f'{prefix}_Dice'].mean()
        mean_metrics[f'{prefix}_Precision'] = df[f'{prefix}_Precision'].mean()
        mean_metrics[f'{prefix}_Recall'] = df[f'{prefix}_Recall'].mean()
        mean_metrics[f'{prefix}_F1'] = df[f'{prefix}_F1'].mean()
        mean_metrics[f'{prefix}_Balanced_Accuracy'] = df[f'{prefix}_Balanced_Accuracy'].mean()
    
    # Calculate std metrics
    std_metrics = {'Image': 'STD'}
    for method_name, prefix in method_prefixes.items():
        std_metrics[f'{prefix}_IoU'] = df[f'{prefix}_IoU'].std()
        std_metrics[f'{prefix}_Dice'] = df[f'{prefix}_Dice'].std()
        std_metrics[f'{prefix}_Precision'] = df[f'{prefix}_Precision'].std()
        std_metrics[f'{prefix}_Recall'] = df[f'{prefix}_Recall'].std()
        std_metrics[f'{prefix}_F1'] = df[f'{prefix}_F1'].std()
        std_metrics[f'{prefix}_Balanced_Accuracy'] = df[f'{prefix}_Balanced_Accuracy'].std()
    
    # Generate report
    print(f"\n{'='*120}")
    print(f"MASK QUALITY EVALUATION REPORT - Frame {frame_id}")
    print(f"{'='*120}")
    
    report = []
    report.append("="*120)
    report.append(f"MASK QUALITY EVALUATION REPORT - Frame {frame_id}")
    report.append(f"Evaluating {len(results)} images with {len(available_methods)} methods")
    report.append("="*120)
    report.append("")
    
    # Print summary table
    print("\n[SUMMARY STATISTICS]\n")
    report.append("[SUMMARY STATISTICS]\n")
    
    summary_data = {
        'Metric': ['IoU', 'Dice', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy'],
    }
    
    for method_name, prefix in method_prefixes.items():
        summary_data[method_name] = [
            f"{mean_metrics[f'{prefix}_IoU']:.4f}",
            f"{mean_metrics[f'{prefix}_Dice']:.4f}",
            f"{mean_metrics[f'{prefix}_Precision']:.4f}",
            f"{mean_metrics[f'{prefix}_Recall']:.4f}",
            f"{mean_metrics[f'{prefix}_F1']:.4f}",
            f"{mean_metrics[f'{prefix}_Balanced_Accuracy']:.4f}",
        ]
    
    summary_df = pd.DataFrame(summary_data)
    summary_table = summary_df.to_string(index=False)
    print(summary_table)
    report.append(summary_table)
    report.append("")
    
    # Standard deviations
    print("\n[STANDARD DEVIATIONS]\n")
    report.append("\n[STANDARD DEVIATIONS]\n")
    
    std_data = {
        'Metric': ['IoU', 'Dice', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy'],
    }
    
    for method_name, prefix in method_prefixes.items():
        std_data[method_name] = [
            f"{std_metrics[f'{prefix}_IoU']:.4f}",
            f"{std_metrics[f'{prefix}_Dice']:.4f}",
            f"{std_metrics[f'{prefix}_Precision']:.4f}",
            f"{std_metrics[f'{prefix}_Recall']:.4f}",
            f"{std_metrics[f'{prefix}_F1']:.4f}",
            f"{std_metrics[f'{prefix}_Balanced_Accuracy']:.4f}",
        ]
    
    std_df = pd.DataFrame(std_data)
    std_table = std_df.to_string(index=False)
    print(std_table)
    report.append(std_table)
    report.append("")
    
    # Key findings
    print("\n[KEY FINDINGS]\n")
    report.append("\n[KEY FINDINGS]\n")
    
    findings = []
    metrics_to_check = ['IoU', 'Dice', 'Precision', 'Recall', 'Balanced Accuracy']
    
    for metric_name in metrics_to_check:
        metric_scores = {}
        metric_key = metric_name.replace(' ', '_')
        for method_name, prefix in method_prefixes.items():
            key = f'{prefix}_{metric_key}'
            metric_scores[method_name] = mean_metrics[key]
        
        best_method = max(metric_scores, key=metric_scores.get)
        best_score = metric_scores[best_method]
        findings.append(f"[+] {metric_name}: {best_method} achieves best score ({best_score:.4f})")
    
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
    report.append("• Balanced Accuracy: (Recall + Specificity) / 2")
    report.append("")
    report.append("="*120)
    
    return '\n'.join(report)


def main():
    """Main analysis function."""
    
    # Find all frame directories
    frame_dirs = find_frame_directories()
    
    if not frame_dirs:
        print("❌ No frame mask directories found")
        return
    
    print(f"🔍 Found {len(frame_dirs)} frame directories")
    
    # Analyze each frame
    for frame_dir in frame_dirs:
        frame_id = get_frame_id(frame_dir)
        analysis_data = analyze_frame(frame_dir, frame_id)
        
        if analysis_data:
            # Generate report
            report_text = generate_report(analysis_data)
            
            # Save report to frame-specific file
            report_file = Path(f"mask_quality_report_frame_{frame_id}.txt")
            with open(report_file, 'w') as f:
                f.write(report_text)
            
            print(f"✅ Report saved to: {report_file}")
        else:
            print(f"⏭️  Skipping frame {frame_id}")
    
    print("\n" + "="*120)
    print("✅ Analysis complete!")
    print("="*120)

if __name__ == "__main__":
    main()

