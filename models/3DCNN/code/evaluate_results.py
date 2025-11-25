import os
import numpy as np
import nibabel as nib


def calculate_dice(pred_mask, gt_mask):
    intersection = np.sum(pred_mask * gt_mask) * 2.0
    total = np.sum(pred_mask) + np.sum(gt_mask)
    if total == 0: return 1.0
    return intersection / total


def calculate_metrics(pred_mask, gt_mask):
    tp = np.sum(pred_mask * gt_mask)
    fp = np.sum(pred_mask * (1 - gt_mask))
    fn = np.sum((1 - pred_mask) * gt_mask)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0

    return sens, prec, tp, fp, fn


def save_comparison_image(pred_mask, gt_mask, affine, output_path):
    comparison = np.zeros_like(pred_mask, dtype=np.int16)
    comparison[(gt_mask == 1) & (pred_mask == 1)] = 1
    comparison[(gt_mask == 1) & (pred_mask == 0)] = 2
    comparison[(gt_mask == 0) & (pred_mask == 1)] = 3

    nib.save(nib.Nifti1Image(comparison, affine), output_path)


def evaluate_patient(patient_id, inference_base, gt_base):
    pred_path = os.path.join(inference_base, patient_id, "my_ms_model", "my_ms_model_debug_prob_0.nii.gz")
    gt_path = os.path.join(gt_base, patient_id, "lesion.nii.gz")

    print(f"\n--- Evaluating {patient_id} ---")
    if not os.path.exists(pred_path):
        print("Prediction file not found.")
        return

    prob_obj = nib.load(pred_path)
    pred_data = prob_obj.get_data()
    gt_data = nib.load(gt_path).get_data()

    pred_mask = (pred_data > 0).astype(int)
    gt_mask = (gt_data > 0).astype(int)

    dice = calculate_dice(pred_mask, gt_mask)
    sens, prec, tp, fp, fn = calculate_metrics(pred_mask, gt_mask)

    print(f"Metrics:")
    print(f"  -> Dice Score:   {dice:.4f}")
    print(f"  -> Sensitivity:  {sens:.4f} (Recall)")
    print(f"  -> Precision:    {prec:.4f} (PPV)")
    print("-" * 30)
    print(f"Volume Stats (in voxels):")
    print(f"  -> True Positives: {tp} (Correct)")
    print(f"  -> False Positives: {fp} (Noise)")
    print(f"  -> False Negatives: {fn} (Missed)")

    comp_path = os.path.join(inference_base, patient_id, "my_ms_model", "comparison_map.nii.gz")
    save_comparison_image(pred_mask, gt_mask, prob_obj.affine, comp_path)
    print(f"\n-> Saved Comparison Map: {comp_path}")
    print("   (Open in ITK-SNAP. Label 1=Green/Hit, 2=Red/Miss, 3=Blue/Noise)")


inference_dir = r"D:\Desktop\MS_Project\open_ms_data\inference"
ground_truth_dir = r"D:\Desktop\MS_Project\open_ms_data\train"

evaluate_patient("patient01", inference_dir, ground_truth_dir)


