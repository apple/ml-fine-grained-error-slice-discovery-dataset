#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Evaluate the GESD dataset using precision@k metric.

Because annotation IDs reset to 1 in each JSON file, each ID must be prefixed with
the dataset name to uniquely identify annotations across the entire dataset.

Prediction Format Requirements:
The prediction file (`--pred_file`) must be a JSON file containing a dictionary where:
- Keys: Any string representing the predicted slice (e.g., "pred_slice_1", "1", or a semantic
  description like "blurry faces").
- Values: A list of string IDs representing the instances in that slice.
  IMPORTANT: The IDs within each list MUST be sorted in descending order of confidence
  (highest confidence first). The evaluation metric relies on this order.
  Because annotation IDs reset to 1 in each JSON file, each ID must be prefixed with
  the dataset name, formatted as "{dataset_name}_{annotation_id}"
  (e.g., "coco_detection_1024", "face_detection_15").

Example of valid prediction format:
{
    "blurry faces": ["coco_detection_1024", "kitti_detection_2048", "face_detection_512"],
    "1": ["coco_detection_99", "coco_detection_100"]
}

Usage (run from the repository root):
  # For Detection (evaluates coco_detection, kitti_detection, face_detection together)
  python code/evaluate_precision.py --task detection --pred_file path/to/det_preds.json --k 10

  # For Segmentation (evaluates coco_segmentation)
  python code/evaluate_precision.py --task segmentation --pred_file path/to/seg_preds.json --k 10
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

# Import the loader from the repository
from load_dataset import GESDDataset


def load_all_gt_slices(data_files: List[str]) -> Dict[str, List[str]]:
    """
    Load all GT slices from the given GESD annotation files.
    Aggregates annotations across multiple datasets into a single GT mapping
    using uniquely constructed prefixed IDs.

    Returns:
        A dictionary mapping from canonical error-slice name (reason)
        to a list of prefixed IDs.
    """
    gt_slices = defaultdict(list)

    print(f"Loading GT slices from {len(data_files)} dataset files...")
    for path in data_files:
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        try:
            ds = GESDDataset(path)
            for ann in ds.error_slices():
                # We use the canonical reason as the slice name
                reason = ann["reason"]

                # Construct prefixed ID to prevent collisions across JSON files
                prefixed_id = f"{dataset_name}_{ann['id']}"

                gt_slices[reason].append(prefixed_id)
        except Exception as e:
            print(f"Warning: Failed to load {path}. Error: {e}")

    # Convert defaultdict to regular dict for return
    return dict(gt_slices)


def load_pred_slices(file_path: str) -> Dict[str, List[str]]:
    """
    Load prediction slices from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        slices = json.load(f)

    return slices


def create_sample_to_slice_mapping(slices: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Create a mapping from prefixed ID to GT slice index.

    Returns:
        Dictionary mapping prefixed ID to its corresponding GT slice index.
    """
    sample_to_slice = {}
    slice_name_to_id = {name: i for i, name in enumerate(slices.keys())}

    for slice_name, sample_list in slices.items():
        slice_id = slice_name_to_id[slice_name]
        for sample in sample_list:
            # We convert sample to string to ensure safe matching
            sample_to_slice[str(sample)] = slice_id

    return sample_to_slice


def calculate_precision_at_k(
    gt_slices: Dict[str, List[str]], pred_slices: Dict[str, List[str]], k: int
) -> Tuple[List[Dict], float]:
    """
    Calculate precision@k metric between GT slices and predicted slices.

    If GT slice size < k, the denominator is the actual GT slice size
    to provide a fair evaluation for small slices.
    """
    gt_sample_to_slice = create_sample_to_slice_mapping(gt_slices)

    gt_slice_names = list(gt_slices.keys())

    evaluation_results = []
    precision_scores = []

    print(f"\nCalculating precision@{k}...")
    print(f"Total GT slices: {len(gt_slice_names)}")
    print(f"Total Pred slices: {len(pred_slices)}")
    print("-" * 60)

    for gt_id, gt_slice_name in enumerate(gt_slice_names):
        gt_slice_size = len(gt_slices[gt_slice_name])

        max_precision_for_gt_slice = 0.0
        best_pred_slice_name = None

        for pred_slice_name, pred_samples in pred_slices.items():
            if not pred_samples:
                continue

            top_k_samples = pred_samples[:k]

            correct_in_top_k = 0
            for sample in top_k_samples:
                # Ensure prefixed ID is treated as string for comparison
                str_sample = str(sample)
                if (
                    str_sample in gt_sample_to_slice
                    and gt_sample_to_slice[str_sample] == gt_id
                ):
                    correct_in_top_k += 1

            denominator = min(k, gt_slice_size)
            if denominator == 0:
                continue

            precision = correct_in_top_k / denominator

            if precision > max_precision_for_gt_slice:
                max_precision_for_gt_slice = precision
                best_pred_slice_name = pred_slice_name

        evaluation_results.append(
            {
                "GT_Slice_Name": gt_slice_name,
                "GT_Slice_Size": gt_slice_size,
                "Best_Pred_Slice": best_pred_slice_name,
                f"Precision@{k}": max_precision_for_gt_slice,
            }
        )

        precision_scores.append(max_precision_for_gt_slice)

    avg_precision_at_k = np.mean(precision_scores) if precision_scores else 0.0

    return evaluation_results, avg_precision_at_k


def print_detailed_results(results: List[Dict], avg_precision: float, k: int):
    """Print detailed evaluation results in a formatted table."""
    print("\n" + "=" * 80)
    print("Detailed Evaluation Results")
    print("=" * 80)

    if not results:
        print("No results to display.")
        return

    df = pd.DataFrame(results)

    column_order = [
        "GT_Slice_Name",
        "GT_Slice_Size",
        "Best_Pred_Slice",
        f"Precision@{k}",
    ]
    df = df[column_order]

    # Sort by Precision (descending) and then GT Slice Size (descending)
    df = df.sort_values(
        by=[f"Precision@{k}", "GT_Slice_Size"], ascending=[False, False]
    )

    df[f"Precision@{k}"] = df[f"Precision@{k}"].apply(lambda x: f"{x:.4f}")

    print(df.to_string(index=False, max_colwidth=40))

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total GT Slices: {len(results)}")
    print(f"Average Precision@{k}: {avg_precision:.4f}")

    perfect_matches = sum(1 for r in results if float(r[f"Precision@{k}"]) == 1.0)
    print(
        f"Perfect Matches (Precision@{k} = 1.0): {perfect_matches}/{len(results)} ({perfect_matches / len(results) * 100:.1f}%)"
    )

    valid_matches = sum(1 for r in results if float(r[f"Precision@{k}"]) > 0.0)
    print(
        f"Valid Matches (Precision@{k} > 0.0): {valid_matches}/{len(results)} ({valid_matches / len(results) * 100:.1f}%)"
    )


def save_results(results: List[Dict], avg_precision: float, k: int, output_file: str):
    """Save the evaluation results to a JSON file."""
    output_data = {
        "average_precision_at_k": avg_precision,
        "total_gt_slices": len(results),
        "summary": {
            "perfect_matches": sum(
                1 for r in results if float(r[f"Precision@{k}"]) == 1.0
            ),
            "valid_matches": sum(
                1 for r in results if float(r[f"Precision@{k}"]) > 0.0
            ),
        },
        "detailed_results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\nResults successfully saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate precision@k metric for GESD dataset."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["detection", "segmentation"],
        help="The task to evaluate. 'detection' merges 3 detection JSONs; 'segmentation' uses the segmentation JSON.",
    )
    parser.add_argument(
        "--pred_file", type=str, required=True, help="Path to the prediction JSON file."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the GT JSON files (default: 'data').",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="k value for precision@k (default: 10)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output results (default: precision_at_k_<task>_results.json).",
    )

    args = parser.parse_args()

    # Determine which files to load based on the task
    if args.task == "detection":
        data_files = [
            f"{args.data_dir}/coco_detection.json",
            f"{args.data_dir}/kitti_detection.json",
            f"{args.data_dir}/face_detection.json",
        ]
    elif args.task == "segmentation":
        data_files = [f"{args.data_dir}/coco_segmentation.json"]

    # Check if files exist
    missing_files = [f for f in data_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing GT files: {missing_files}")
        return 1

    try:
        # 1. Load and parse GT slices
        gt_slices = load_all_gt_slices(data_files)
        if not gt_slices:
            print("Error: No GT slices found. Please check your data directory.")
            return 1

        # 2. Load prediction slices
        print(f"Loading predictions from {args.pred_file}...")
        pred_slices = load_pred_slices(args.pred_file)

        # 3. Calculate metric
        results, avg_precision = calculate_precision_at_k(
            gt_slices, pred_slices, args.k
        )

        # 4. Print and save
        print_detailed_results(results, avg_precision, args.k)

        output_file = (
            args.output
            if args.output
            else f"precision_at_{args.k}_{args.task}_results.json"
        )
        save_results(results, avg_precision, args.k, output_file)

    except Exception as e:
        print(f"Evaluation Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
