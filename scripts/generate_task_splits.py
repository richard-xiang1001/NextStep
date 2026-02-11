"""
Generate frozen task splits for WebShop experiments.

This script creates the 5 required splits:
- S_rm_train: For RM data generation/training
- S_rm_dev: RM model selection/calibration
- S_policy_train: For orchestrator RL rollouts
- S_policy_dev: Policy dev tuning
- S_final_test: Final reporting

Usage:
    python scripts/generate_task_splits.py --num_tasks 1000 --output configs/env/webshop_task_manifest_v0.1.0.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def generate_task_splits(
    total_tasks: int,
    rm_train_ratio: float = 0.4,
    rm_dev_ratio: float = 0.1,
    policy_train_ratio: float = 0.3,
    policy_dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Generate task splits with specified ratios.

    Args:
        total_tasks: Total number of tasks
        rm_train_ratio: Ratio for RM training
        rm_dev_ratio: Ratio for RM dev
        policy_train_ratio: Ratio for policy training
        policy_dev_ratio: Ratio for policy dev
        test_ratio: Ratio for final test
        seed: Random seed

    Returns:
        Dictionary mapping split names to task ID lists
    """
    random.seed(seed)

    # Calculate split sizes
    rm_train_size = int(total_tasks * rm_train_ratio)
    rm_dev_size = int(total_tasks * rm_dev_ratio)
    policy_train_size = int(total_tasks * policy_train_ratio)
    policy_dev_size = int(total_tasks * policy_dev_ratio)
    test_size = total_tasks - rm_train_size - rm_dev_size - policy_train_size - policy_dev_size

    # Generate all task IDs
    all_task_ids = [f"task_{i:05d}" for i in range(total_tasks)]

    # Shuffle and split
    random.shuffle(all_task_ids)

    splits = {}
    idx = 0

    splits["S_rm_train"] = all_task_ids[idx : idx + rm_train_size]
    idx += rm_train_size

    splits["S_rm_dev"] = all_task_ids[idx : idx + rm_dev_size]
    idx += rm_dev_size

    splits["S_policy_train"] = all_task_ids[idx : idx + policy_train_size]
    idx += policy_train_size

    splits["S_policy_dev"] = all_task_ids[idx : idx + policy_dev_size]
    idx += policy_dev_size

    splits["S_final_test"] = all_task_ids[idx : idx + test_size]

    return splits


def verify_no_leakage(splits: Dict[str, List[str]]) -> bool:
    """
    Verify that splits have no overlap.

    Args:
        splits: Dictionary of splits

    Returns:
        True if no leakage, False otherwise
    """
    all_sets = {}
    for split_name, task_ids in splits.items():
        all_sets[split_name] = set(task_ids)

    # Check each pair
    for split1 in all_sets:
        for split2 in all_sets:
            if split1 < split2:  # Check each pair once
                overlap = all_sets[split1] & all_sets[split2]
                if overlap:
                    print(f"ERROR: Overlap between {split1} and {split2}: {len(overlap)} tasks")
                    return False

    print("✓ No leakage detected between splits")
    return True


def create_task_manifest(
    splits: Dict[str, List[str]],
    output_path: str,
    benchmark_name: str = "WebShop",
) -> Dict[str, Any]:
    """
    Create task manifest JSON file.

    Args:
        splits: Dictionary of task ID lists
        output_path: Output file path
        benchmark_name: Name of benchmark

    Returns:
        Manifest dictionary
    """
    manifest = {
        "version": "v0.1.0",
        "benchmark": benchmark_name,
        "generated_at": datetime.now().isoformat(),
        "total_tasks": sum(len(ids) for ids in splits.values()),
        "splits": {
            split_name: {
                "purpose": get_split_purpose(split_name),
                "size": len(task_ids),
                "task_ids": task_ids,
                "leakage_control": get_leakage_control(split_name),
            }
            for split_name, task_ids in splits.items()
        },
        "overlap_matrix": {
            split_name: [] for split_name in splits.keys()
        },
        "generation_seed": 42,
        "stratification": "random",
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Task manifest saved to {output_path}")

    return manifest


def get_split_purpose(split_name: str) -> str:
    """Get purpose description for a split."""
    purposes = {
        "S_rm_train": "RM data generation and training only",
        "S_rm_dev": "RM model selection and calibration only",
        "S_policy_train": "Orchestrator RL rollouts and online mining",
        "S_policy_dev": "Policy dev tuning and tau_hack quantile estimation",
        "S_final_test": "Final reporting only",
    }
    return purposes.get(split_name, "Unknown")


def get_leakage_control(split_name: str) -> str:
    """Get leakage control description for a split."""
    controls = {
        "S_rm_train": "Never used for policy training or final testing",
        "S_rm_dev": "No overlap with any other split",
        "S_policy_train": "Never used for final testing",
        "S_policy_dev": "Persisted explicitly, not generated on the fly",
        "S_final_test": "Never used for any training or model selection",
    }
    return controls.get(split_name, "")


def print_split_summary(manifest: Dict[str, Any]):
    """Print summary of task splits."""
    print("\n" + "=" * 60)
    print("TASK SPLIT SUMMARY")
    print("=" * 60)

    for split_name, split_info in manifest["splits"].items():
        print(f"\n{split_name}:")
        print(f"  Size: {split_info['size']}")
        print(f"  Purpose: {split_info['purpose']}")
        print(f"  Leakage Control: {split_info['leakage_control']}")

    total = manifest["total_tasks"]
    print(f"\nTotal tasks: {total}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate task splits for experiments")
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=1000,
        help="Total number of tasks (default: 1000 for small WebShop)"
    )
    parser.add_argument(
        "--rm_train_ratio",
        type=float,
        default=0.4,
        help="Ratio for RM training split"
    )
    parser.add_argument(
        "--rm_dev_ratio",
        type=float,
        default=0.1,
        help="Ratio for RM dev split"
    )
    parser.add_argument(
        "--policy_train_ratio",
        type=float,
        default=0.3,
        help="Ratio for policy training split"
    )
    parser.add_argument(
        "--policy_dev_ratio",
        type=float,
        default=0.1,
        help="Ratio for policy dev split"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio for final test split"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/env/webshop_task_manifest_v0.1.0.json",
        help="Output manifest path"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="WebShop",
        help="Benchmark name"
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = (
        args.rm_train_ratio +
        args.rm_dev_ratio +
        args.policy_train_ratio +
        args.policy_dev_ratio +
        args.test_ratio
    )
    if abs(total_ratio - 1.0) > 0.01:
        print(f"ERROR: Ratios must sum to 1.0, got {total_ratio}")
        return 1

    # Generate splits
    print(f"Generating task splits for {args.num_tasks} tasks...")
    splits = generate_task_splits(
        total_tasks=args.num_tasks,
        rm_train_ratio=args.rm_train_ratio,
        rm_dev_ratio=args.rm_dev_ratio,
        policy_train_ratio=args.policy_train_ratio,
        policy_dev_ratio=args.policy_dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Verify no leakage
    if not verify_no_leakage(splits):
        return 1

    # Create manifest
    manifest = create_task_manifest(
        splits=splits,
        output_path=args.output,
        benchmark_name=args.benchmark,
    )

    # Print summary
    print_split_summary(manifest)

    return 0


if __name__ == "__main__":
    exit(main())
