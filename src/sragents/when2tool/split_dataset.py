"""
Dataset splitting utility for 30/70 train/test split.
Handles multiple dataset formats (JSON, JSONL).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union
from tqdm import tqdm


def load_dataset(data_path: Union[str, Path]) -> List[Dict]:
    """
    Load dataset from JSON or JSONL file.
    
    Args:
        data_path: Path to dataset file (JSON or JSONL)
        
    Returns:
        List of dataset items
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    items = []
    
    if data_path.suffix == ".json":
        with open(data_path, "r") as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = list(data.values()) if data else []
    elif data_path.suffix == ".jsonl":
        with open(data_path, "r") as f:
            items = [json.loads(line.strip()) for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    return items


def save_dataset(items: List[Dict], output_path: Union[str, Path], format: str = "json") -> None:
    """
    Save dataset to file.
    
    Args:
        items: List of dataset items
        output_path: Path to save dataset
        format: Output format ("json" or "jsonl")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, "w") as f:
            json.dump(items, f, indent=2)
    elif format == "jsonl":
        with open(output_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
    else:
        raise ValueError(f"Unsupported format: {format}")


def split_dataset(
    data_path: Union[str, Path],
    train_ratio: float = 0.3,
    test_ratio: float = 0.7,
    random_seed: int = 42,
    output_dir: Union[str, Path] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train/test sets with specified ratios.
    
    Args:
        data_path: Path to dataset file
        train_ratio: Ratio of training data (default: 0.3 for 30%)
        test_ratio: Ratio of test data (default: 0.7 for 70%)
        random_seed: Random seed for reproducibility
        output_dir: If provided, save split datasets to this directory
        
    Returns:
        Tuple of (train_items, test_items)
    """
    if abs((train_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + test_ratio}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    items = load_dataset(data_path)
    print(f"Total items loaded: {len(items)}")
    
    # Shuffle and split
    shuffled_items = items.copy()
    random.shuffle(shuffled_items)
    
    split_idx = int(len(shuffled_items) * train_ratio)
    train_items = shuffled_items[:split_idx]
    test_items = shuffled_items[split_idx:]
    
    print(f"Train set: {len(train_items)} items ({len(train_items)/len(items)*100:.1f}%)")
    print(f"Test set: {len(test_items)} items ({len(test_items)/len(items)*100:.1f}%)")
    
    # Save splits if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine format from input file
        input_format = Path(data_path).suffix[1:]  # Remove leading dot
        
        train_path = output_dir / f"train_{input_format}.{input_format}"
        test_path = output_dir / f"test_{input_format}.{input_format}"
        
        print(f"\nSaving train set to {train_path}...")
        save_dataset(train_items, train_path, format=input_format)
        
        print(f"Saving test set to {test_path}...")
        save_dataset(test_items, test_path, format=input_format)
    
    return train_items, test_items


def split_dataset_by_dataset_type(
    data_path: Union[str, Path],
    train_ratio: float = 0.3,
    output_dir: Union[str, Path] = None,
    dataset_type_key: str = "dataset",
    random_seed: int = 42,
) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """
    Split dataset while preserving the distribution of dataset types.
    Useful for benchmarks with multiple dataset types.
    
    Args:
        data_path: Path to dataset file
        train_ratio: Ratio of training data per dataset type
        output_dir: If provided, save split datasets
        dataset_type_key: Key to identify dataset type in items
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict mapping dataset type to (train_items, test_items) tuples
    """
    random.seed(random_seed)
    
    print(f"Loading dataset from {data_path}...")
    items = load_dataset(data_path)
    print(f"Total items loaded: {len(items)}")
    
    # Group by dataset type
    dataset_groups = {}
    for item in items:
        ds_type = item.get(dataset_type_key, "unknown")
        if ds_type not in dataset_groups:
            dataset_groups[ds_type] = []
        dataset_groups[ds_type].append(item)
    
    print(f"\nDataset types: {list(dataset_groups.keys())}")
    for ds_type, items_list in dataset_groups.items():
        print(f"  {ds_type}: {len(items_list)} items")
    
    # Split each dataset type
    splits = {}
    all_train = []
    all_test = []
    
    for ds_type, items_list in dataset_groups.items():
        shuffled = items_list.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]
        
        splits[ds_type] = (train, test)
        all_train.extend(train)
        all_test.extend(test)
        
        print(f"\n{ds_type}:")
        print(f"  Train: {len(train)} items ({len(train)/len(items_list)*100:.1f}%)")
        print(f"  Test: {len(test)} items ({len(test)/len(items_list)*100:.1f}%)")
    
    # Save splits if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_format = Path(data_path).suffix[1:]
        
        train_path = output_dir / f"train_split.{input_format}"
        test_path = output_dir / f"test_split.{input_format}"
        
        print(f"\nSaving combined train set ({len(all_train)} items) to {train_path}...")
        save_dataset(all_train, train_path, format=input_format)
        
        print(f"Saving combined test set ({len(all_test)} items) to {test_path}...")
        save_dataset(all_test, test_path, format=input_format)
        
        # Also save per-type splits
        per_type_dir = output_dir / "by_type"
        per_type_dir.mkdir(exist_ok=True)
        
        for ds_type, (train, test) in splits.items():
            train_path = per_type_dir / f"{ds_type}_train.{input_format}"
            test_path = per_type_dir / f"{ds_type}_test.{input_format}"
            save_dataset(train, train_path, format=input_format)
            save_dataset(test, test_path, format=input_format)
            print(f"Saved {ds_type}: {train_path}, {test_path}")
    
    return splits


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train/test sets")
    parser.add_argument("data_path", help="Path to dataset file (JSON or JSONL)")
    parser.add_argument("--train-ratio", type=float, default=0.3, help="Training ratio (default: 0.3)")
    parser.add_argument("--output-dir", help="Directory to save split datasets")
    parser.add_argument("--by-type", action="store_true", help="Split by dataset type (preserve distribution)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.by_type:
        split_dataset_by_dataset_type(
            args.data_path,
            train_ratio=args.train_ratio,
            output_dir=args.output_dir,
            random_seed=args.seed,
        )
    else:
        split_dataset(
            args.data_path,
            train_ratio=args.train_ratio,
            output_dir=args.output_dir,
            random_seed=args.seed,
        )
