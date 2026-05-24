"""
Complete When2Tool pipeline for SR-Agents.
Combines dataset splitting, hidden state extraction, linear probe training, and prefill inference.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass, asdict
import sys

from .split_dataset import split_dataset, split_dataset_by_dataset_type
from .extract_hidden_states import extract_hidden_states_for_split, HiddenStateExtractor
from .train_linear_probe import train_probe
from .prefill_inference import run_prefill_inference


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""
    
    # Data
    dataset_path: str
    output_base_dir: str = "./when2tool_outputs"
    
    # Splitting
    train_ratio: float = 0.3
    split_by_dataset_type: bool = False
    random_seed: int = 42
    
    # Model
    model_name: str = "Qwen/Qwen3-4B-Instruct"
    device: str = "cuda"
    api_base: Optional[str] = None  # vLLM API base URL (e.g., http://localhost:8000/v1)
    use_vllm: bool = False  # Use vLLM backend locally
    
    # Feature extraction
    batch_size_extraction: int = 4
    max_length: int = 512
    
    # Linear probe training
    regularization: float = 1.0
    concatenate_layers: bool = True
    validation_split: float = 0.1
    
    # Prefill inference
    prefill_mode: str = "soft"  # soft or hard
    threshold_values: list = None  # List of thresholds to sweep
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    
    # Pipeline steps to run
    steps: str = "1,2,3"  # Comma-separated step numbers
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.threshold_values is None:
            self.threshold_values = [0.1, 0.3, 0.5, 0.7, 0.9]


class When2ToolPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.output_dir = Path(config.output_base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        print(f"Pipeline output directory: {self.output_dir}")
        print(f"Configuration saved to {config_path}")
    
    def run_step_1_split_dataset(self) -> Dict:
        """Step 1: Split dataset into 30/70 train/test."""
        print("\n" + "="*70)
        print("STEP 1: Dataset Splitting (30% train / 70% test)")
        print("="*70)
        
        split_dir = self.output_dir / "splits"
        split_dir.mkdir(exist_ok=True)
        
        if self.config.split_by_dataset_type:
            print("Splitting by dataset type (preserving distribution)...")
            splits = split_dataset_by_dataset_type(
                self.config.dataset_path,
                train_ratio=self.config.train_ratio,
                output_dir=split_dir,
                random_seed=self.config.random_seed,
            )
            
            # Get the combined splits
            train_items = []
            test_items = []
            for ds_type, (train, test) in splits.items():
                train_items.extend(train)
                test_items.extend(test)
        else:
            print("Splitting uniformly...")
            train_items, test_items = split_dataset(
                self.config.dataset_path,
                train_ratio=self.config.train_ratio,
                output_dir=split_dir,
                random_seed=self.config.random_seed,
            )
        
        return {
            "train_size": len(train_items),
            "test_size": len(test_items),
            "train_path": str(split_dir / "train_json.json"),
            "test_path": str(split_dir / "test_json.json"),
        }
    
    def run_step_2_extract_features(self, split_results: Dict) -> Dict:
        """Step 2: Extract hidden states from model."""
        print("\n" + "="*70)
        print("STEP 2: Feature Extraction (Hidden States)")
        if self.config.api_base:
            print(f"Using vLLM API: {self.config.api_base}")
        print("="*70)
        
        features_dir = self.output_dir / "features"
        features_dir.mkdir(exist_ok=True)
        
        extract_hidden_states_for_split(
            self.config.model_name,
            split_results["train_path"],
            split_results["test_path"],
            features_dir,
            batch_size=self.config.batch_size_extraction,
            max_length=self.config.max_length,
            device=self.config.device,
            use_vllm=bool(self.config.api_base),
            api_base=self.config.api_base,
        )
        
        return {
            "features_dir": str(features_dir),
            "train_hidden_states": str(features_dir / "train" / "hidden_states.pt"),
            "test_hidden_states": str(features_dir / "test" / "hidden_states.pt"),
            "train_items": str(features_dir / "train" / "items.json"),
            "test_items": str(features_dir / "test" / "items.json"),
        }
    
    def run_step_3_train_probe(self, feature_results: Dict) -> Dict:
        """Step 3: Train linear probe."""
        print("\n" + "="*70)
        print("STEP 3: Linear Probe Training")
        print("="*70)
        
        probe_dir = self.output_dir / "probe"
        probe_dir.mkdir(exist_ok=True)
        
        # Train probe
        results = train_probe(
            feature_results["train_hidden_states"],
            feature_results["test_hidden_states"],
            feature_results["train_items"],
            probe_dir,
            regularization=self.config.regularization,
            concatenate_layers=self.config.concatenate_layers,
            validation_split=self.config.validation_split,
        )
        
        return {
            "probe_dir": str(probe_dir),
            "probe_path": str(probe_dir / "probe.pkl"),
            "results": results,
        }
    
    def run_step_4_prefill_inference(self, feature_results: Dict, probe_results: Dict) -> Dict:
        """Step 4: Run prefill-guided inference with threshold sweep."""
        print("\n" + "="*70)
        print("STEP 4: Prefill-Guided Inference")
        print("="*70)
        
        inference_dir = self.output_dir / "inference"
        inference_dir.mkdir(exist_ok=True)
        
        all_results = {}
        
        for threshold in self.config.threshold_values:
            print(f"\nRunning inference with threshold={threshold}...")
            
            threshold_dir = inference_dir / f"threshold_{threshold:.1f}"
            threshold_dir.mkdir(exist_ok=True)
            
            results, stats = run_prefill_inference(
                self.config.model_name,
                probe_results["probe_path"],
                feature_results["test_items"],
                feature_results["test_hidden_states"],
                threshold_dir,
                threshold=threshold,
                prefill_mode=self.config.prefill_mode,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                device=self.config.device,
            )
            
            all_results[threshold] = {
                "results_path": str(threshold_dir),
                "stats": stats,
            }
        
        return {
            "inference_dir": str(inference_dir),
            "threshold_results": all_results,
        }
    
    def run(self) -> Dict:
        """Run the complete pipeline."""
        print("\n" + "="*70)
        print("When2Tool Pipeline for SR-Agents")
        print("="*70)
        
        # Parse steps to run
        steps_to_run = set(int(s.strip()) for s in self.config.steps.split(","))
        
        pipeline_results = {}
        
        # Step 1: Split dataset
        if 1 in steps_to_run:
            split_results = self.run_step_1_split_dataset()
            pipeline_results["step1_split"] = split_results
        else:
            # Load existing split
            split_dir = self.output_dir / "splits"
            split_results = {
                "train_path": str(split_dir / "train_json.json"),
                "test_path": str(split_dir / "test_json.json"),
            }
        
        # Step 2: Extract features
        if 2 in steps_to_run:
            feature_results = self.run_step_2_extract_features(split_results)
            pipeline_results["step2_features"] = feature_results
        else:
            # Load existing features
            features_dir = self.output_dir / "features"
            feature_results = {
                "train_hidden_states": str(features_dir / "train" / "hidden_states.pt"),
                "test_hidden_states": str(features_dir / "test" / "hidden_states.pt"),
                "train_items": str(features_dir / "train" / "items.json"),
                "test_items": str(features_dir / "test" / "items.json"),
            }
        
        # Step 3: Train probe
        if 3 in steps_to_run:
            probe_results = self.run_step_3_train_probe(feature_results)
            pipeline_results["step3_probe"] = probe_results
        else:
            # Load existing probe
            probe_dir = self.output_dir / "probe"
            probe_results = {
                "probe_path": str(probe_dir / "probe.pkl"),
            }
        
        # Step 4: Prefill inference (optional - not in original request but useful)
        if 4 in steps_to_run:
            inference_results = self.run_step_4_prefill_inference(feature_results, probe_results)
            pipeline_results["step4_inference"] = inference_results
        
        # Save pipeline results
        results_file = self.output_dir / "pipeline_results.json"
        with open(results_file, "w") as f:
            json.dump(pipeline_results, f, indent=2)
        
        print("\n" + "="*70)
        print("Pipeline Completed!")
        print(f"Results saved to {self.output_dir}")
        print("="*70)
        
        return pipeline_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="When2Tool Pipeline - Dataset splitting, feature extraction, probe training, and prefill inference"
    )
    
    # Required arguments
    parser.add_argument("dataset_path", help="Path to dataset file (JSON or JSONL)")
    
    # Optional arguments
    parser.add_argument("--output-dir", default="./when2tool_outputs", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.3, help="Training data ratio (default: 0.3)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct", help="Model name")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--api-base", default=None, help="vLLM API base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM backend for extraction (requires vllm package)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for extraction")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--steps", default="1,2,3", help="Steps to run (comma-separated)")
    parser.add_argument("--regularization", type=float, default=1.0, help="L2 regularization")
    parser.add_argument("--prefill-mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5], help="Thresholds to sweep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split-by-type", action="store_true", help="Split by dataset type")
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        dataset_path=args.dataset_path,
        output_base_dir=args.output_dir,
        train_ratio=args.train_ratio,
        model_name=args.model,
        device=args.device,
        api_base=args.api_base,
        use_vllm=args.use_vllm,
        batch_size_extraction=args.batch_size,
        max_length=args.max_length,
        steps=args.steps,
        regularization=args.regularization,
        prefill_mode=args.prefill_mode,
        threshold_values=args.thresholds,
        random_seed=args.seed,
        split_by_dataset_type=args.split_by_type,
    )
    
    # Run pipeline
    pipeline = When2ToolPipeline(config)
    results = pipeline.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
