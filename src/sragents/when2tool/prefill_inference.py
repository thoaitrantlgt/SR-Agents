"""
Prefill-guided inference using trained probe.
This is Step 3 of the When2Tool pipeline.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Literal
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass

from .train_linear_probe import LinearProbe


@dataclass
class PrefillConfig:
    """Configuration for prefill inference."""
    
    soft_prefill_tool_no: str = "I can solve this directly without using a tool."
    soft_prefill_tool_yes: str = "I need to use a tool for this question."
    hard_prefill_tool_no: str = "\\boxed{"
    hard_prefill_tool_yes: str = '{"name": "'


class PrefillInference:
    """Inference with probe-guided prefilling."""
    
    def __init__(
        self,
        model_name_or_path: str,
        probe_path: Union[str, Path],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
        config: Optional[PrefillConfig] = None,
    ):
        """
        Initialize prefill inference engine.
        
        Args:
            model_name_or_path: Model identifier
            probe_path: Path to trained probe
            device: Device to use
            dtype: Data type
            config: Prefill configuration
        """
        self.device = device
        self.dtype = dtype
        self.config = config or PrefillConfig()
        
        # Load model and tokenizer
        print(f"Loading model {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load probe
        print(f"Loading probe from {probe_path}...")
        self.probe = LinearProbe.load(probe_path)
    
    def predict_tool_necessity(
        self,
        hidden_states: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict tool necessity using probe.
        
        Args:
            hidden_states: Hidden states from model (n_samples, n_layers, hidden_dim)
            threshold: Probability threshold for tool necessity decision
            
        Returns:
            Tuple of (decisions, probabilities, predictions)
            - decisions: Binary decisions based on threshold (1 = need tool, 0 = don't need)
            - probabilities: Raw probabilities from probe
            - predictions: Probe predictions (before thresholding)
        """
        # Prepare features (same as training)
        n_samples, n_layers, hidden_dim = hidden_states.shape
        X = hidden_states.reshape(n_samples, n_layers * hidden_dim)
        
        # Get predictions
        predictions, probabilities = self.probe.predict(X)
        
        # Apply threshold
        decisions = (probabilities >= threshold).astype(int)
        
        return decisions, probabilities, predictions
    
    def get_prefill_text(
        self,
        need_tool: bool,
        prefill_mode: Literal["soft", "hard"] = "soft",
    ) -> str:
        """
        Get prefill text based on tool necessity.
        
        Args:
            need_tool: Whether tool is necessary
            prefill_mode: "soft" (natural language) or "hard" (forced format)
            
        Returns:
            Prefill text to prepend to model output
        """
        if prefill_mode == "soft":
            if need_tool:
                return self.config.soft_prefill_tool_yes
            else:
                return self.config.soft_prefill_tool_no
        elif prefill_mode == "hard":
            if need_tool:
                return self.config.hard_prefill_tool_yes
            else:
                return self.config.hard_prefill_tool_no
        else:
            raise ValueError(f"Unknown prefill mode: {prefill_mode}")
    
    def generate_with_prefill(
        self,
        prompt: str,
        hidden_state: np.ndarray,
        threshold: float = 0.5,
        prefill_mode: Literal["soft", "hard"] = "soft",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Dict:
        """
        Generate response with prefill guidance.
        
        Args:
            prompt: Input prompt
            hidden_state: Hidden state for this prompt (n_layers, hidden_dim)
            threshold: Probe threshold
            prefill_mode: Prefill mode
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary with generation results
        """
        # Predict tool necessity
        hidden_state_batch = hidden_state[np.newaxis, :, :]  # (1, n_layers, hidden_dim)
        decisions, proba, pred = self.predict_tool_necessity(hidden_state_batch, threshold)
        need_tool = decisions[0] == 1
        tool_prob = proba[0]
        
        # Get prefill text
        prefill = self.get_prefill_text(need_tool, prefill_mode)
        
        # Prepare input with prefill
        if prefill_mode == "soft":
            # Soft prefill: natural language guidance (can be overridden)
            full_prompt = prompt + " " + prefill
        else:
            # Hard prefill: forced format (cannot be overridden by model)
            full_prompt = prompt + " " + prefill
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the input)
        prompt_length = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_part = generated_text[prompt_length:].strip()
        
        return {
            "prompt": prompt,
            "full_prompt": full_prompt,
            "generated_text": generated_part,
            "full_response": generated_text,
            "tool_necessary": bool(need_tool),
            "tool_probability": float(tool_prob),
            "prefill": prefill,
            "prefill_mode": prefill_mode,
        }
    
    def batch_generate_with_prefill(
        self,
        prompts: List[str],
        hidden_states: np.ndarray,
        threshold: float = 0.5,
        prefill_mode: Literal["soft", "hard"] = "soft",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[Dict]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of prompts
            hidden_states: Batch of hidden states (n_samples, n_layers, hidden_dim)
            threshold: Probe threshold
            prefill_mode: Prefill mode
            max_new_tokens: Maximum tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating with prefill")):
            result = self.generate_with_prefill(
                prompt,
                hidden_states[i],
                threshold=threshold,
                prefill_mode=prefill_mode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results.append(result)
        
        return results


def run_prefill_inference(
    model_name_or_path: str,
    probe_path: Union[str, Path],
    dataset_path: Union[str, Path],
    hidden_states_path: Union[str, Path],
    output_dir: Union[str, Path],
    threshold: float = 0.5,
    prefill_mode: Literal["soft", "hard"] = "soft",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run prefill-guided inference on a dataset.
    
    Args:
        model_name_or_path: Model identifier
        probe_path: Path to trained probe
        dataset_path: Path to dataset (JSON/JSONL)
        hidden_states_path: Path to pre-extracted hidden states (PT file)
        output_dir: Directory to save results
        threshold: Probe threshold
        prefill_mode: Prefill mode (soft/hard)
        batch_size: Batch size (for organization, not actual batching)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset_path = Path(dataset_path)
    
    if dataset_path.suffix == ".json":
        with open(dataset_path, "r") as f:
            items = json.load(f)
            if isinstance(items, dict):
                items = list(items.values())
    elif dataset_path.suffix == ".jsonl":
        with open(dataset_path, "r") as f:
            items = [json.loads(line) for line in f if line.strip()]
    
    prompts = [item.get("question", "") for item in items]
    print(f"Loaded {len(prompts)} prompts")
    
    # Load hidden states
    print(f"Loading hidden states from {hidden_states_path}...")
    hidden_states = torch.load(hidden_states_path).numpy()
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Initialize inference engine
    print("\nInitializing inference engine...")
    inference = PrefillInference(
        model_name_or_path,
        probe_path,
        device=device,
    )
    
    # Run inference
    print("\n" + "="*50)
    print("Running Prefill-Guided Inference")
    print(f"Prefill mode: {prefill_mode}")
    print(f"Threshold: {threshold}")
    print("="*50 + "\n")
    
    results = inference.batch_generate_with_prefill(
        prompts,
        hidden_states,
        threshold=threshold,
        prefill_mode=prefill_mode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Save results
    output_file = output_dir / f"results_prefill_{prefill_mode}_threshold_{threshold}.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nResults saved to {output_file}")
    
    # Compute statistics
    tool_calls = sum(1 for r in results if r["tool_necessary"])
    tool_call_rate = tool_calls / len(results)
    avg_tool_prob = np.mean([r["tool_probability"] for r in results])
    
    stats = {
        "total_samples": len(results),
        "tool_calls": tool_calls,
        "tool_call_rate": float(tool_call_rate),
        "avg_tool_probability": float(avg_tool_prob),
        "threshold": threshold,
        "prefill_mode": prefill_mode,
    }
    
    stats_file = output_dir / f"stats_prefill_{prefill_mode}_threshold_{threshold}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\nInference Statistics:")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    return results, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run prefill-guided inference")
    parser.add_argument("model_path", help="Model identifier")
    parser.add_argument("--probe", required=True, help="Path to trained probe")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--hidden-states", required=True, help="Path to hidden states")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probe threshold")
    parser.add_argument("--prefill-mode", choices=["soft", "hard"], default="soft")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    run_prefill_inference(
        args.model_path,
        args.probe,
        args.dataset,
        args.hidden_states,
        args.output_dir,
        threshold=args.threshold,
        prefill_mode=args.prefill_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
