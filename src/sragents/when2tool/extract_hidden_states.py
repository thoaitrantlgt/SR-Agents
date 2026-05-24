"""
Extract hidden states from model at last token position.
This is Step 1 of the When2Tool pipeline.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class HiddenStateExtractor:
    """Extract hidden states from language models."""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
        use_vllm: bool = False,
        api_base: Optional[str] = None,
    ):
        """
        Initialize the hidden state extractor.
        
        Args:
            model_name_or_path: Model identifier from HuggingFace
            device: Device to use (cuda/cpu)
            dtype: Data type for model (float32/float16)
            use_vllm: Use vLLM backend instead of transformers
            api_base: vLLM API base URL (for future use)
        """
        self.device = device
        
        # Automatically select float16/bfloat16 on CUDA to save 50% VRAM and prevent NVML/OOM failures
        if device == "cuda" and dtype == torch.float32:
            try:
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                    print("Automatically setting dtype to bfloat16 to optimize VRAM usage on CUDA")
                else:
                    dtype = torch.float16
                    print("Automatically setting dtype to float16 to optimize VRAM usage on CUDA")
            except Exception:
                dtype = torch.float16
                print("Automatically setting dtype to float16 to optimize VRAM usage on CUDA")
                
        self.dtype = dtype
        self.use_vllm = use_vllm
        self.api_base = api_base
        
        if use_vllm:
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM not installed. Install with: pip install vllm\n"
                    "Or use transformers backend (default) by removing --use-vllm flag"
                )
            print(f"Loading model {model_name_or_path} using vLLM...")
            self.model = LLM(
                model=model_name_or_path,
                dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
            )
            self.tokenizer = self.model.get_tokenizer()
        else:
            print(f"Loading model {model_name_or_path} using transformers...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
            )
            self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract_hidden_states(
        self,
        texts: List[str],
        batch_size: int = 4,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Extract hidden states at last token position for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Array of shape (n_texts, n_layers, hidden_dim)
        """
        all_hidden_states = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Extract hidden states at last token position
            # outputs.hidden_states is a tuple of (batch_size, seq_len, hidden_dim) for each layer
            hidden_states = outputs.hidden_states  # tuple of length num_layers
            
            # Get attention masks to find actual last token positions
            attention_mask = inputs["attention_mask"]
            
            # Extract last token hidden states for each sample in batch
            batch_hidden_states = []
            for j in range(len(batch_texts)):
                # Find actual sequence length (last non-padding token)
                seq_len = attention_mask[j].sum().item()
                
                # Get hidden states from all layers at last position
                layer_hidden_states = []
                for layer_idx in range(len(hidden_states)):
                    # hidden_states[layer_idx] shape: (batch_size, seq_len, hidden_dim)
                    last_hidden = hidden_states[layer_idx][j, seq_len - 1, :]  # (hidden_dim,)
                    last_hidden = last_hidden.detach().cpu()
                    if last_hidden.dtype in (torch.bfloat16, torch.float16):
                        last_hidden = last_hidden.to(torch.float32)
                    layer_hidden_states.append(last_hidden.numpy())
                
                # Stack layers: (n_layers, hidden_dim)
                batch_hidden_states.append(np.stack(layer_hidden_states))
            
            all_hidden_states.extend(batch_hidden_states)
        
        # Stack all: (n_texts, n_layers, hidden_dim)
        return np.stack(all_hidden_states)
    
    def extract_and_save(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        question_key: str = "question",
        batch_size: int = 4,
        max_length: int = 512,
        save_metadata: bool = True,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract hidden states from dataset and save to files.
        
        Args:
            dataset_path: Path to dataset JSON/JSONL file
            output_dir: Directory to save hidden states and metadata
            question_key: Key for question text in dataset items
            batch_size: Batch size for extraction
            max_length: Maximum sequence length
            save_metadata: Whether to save metadata
            
        Returns:
            Tuple of (hidden_states array, dataset items)
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
        else:
            raise ValueError(f"Unsupported format: {dataset_path.suffix}")
        
        print(f"Loaded {len(items)} items")
        
        # Extract questions
        texts = [item.get(question_key, "") for item in items]
        
        # Extract hidden states
        print(f"Extracting hidden states...")
        hidden_states = self.extract_hidden_states(texts, batch_size, max_length)
        
        print(f"Hidden states shape: {hidden_states.shape}")
        
        # Save hidden states
        hidden_states_path = output_dir / "hidden_states.pt"
        torch.save(torch.from_numpy(hidden_states), hidden_states_path)
        print(f"Saved hidden states to {hidden_states_path}")
        
        # Save metadata if requested
        if save_metadata:
            metadata = {
                "num_items": len(items),
                "num_layers": hidden_states.shape[1],
                "hidden_dim": hidden_states.shape[2],
                "model": self.model.config.model_type,
            }
            
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved metadata to {metadata_path}")
            
            # Save dataset items (for reference)
            items_path = output_dir / "items.json"
            with open(items_path, "w") as f:
                json.dump(items, f, indent=2)
            print(f"Saved items to {items_path}")
        
        return hidden_states, items


def extract_hidden_states_for_split(
    model_name_or_path: str,
    train_data_path: Union[str, Path],
    test_data_path: Union[str, Path],
    output_dir: Union[str, Path],
    batch_size: int = 4,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_vllm: bool = False,
    api_base: Optional[str] = None,
):
    """
    Extract hidden states for both train and test splits.
    
    Args:
        model_name_or_path: Model identifier
        train_data_path: Path to training data
        test_data_path: Path to test data
        output_dir: Directory to save extracted states
        batch_size: Batch size
        max_length: Maximum sequence length
        device: Device to use
        use_vllm: Use vLLM backend
        api_base: vLLM API base URL
    """
    extractor = HiddenStateExtractor(
        model_name_or_path,
        device=device,
        use_vllm=use_vllm,
        api_base=api_base,
    )
    
    # Extract for train set
    print("\n" + "="*50)
    print("Extracting hidden states for TRAINING set")
    print("="*50)
    train_dir = Path(output_dir) / "train"
    train_hidden_states, train_items = extractor.extract_and_save(
        train_data_path,
        train_dir,
        batch_size=batch_size,
        max_length=max_length,
    )
    
    # Extract for test set
    print("\n" + "="*50)
    print("Extracting hidden states for TEST set")
    print("="*50)
    test_dir = Path(output_dir) / "test"
    test_hidden_states, test_items = extractor.extract_and_save(
        test_data_path,
        test_dir,
        batch_size=batch_size,
        max_length=max_length,
    )
    
    print("\n" + "="*50)
    print("Hidden state extraction completed!")
    print(f"Train: {train_hidden_states.shape}")
    print(f"Test: {test_hidden_states.shape}")
    print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract hidden states from model")
    parser.add_argument("model_path", help="Model identifier from HuggingFace")
    parser.add_argument("--train-data", required=True, help="Path to training data")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    extract_hidden_states_for_split(
        args.model_path,
        args.train_data,
        args.test_data,
        args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
