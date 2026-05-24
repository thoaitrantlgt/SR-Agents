"""
Train linear probe (logistic regression) on hidden states.
This is Step 2 of the When2Tool pipeline.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm
import pickle


class LinearProbe:
    """Linear probe (logistic regression) for tool-necessity prediction."""
    
    def __init__(
        self,
        n_features: int,
        regularization: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        n_pca_components: Optional[int] = 64,
    ):
        """
        Initialize linear probe.
        
        Args:
            n_features: Number of features (concatenated hidden states from all layers)
            regularization: L2 regularization strength (1/C in sklearn)
            solver: Optimization solver
            max_iter: Maximum iterations
            n_pca_components: Number of PCA components for dimensionality reduction.
                Set to None to disable PCA. Recommended: 32-128 for small datasets.
        """
        self.n_features = n_features
        self.regularization = regularization
        self.n_pca_components = n_pca_components
        
        # Initialize scaler and classifier
        self.scaler = StandardScaler()
        
        # PCA for dimensionality reduction (helps with overfitting when n_features >> n_samples)
        self.pca = PCA(n_components=n_pca_components) if n_pca_components else None
        
        self.classifier = LogisticRegression(
            C=1.0 / regularization,
            solver=solver,
            max_iter=max_iter,
            random_state=42,
            verbose=1,
        )
        
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.1,
    ) -> Dict[str, float]:
        """
        Train linear probe on hidden states.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Binary labels (n_samples,) - 1 for tool-necessary, 0 for tool-unnecessary
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        print(f"Training set shape: {X.shape}")
        print(f"Labels distribution: {np.bincount(y)}")
        
        # Split into train/val
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        print(f"\nTrain set: {X_train.shape}, Labels: {np.bincount(y_train)}")
        print(f"Val set: {X_val.shape}, Labels: {np.bincount(y_val)}")
        
        # Fit scaler on training data
        print("\nFitting scaler...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Apply PCA for dimensionality reduction
        if self.pca is not None:
            # Clamp n_components to min(n_samples, n_features)
            max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
            actual_components = min(self.n_pca_components, max_components)
            if actual_components != self.n_pca_components:
                print(f"  Adjusting PCA components: {self.n_pca_components} → {actual_components} (limited by data size)")
                self.pca = PCA(n_components=actual_components)
            
            print(f"Fitting PCA ({self.n_pca_components} components): {X_train_scaled.shape} → ", end="")
            self.pca.fit(X_train_scaled)
            X_train_scaled = self.pca.transform(X_train_scaled)
            X_val_scaled = self.pca.transform(X_val_scaled)
            explained = self.pca.explained_variance_ratio_.sum()
            print(f"{X_train_scaled.shape}  (explained variance: {explained:.1%})")
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = self.classifier.predict(X_train_scaled)
        y_val_pred = self.classifier.predict(X_val_scaled)
        y_train_proba = self.classifier.predict_proba(X_train_scaled)[:, 1]
        y_val_proba = self.classifier.predict_proba(X_val_scaled)[:, 1]
        
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "train_auc": roc_auc_score(y_train, y_train_proba),
            "val_auc": roc_auc_score(y_val, y_val_proba),
        }
        
        print("\nTraining Results:")
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")
        
        self.is_fitted = True
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict tool necessity.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        predictions = self.classifier.predict(X_scaled)
        probabilities = self.classifier.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save probe to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "scaler": self.scaler,
            "pca": self.pca,
            "classifier": self.classifier,
            "n_features": self.n_features,
            "n_pca_components": self.n_pca_components,
            "is_fitted": self.is_fitted,
        }
        
        with open(output_path, "wb") as f:
            pickle.dump(state, f)
        
        print(f"Probe saved to {output_path}")
    
    @classmethod
    def load(cls, checkpoint_path: Union[str, Path]) -> "LinearProbe":
        """Load probe from disk."""
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        
        probe = cls(
            n_features=state["n_features"],
            n_pca_components=state.get("n_pca_components", None),
        )
        probe.scaler = state["scaler"]
        probe.pca = state.get("pca", None)
        probe.classifier = state["classifier"]
        probe.is_fitted = state["is_fitted"]
        
        print(f"Probe loaded from {checkpoint_path}")
        return probe


def prepare_hidden_states(
    hidden_states: np.ndarray,
    concatenate_layers: bool = True,
) -> np.ndarray:
    """
    Prepare hidden states for training.
    
    Args:
        hidden_states: Array of shape (n_samples, n_layers, hidden_dim)
        concatenate_layers: If True, concatenate all layers; if False, use only last layer
        
    Returns:
        Prepared features of shape (n_samples, n_features)
    """
    if concatenate_layers:
        # Concatenate all layers
        # (n_samples, n_layers, hidden_dim) -> (n_samples, n_layers * hidden_dim)
        n_samples, n_layers, hidden_dim = hidden_states.shape
        X = hidden_states.reshape(n_samples, n_layers * hidden_dim)
    else:
        # Use only last layer
        X = hidden_states[:, -1, :]  # (n_samples, hidden_dim)
    
    print(f"Prepared features shape: {X.shape}")
    return X


def generate_labels_from_evaluation(
    train_eval_results: Dict,
    test_eval_results: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binary labels based on whether tasks can be solved without tools.
    
    Args:
        train_eval_results: Training evaluation results
        test_eval_results: Test evaluation results
        
    Returns:
        Tuple of (train_labels, test_labels)
    """
    # This is a placeholder - in practice, you would run the model without tools
    # and check if it can solve each task, then label accordingly
    raise NotImplementedError(
        "Label generation requires running model evaluation without tools. "
        "See extract_hidden_states.py documentation."
    )


def load_labels_from_eval_file(
    items: list,
    eval_file_path: Union[str, Path] = None,
) -> np.ndarray:
    """
    Extract binary labels indicating if tool is necessary.
    
    Strategy:
    1. If eval_file provided, use it: correct=false → tool_necessary=1, correct=true → tool_necessary=0
    2. Otherwise, use explicit tool_necessary field or infer from skill_annotations
    
    Args:
        items: List of dataset items with instance_id
        eval_file_path: Path to eval JSON file (from noskill inference)
        
    Returns:
        Binary labels array (1 = tool needed, 0 = no tool needed)
    """
    labels = []
    
    # If eval file provided, build mapping
    eval_map = {}
    if eval_file_path and Path(eval_file_path).exists():
        print(f"Loading evaluation results from {eval_file_path}...")
        with open(eval_file_path, 'r') as f:
            eval_data = json.load(f)
            if isinstance(eval_data, dict) and 'details' in eval_data:
                details = eval_data['details']
                if isinstance(details, list):
                    for detail in details:
                        instance_id = detail.get('instance_id')
                        correct = detail.get('correct', True)
                        # If noskill model got it wrong, tool is necessary
                        eval_map[instance_id] = 1 if not correct else 0
                elif isinstance(details, dict):
                    for instance_id, detail in details.items():
                        correct = detail.get('correct', True)
                        eval_map[instance_id] = 1 if not correct else 0
        
        print(f"Loaded eval mapping for {len(eval_map)} instances")
    
    # Extract labels for each item
    for item in items:
        instance_id = item.get('instance_id')
        
        # Try eval file first
        if instance_id in eval_map:
            label = eval_map[instance_id]
        # Fall back to explicit field
        elif 'tool_necessary' in item:
            label = item['tool_necessary']
        # Infer from skill_annotations
        elif 'skill_annotations' in item and item['skill_annotations']:
            label = 1
        else:
            label = 0
        
        labels.append(label)
    
    return np.array(labels)


def train_probe(
    train_hidden_states_path: Union[str, Path],
    test_hidden_states_path: Union[str, Path],
    train_labels_path: Union[str, Path],
    output_dir: Union[str, Path],
    regularization: float = 1.0,
    concatenate_layers: bool = False,
    validation_split: float = 0.1,
    eval_file_path: Union[str, Path] = None,
    n_pca_components: Optional[int] = 64,
) -> Dict:
    """
    Train linear probe on hidden states.
    
    Args:
        train_hidden_states_path: Path to training hidden states (PT file)
        test_hidden_states_path: Path to test hidden states (PT file)
        train_labels_path: Path to training labels (JSON file with tool_necessary field)
        output_dir: Directory to save probe
        regularization: L2 regularization strength
        concatenate_layers: Whether to concatenate all layers (default: False = last layer only)
        validation_split: Validation split ratio
        n_pca_components: PCA components for dim reduction (None to disable, default: 64)
        
    Returns:
        Dictionary with results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hidden states
    print("Loading hidden states...")
    train_hidden_states = torch.load(train_hidden_states_path).numpy()
    test_hidden_states = torch.load(test_hidden_states_path).numpy()
    
    print(f"Train hidden states shape: {train_hidden_states.shape}")
    print(f"Test hidden states shape: {test_hidden_states.shape}")
    
    # Load labels
    print(f"Loading labels from {train_labels_path}...")
    with open(train_labels_path, "r") as f:
        items = json.load(f)
        if isinstance(items, dict):
            items = list(items.values())

    # Extract labels using eval file if available
    train_labels = load_labels_from_eval_file(items, eval_file_path)
    print(f"Labels distribution: {np.bincount(train_labels)}")

    if len(np.unique(train_labels)) < 2:
        raise ValueError(
            f"Training data has only {len(np.unique(train_labels))} class(es). "
            "Need at least 2 classes (tool_necessary=0 and tool_necessary=1) to train probe. "
            "If not using eval file, pass --eval-file <path> to extract labels from evaluation results. "
            "Expected format: JSON with 'details' list/dict containing 'instance_id' and 'correct' fields."
        )
    
    # Prepare features
    print("\nPreparing features...")
    X_train = prepare_hidden_states(train_hidden_states, concatenate_layers)
    X_test = prepare_hidden_states(test_hidden_states, concatenate_layers)
    
    # Train probe
    print("\n" + "="*50)
    print("Training Linear Probe")
    print("="*50)
    
    probe = LinearProbe(
        n_features=X_train.shape[1],
        regularization=regularization,
        n_pca_components=n_pca_components,
    )
    
    metrics = probe.fit(X_train, train_labels, validation_split=validation_split)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Test Set Evaluation")
    print("="*50)
    
    test_pred, test_proba = probe.predict(X_test)
    print(f"Test predictions shape: {test_pred.shape}")
    print(f"Test predictions distribution: {np.bincount(test_pred)}")
    
    # Save results
    results = {
        "train_metrics": metrics,
        "train_hidden_states_shape": list(train_hidden_states.shape),
        "test_hidden_states_shape": list(test_hidden_states.shape),
        "features_shape": list(X_train.shape),
        "concatenate_layers": concatenate_layers,
        "regularization": regularization,
        "n_pca_components": n_pca_components,
    }
    
    # Save probe
    probe_path = output_dir / "probe.pkl"
    probe.save(probe_path)
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProbe saved to {probe_path}")
    print(f"Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train linear probe on hidden states")
    parser.add_argument("--train-hidden", required=True, help="Path to training hidden states")
    parser.add_argument("--test-hidden", required=True, help="Path to test hidden states")
    parser.add_argument("--train-labels", required=True, help="Path to training labels")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--regularization", type=float, default=1.0, help="L2 regularization")
    parser.add_argument("--concat-layers", action="store_true", default=False,
                        help="Concatenate all layers (default: last layer only)")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--pca-components", type=int, default=64,
                        help="PCA components for dim reduction (0 to disable, default: 64)")
    
    args = parser.parse_args()
    
    train_probe(
        args.train_hidden,
        args.test_hidden,
        args.train_labels,
        args.output_dir,
        regularization=args.regularization,
        concatenate_layers=args.concat_layers,
        validation_split=args.validation_split,
        n_pca_components=args.pca_components if args.pca_components > 0 else None,
    )
