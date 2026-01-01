"""
Subspace analysis for identifying refusal and acceptance directions.

This module implements techniques for:
- Identifying distinct activation subspaces for safe vs harmful inputs
- Computing refusal/acceptance directions using linear methods
- Measuring distances between embeddings and subspaces
- Training linear probes for classification
"""

from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass
import torch
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class SubspaceResult:
    """Results from subspace analysis."""
    
    # Primary direction separating safe from harmful
    refusal_direction: torch.Tensor
    acceptance_direction: torch.Tensor
    
    # Full subspace basis vectors
    refusal_basis: Optional[torch.Tensor] = None
    acceptance_basis: Optional[torch.Tensor] = None
    
    # Linear probe if trained
    probe_weights: Optional[torch.Tensor] = None
    probe_bias: Optional[float] = None
    probe_accuracy: Optional[float] = None
    
    # Analysis metadata
    layer_idx: Optional[int] = None
    token_position: str = "last"  # "last", "first", or "mean"


class SubspaceAnalyzer:
    """
    Analyzer for identifying and working with activation subspaces.
    
    This class enables:
    - Identification of refusal/acceptance directions
    - Subspace distance computation
    - Linear probe training and evaluation
    - Projection operations for intervention
    """
    
    def __init__(
        self,
        model_wrapper,
        layer_idx: Optional[int] = None,
        n_components: int = 64,
        token_position: str = "last",
    ):
        """
        Initialize subspace analyzer.
        
        Args:
            model_wrapper: ModelWrapper instance
            layer_idx: Layer to analyze (default: middle layer)
            n_components: Number of PCA components for dimensionality reduction
            token_position: Which token position to use ("last", "first", "mean")
        """
        self.model = model_wrapper
        self.layer_idx = layer_idx or (model_wrapper.n_layers // 2)
        self.n_components = n_components
        self.token_position = token_position
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(n_components, model_wrapper.hidden_size))
    
    def collect_activations(
        self,
        texts: List[str],
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Collect activations for a list of texts.
        
        Args:
            texts: List of input texts
            layer_idx: Layer to collect from (uses default if None)
            
        Returns:
            Tensor of shape (num_texts, hidden_size)
        """
        layer_idx = layer_idx or self.layer_idx
        activations = []
        
        for text in texts:
            _, cache = self.model.run_with_cache(text)
            hidden = cache.hidden_states.get(layer_idx)
            
            if hidden is None:
                raise ValueError(f"No activations found for layer {layer_idx}")
            
            # Extract based on token position
            if self.token_position == "last":
                act = hidden[0, -1, :]  # Last token of first batch item
            elif self.token_position == "first":
                act = hidden[0, 0, :]
            elif self.token_position == "mean":
                act = hidden[0].mean(dim=0)
            else:
                raise ValueError(f"Unknown token position: {self.token_position}")
            
            activations.append(act.cpu())
        
        return torch.stack(activations)
    
    def identify_subspaces(
        self,
        safe_prompts: List[str],
        unsafe_prompts: List[str],
        layer_idx: Optional[int] = None,
    ) -> SubspaceResult:
        """
        Identify refusal and acceptance subspaces.
        
        Uses mean difference direction as the primary separating direction,
        and PCA to find additional basis vectors.
        
        Args:
            safe_prompts: List of safe/benign prompts
            unsafe_prompts: List of unsafe/harmful prompts
            layer_idx: Layer to analyze
            
        Returns:
            SubspaceResult with identified directions and subspaces
        """
        layer_idx = layer_idx or self.layer_idx
        
        # Collect activations
        safe_acts = self.collect_activations(safe_prompts, layer_idx)
        unsafe_acts = self.collect_activations(unsafe_prompts, layer_idx)
        
        # Compute mean difference direction
        safe_mean = safe_acts.mean(dim=0)
        unsafe_mean = unsafe_acts.mean(dim=0)
        
        # Refusal direction: from safe toward unsafe (model learns to refuse unsafe)
        refusal_direction = unsafe_mean - safe_mean
        refusal_direction = refusal_direction / refusal_direction.norm()
        
        # Acceptance direction is opposite
        acceptance_direction = -refusal_direction
        
        # Compute subspace bases using PCA
        all_acts = torch.cat([safe_acts, unsafe_acts], dim=0).numpy()
        all_acts_scaled = self.scaler.fit_transform(all_acts)
        
        # Sanitize NaN/Inf before PCA (critical for float16 models)
        if np.any(np.isnan(all_acts_scaled)) or np.any(np.isinf(all_acts_scaled)):
            all_acts_scaled = np.nan_to_num(all_acts_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.pca.fit(all_acts_scaled)
        
        # Project each class to get class-specific subspaces
        n_basis = min(16, self.n_components)
        
        # Safe subspace: top principal components of safe activations
        safe_scaled = self.scaler.transform(safe_acts.numpy())
        safe_centered = safe_scaled - safe_scaled.mean(axis=0)
        safe_pca = PCA(n_components=n_basis)
        safe_pca.fit(safe_centered)
        acceptance_basis = torch.tensor(safe_pca.components_, dtype=torch.float32)
        
        # Unsafe subspace: top principal components of unsafe activations
        unsafe_scaled = self.scaler.transform(unsafe_acts.numpy())
        unsafe_centered = unsafe_scaled - unsafe_scaled.mean(axis=0)
        unsafe_pca = PCA(n_components=n_basis)
        unsafe_pca.fit(unsafe_centered)
        refusal_basis = torch.tensor(unsafe_pca.components_, dtype=torch.float32)
        
        return SubspaceResult(
            refusal_direction=refusal_direction,
            acceptance_direction=acceptance_direction,
            refusal_basis=refusal_basis,
            acceptance_basis=acceptance_basis,
            layer_idx=layer_idx,
            token_position=self.token_position,
        )
    
    def train_probe(
        self,
        safe_prompts: List[str],
        unsafe_prompts: List[str],
        layer_idx: Optional[int] = None,
    ) -> SubspaceResult:
        """
        Train a linear probe to classify safe vs unsafe activations.
        
        This provides a more principled way to find the decision boundary.
        
        Args:
            safe_prompts: List of safe prompts (label 0)
            unsafe_prompts: List of unsafe prompts (label 1)
            layer_idx: Layer to analyze
            
        Returns:
            SubspaceResult with probe weights and direction
        """
        layer_idx = layer_idx or self.layer_idx
        
        # Collect activations
        safe_acts = self.collect_activations(safe_prompts, layer_idx)
        unsafe_acts = self.collect_activations(unsafe_prompts, layer_idx)
        
        # Prepare training data
        X = torch.cat([safe_acts, unsafe_acts], dim=0).numpy()
        y = np.array([0] * len(safe_prompts) + [1] * len(unsafe_prompts))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Sanitize NaN/Inf values (can occur with float16 precision)
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("  ⚠ Warning: NaN/Inf in activations, sanitizing...")
            # Replace NaN with 0, Inf with large finite values
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Train logistic regression probe
        probe = LogisticRegression(max_iter=1000, class_weight="balanced")
        probe.fit(X_scaled, y)
        
        accuracy = probe.score(X_scaled, y)
        
        # Extract decision direction from probe weights
        # The weight vector points toward class 1 (unsafe)
        probe_weights = torch.tensor(probe.coef_[0], dtype=torch.float32)
        probe_bias = float(probe.intercept_[0])
        
        # Normalize to get direction
        refusal_direction = probe_weights / probe_weights.norm()
        acceptance_direction = -refusal_direction
        
        return SubspaceResult(
            refusal_direction=refusal_direction,
            acceptance_direction=acceptance_direction,
            probe_weights=probe_weights,
            probe_bias=probe_bias,
            probe_accuracy=accuracy,
            layer_idx=layer_idx,
            token_position=self.token_position,
        )
    
    def compute_distance(
        self,
        activation: torch.Tensor,
        subspace_result: SubspaceResult,
        direction: str = "refusal",
    ) -> float:
        """
        Compute distance/projection along a direction.
        
        Args:
            activation: Activation tensor to measure
            subspace_result: Previously computed subspace result
            direction: "refusal" or "acceptance"
            
        Returns:
            Signed distance along the direction
        """
        if direction == "refusal":
            dir_vec = subspace_result.refusal_direction
        elif direction == "acceptance":
            dir_vec = subspace_result.acceptance_direction
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Handle different activation shapes
        if activation.dim() == 1:
            act = activation
        elif activation.dim() == 2:
            if self.token_position == "last":
                act = activation[-1]
            elif self.token_position == "first":
                act = activation[0]
            else:
                act = activation.mean(dim=0)
        elif activation.dim() == 3:
            if self.token_position == "last":
                act = activation[0, -1]
            elif self.token_position == "first":
                act = activation[0, 0]
            else:
                act = activation[0].mean(dim=0)
        else:
            raise ValueError(f"Unexpected activation shape: {activation.shape}")
        
        # Compute projection
        act = act.to(dir_vec.device)
        return float(torch.dot(act, dir_vec))
    
    def project_to_subspace(
        self,
        activation: torch.Tensor,
        subspace_result: SubspaceResult,
        target: str = "acceptance",
    ) -> torch.Tensor:
        """
        Project activation onto a subspace.
        
        Args:
            activation: Activation to project
            subspace_result: Subspace definitions
            target: "acceptance" or "refusal"
            
        Returns:
            Projected activation
        """
        if target == "acceptance":
            basis = subspace_result.acceptance_basis
        elif target == "refusal":
            basis = subspace_result.refusal_basis
        else:
            raise ValueError(f"Unknown target: {target}")
        
        if basis is None:
            raise ValueError(f"No basis vectors for {target} subspace")
        
        # Project: sum of projections onto each basis vector
        basis = basis.to(activation.device)
        projections = torch.matmul(activation, basis.T)  # (seq, n_basis)
        reconstructed = torch.matmul(projections, basis)  # (seq, hidden)
        
        return reconstructed
    
    def steering_vector(
        self,
        subspace_result: SubspaceResult,
        target: str = "acceptance",
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Get a steering vector to move toward target subspace.
        
        Args:
            subspace_result: Subspace definitions
            target: "acceptance" or "refusal"
            strength: Magnitude of steering
            
        Returns:
            Steering vector
        """
        if target == "acceptance":
            direction = subspace_result.acceptance_direction
        elif target == "refusal":
            direction = subspace_result.refusal_direction
        else:
            raise ValueError(f"Unknown target: {target}")
        
        return strength * direction
    
    def save(self, path: str, subspace_result: SubspaceResult) -> None:
        """Save subspace result to file."""
        data = {
            "refusal_direction": subspace_result.refusal_direction,
            "acceptance_direction": subspace_result.acceptance_direction,
            "refusal_basis": subspace_result.refusal_basis,
            "acceptance_basis": subspace_result.acceptance_basis,
            "probe_weights": subspace_result.probe_weights,
            "probe_bias": subspace_result.probe_bias,
            "probe_accuracy": subspace_result.probe_accuracy,
            "layer_idx": subspace_result.layer_idx,
            "token_position": subspace_result.token_position,
        }
        torch.save(data, path)
    
    def load(self, path: str) -> SubspaceResult:
        """Load subspace result from file."""
        data = torch.load(path)
        return SubspaceResult(**data)
