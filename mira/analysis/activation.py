"""
Activation analysis for studying internal representations.

This module provides tools for:
- Extracting and comparing activations across layers
- Identifying critical neurons for safety behavior
- Computing activation statistics and patterns
- Tracking activation changes during attacks
"""

from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class NeuronImportance:
    """Importance score for a neuron."""
    layer_idx: int
    neuron_idx: int
    importance_score: float
    activation_mean: float
    activation_std: float


@dataclass 
class ActivationDiff:
    """Difference between two activation states."""
    layer_idx: int
    l2_distance: float
    cosine_similarity: float
    max_diff: float
    changed_neurons: List[int]


class ActivationAnalyzer:
    """
    Analyzer for studying model activation patterns.
    
    Provides methods for:
    - Extracting layer-wise activations
    - Comparing activations between conditions
    - Identifying important neurons for specific behaviors
    - Computing activation statistics
    """
    
    def __init__(self, model_wrapper):
        """
        Initialize activation analyzer.
        
        Args:
            model_wrapper: ModelWrapper instance
        """
        self.model = model_wrapper
    
    def get_all_activations(
        self,
        text: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Get activations from all specified layers.
        
        Args:
            text: Input text
            layers: Layers to collect (default: all)
            
        Returns:
            Dictionary mapping layer index to activation tensor
        """
        _, cache = self.model.run_with_cache(text)
        
        if layers is None:
            layers = list(range(self.model.n_layers))
        
        return {
            layer_idx: cache.hidden_states.get(layer_idx)
            for layer_idx in layers
            if layer_idx in cache.hidden_states
        }
    
    def get_neuron_activations(
        self,
        text: str,
        layer_idx: int,
        neuron_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Get activations for specific neurons.
        
        Args:
            text: Input text
            layer_idx: Layer to extract from
            neuron_indices: Specific neurons (default: all)
            
        Returns:
            Tensor of neuron activations
        """
        activations = self.get_all_activations(text, [layer_idx])
        hidden = activations.get(layer_idx)
        
        if hidden is None:
            raise ValueError(f"No activations for layer {layer_idx}")
        
        if neuron_indices is not None:
            return hidden[:, :, neuron_indices]
        return hidden
    
    def compare_activations(
        self,
        text1: str,
        text2: str,
        layers: Optional[List[int]] = None,
    ) -> List[ActivationDiff]:
        """
        Compare activations between two inputs.
        
        Args:
            text1: First input
            text2: Second input
            layers: Layers to compare
            
        Returns:
            List of ActivationDiff for each layer
        """
        acts1 = self.get_all_activations(text1, layers)
        acts2 = self.get_all_activations(text2, layers)
        
        diffs = []
        for layer_idx in acts1.keys():
            if layer_idx not in acts2:
                continue
            
            h1 = acts1[layer_idx][0, -1, :]  # Last token
            h2 = acts2[layer_idx][0, -1, :]
            
            # Compute differences
            diff = h1 - h2
            l2_dist = float(diff.norm())
            cos_sim = float(torch.nn.functional.cosine_similarity(
                h1.unsqueeze(0), h2.unsqueeze(0)
            ))
            max_diff = float(diff.abs().max())
            
            # Find neurons with large changes
            threshold = diff.abs().mean() + 2 * diff.abs().std()
            changed = (diff.abs() > threshold).nonzero().squeeze(-1).tolist()
            if isinstance(changed, int):
                changed = [changed]
            
            diffs.append(ActivationDiff(
                layer_idx=layer_idx,
                l2_distance=l2_dist,
                cosine_similarity=cos_sim,
                max_diff=max_diff,
                changed_neurons=changed[:20],  # Top 20
            ))
        
        return diffs
    
    def find_important_neurons(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        layer_idx: int,
        top_k: int = 50,
    ) -> List[NeuronImportance]:
        """
        Find neurons that distinguish positive from negative examples.
        
        Uses mean difference as importance score.
        
        Args:
            positive_texts: Examples of target behavior
            negative_texts: Examples of non-target behavior
            layer_idx: Layer to analyze
            top_k: Number of important neurons to return
            
        Returns:
            List of NeuronImportance sorted by importance
        """
        # Collect activations
        pos_acts = []
        neg_acts = []
        
        for text in positive_texts:
            acts = self.get_all_activations(text, [layer_idx])
            pos_acts.append(acts[layer_idx][0, -1, :])
        
        for text in negative_texts:
            acts = self.get_all_activations(text, [layer_idx])
            neg_acts.append(acts[layer_idx][0, -1, :])
        
        pos_acts = torch.stack(pos_acts)
        neg_acts = torch.stack(neg_acts)
        
        # Compute statistics
        pos_mean = pos_acts.mean(dim=0)
        neg_mean = neg_acts.mean(dim=0)
        pos_std = pos_acts.std(dim=0)
        neg_std = neg_acts.std(dim=0)
        
        # Importance: normalized mean difference
        mean_diff = pos_mean - neg_mean
        pooled_std = torch.sqrt((pos_std**2 + neg_std**2) / 2 + 1e-8)
        importance = (mean_diff.abs() / pooled_std).cpu()
        
        # Get top-k neurons
        top_indices = importance.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            idx = int(idx)
            results.append(NeuronImportance(
                layer_idx=layer_idx,
                neuron_idx=idx,
                importance_score=float(importance[idx]),
                activation_mean=float((pos_mean[idx] + neg_mean[idx]) / 2),
                activation_std=float((pos_std[idx] + neg_std[idx]) / 2),
            ))
        
        return results
    
    def compute_activation_stats(
        self,
        texts: List[str],
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute activation statistics across multiple inputs.
        
        Args:
            texts: List of input texts
            layer_idx: Layer to analyze
            
        Returns:
            Dictionary with mean, std, min, max statistics
        """
        all_acts = []
        
        for text in texts:
            acts = self.get_all_activations(text, [layer_idx])
            all_acts.append(acts[layer_idx][0, -1, :])
        
        all_acts = torch.stack(all_acts)
        
        return {
            "mean": all_acts.mean(dim=0),
            "std": all_acts.std(dim=0),
            "min": all_acts.min(dim=0).values,
            "max": all_acts.max(dim=0).values,
            "median": all_acts.median(dim=0).values,
        }
    
    def activation_trajectory(
        self,
        text: str,
        token_positions: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Track how activations evolve across layers for specific tokens.
        
        Args:
            text: Input text
            token_positions: Which token positions to track
            
        Returns:
            Dictionary mapping layer to activations at each position
        """
        _, cache = self.model.run_with_cache(text)
        
        trajectory = {}
        for layer_idx in range(self.model.n_layers):
            hidden = cache.hidden_states.get(layer_idx)
            if hidden is not None:
                if token_positions is not None:
                    trajectory[layer_idx] = hidden[0, token_positions, :]
                else:
                    trajectory[layer_idx] = hidden[0, :, :]
        
        return trajectory
    
    def residual_stream_contribution(
        self,
        text: str,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze how different components contribute to the residual stream.
        
        Args:
            text: Input text
            layer_idx: Layer to analyze
            
        Returns:
            Component contributions (attention, mlp, etc.)
        """
        _, cache = self.model.run_with_cache(text)
        
        # Get hidden states from consecutive layers
        if layer_idx == 0:
            prev_hidden = self.model.embed_tokens(
                self.model.tokenize(text)["input_ids"]
            )
        else:
            prev_hidden = cache.hidden_states.get(layer_idx - 1)
        
        curr_hidden = cache.hidden_states.get(layer_idx)
        
        if prev_hidden is None or curr_hidden is None:
            raise ValueError(f"Missing activations for layer {layer_idx}")
        
        # Residual contribution
        residual = curr_hidden - prev_hidden
        
        return {
            "input": prev_hidden[0, -1, :],
            "output": curr_hidden[0, -1, :],
            "residual": residual[0, -1, :],
            "residual_norm": float(residual[0, -1, :].norm()),
        }
