"""
Feature Heatmap - Extract and compare features between baseline and attack

Extracts activation features, attention patterns, and safety probe scores
for visualization and analysis of attack success patterns.

Reference: CGC.md research methodology
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayerFeatures:
    """Features extracted from a single layer."""
    layer_idx: int
    residual_norm: float = 0.0
    attn_out_norm: float = 0.0
    mlp_out_norm: float = 0.0
    attn_entropy: float = 0.0
    head_dominance: float = 0.0


@dataclass
class ExtractedFeatures:
    """Complete feature extraction result."""
    prompt: str
    is_attack: bool
    layer_features: List[LayerFeatures] = field(default_factory=list)
    
    # Safety probe features
    refusal_score: float = 0.0
    acceptance_score: float = 0.0
    
    # Generation stability
    output_entropy: float = 0.0
    repetition_ratio: float = 0.0
    token_count: int = 0
    
    # Metadata
    model_name: str = ""
    num_layers: int = 0


class FeatureExtractor:
    """
    Extract features from model for baseline vs attack comparison.
    
    Features are organized into categories:
    - (A) Activation Features: residual, attention, MLP norms per layer
    - (B) Attention Features: entropy, head dominance
    - (C) Safety Probe: refusal/acceptance scores
    - (D) Generation Stability: token entropy, repetition
    """
    
    def __init__(self, model=None, tokenizer=None):
        """
        Initialize feature extractor.
        
        Args:
            model: HuggingFace model (optional, can set later)
            tokenizer: HuggingFace tokenizer (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.activations = {}
    
    def set_model(self, model, tokenizer=None):
        """Set the model for feature extraction."""
        self.model = model
        if tokenizer:
            self.tokenizer = tokenizer
    
    def extract_features(
        self,
        prompt: str,
        is_attack: bool = False,
        response: Optional[str] = None
    ) -> ExtractedFeatures:
        """
        Extract all features for a prompt.
        
        Args:
            prompt: Input prompt
            is_attack: Whether this is an attack prompt
            response: Optional model response for generation stability
            
        Returns:
            ExtractedFeatures with all metrics
        """
        result = ExtractedFeatures(
            prompt=prompt,
            is_attack=is_attack
        )
        
        if self.model is None:
            return result
        
        # Get model config
        config = self.model.config
        num_layers = getattr(config, 'num_hidden_layers', 
                            getattr(config, 'n_layer', 12))
        result.num_layers = num_layers
        result.model_name = getattr(config, '_name_or_path', 'unknown')
        
        # Register hooks for activation capture
        self._register_hooks()
        
        try:
            # Tokenize
            if self.tokenizer:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            else:
                return result
            
            # Forward pass to capture activations
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            
            # Extract layer features
            hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else []
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else []
            
            for layer_idx in range(min(num_layers, len(hidden_states) - 1)):
                layer_feat = LayerFeatures(layer_idx=layer_idx)
                
                # Residual norm (hidden state norm)
                if layer_idx + 1 < len(hidden_states):
                    hs = hidden_states[layer_idx + 1]
                    layer_feat.residual_norm = float(torch.norm(hs).item())
                
                # Attention entropy and head dominance
                if layer_idx < len(attentions):
                    attn = attentions[layer_idx]  # [batch, heads, seq, seq]
                    
                    # Mean attention entropy across heads
                    attn_probs = attn.mean(dim=1)  # Average over heads
                    entropy = -torch.sum(
                        attn_probs * torch.log(attn_probs + 1e-10),
                        dim=-1
                    ).mean()
                    layer_feat.attn_entropy = float(entropy.item())
                    
                    # Head dominance (max attention weight)
                    head_maxes = attn.max(dim=-1)[0].max(dim=-1)[0]  # [batch, heads]
                    layer_feat.head_dominance = float(head_maxes.max().item())
                
                result.layer_features.append(layer_feat)
            
            # Calculate safety probe scores (simple heuristic)
            result.refusal_score = self._estimate_refusal_score(prompt)
            
            # Calculate generation stability if response provided
            if response:
                result.output_entropy = self._calculate_text_entropy(response)
                result.repetition_ratio = self._calculate_repetition(response)
                result.token_count = len(response.split())
            
        finally:
            self._remove_hooks()
        
        return result
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        self.activations = {}
        self.hooks = []
        
        if self.model is None:
            return
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        # Register hooks on transformer layers
        for name, module in self.model.named_modules():
            if 'layer' in name.lower() and 'norm' not in name.lower():
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def _estimate_refusal_score(self, text: str) -> float:
        """Estimate refusal likelihood from text patterns."""
        refusal_keywords = [
            "cannot", "can't", "unable", "sorry", "apologize",
            "as an ai", "ethical", "harmful", "dangerous"
        ]
        text_lower = text.lower()
        matches = sum(1 for kw in refusal_keywords if kw in text_lower)
        return min(matches / 3.0, 1.0)  # Normalize to 0-1
    
    def _calculate_text_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        from collections import Counter
        freq = Counter(text.lower())
        total = sum(freq.values())
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _calculate_repetition(self, text: str) -> float:
        """Calculate repetition ratio."""
        words = text.split()
        if len(words) < 5:
            return 0.0
        from collections import Counter
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        if not trigrams:
            return 0.0
        counter = Counter(trigrams)
        repeated = sum(count - 1 for count in counter.values() if count > 1)
        return repeated / len(trigrams)
    
    def compute_delta(
        self,
        baseline: ExtractedFeatures,
        attack: ExtractedFeatures
    ) -> Dict[str, Any]:
        """
        Compute difference between attack and baseline features.
        
        Args:
            baseline: Features from clean prompt
            attack: Features from attack prompt
            
        Returns:
            Dict with delta values for each feature
        """
        delta = {
            "layer_deltas": [],
            "refusal_delta": attack.refusal_score - baseline.refusal_score,
            "entropy_delta": attack.output_entropy - baseline.output_entropy,
            "repetition_delta": attack.repetition_ratio - baseline.repetition_ratio,
        }
        
        # Compute per-layer deltas
        num_layers = min(len(baseline.layer_features), len(attack.layer_features))
        for i in range(num_layers):
            base_layer = baseline.layer_features[i]
            atk_layer = attack.layer_features[i]
            
            layer_delta = {
                "layer": i,
                "residual_delta": atk_layer.residual_norm - base_layer.residual_norm,
                "attn_entropy_delta": atk_layer.attn_entropy - base_layer.attn_entropy,
                "head_dominance_delta": atk_layer.head_dominance - base_layer.head_dominance,
            }
            delta["layer_deltas"].append(layer_delta)
        
        return delta
    
    def to_heatmap_data(
        self,
        features_list: List[ExtractedFeatures],
        labels: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Convert features to heatmap-ready format.
        
        Args:
            features_list: List of ExtractedFeatures
            labels: Optional labels for each feature set
            
        Returns:
            Dict with numpy arrays for heatmap plotting
        """
        if not features_list:
            return {}
        
        num_samples = len(features_list)
        num_layers = max(len(f.layer_features) for f in features_list)
        
        # Initialize arrays
        residual_norms = np.zeros((num_samples, num_layers))
        attn_entropies = np.zeros((num_samples, num_layers))
        head_dominances = np.zeros((num_samples, num_layers))
        
        for i, features in enumerate(features_list):
            for j, layer_feat in enumerate(features.layer_features):
                residual_norms[i, j] = layer_feat.residual_norm
                attn_entropies[i, j] = layer_feat.attn_entropy
                head_dominances[i, j] = layer_feat.head_dominance
        
        return {
            "residual_norms": residual_norms,
            "attn_entropies": attn_entropies,
            "head_dominances": head_dominances,
            "labels": labels or [f"Sample {i}" for i in range(num_samples)],
            "layer_indices": list(range(num_layers)),
        }
