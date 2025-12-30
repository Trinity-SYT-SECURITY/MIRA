"""
Logit projection analysis for visualizing prediction formation across layers.

Projects intermediate layer hidden states to vocabulary space to track
how model predictions evolve from input to output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from ..core.hooks import ActivationHookManager, ActivationCache, get_unembedding_matrix


@dataclass
class LayerPrediction:
    """Prediction state at a single layer."""
    layer_idx: int
    top_tokens: List[int]
    top_probs: List[float]
    entropy: float
    max_prob: float
    logits: Optional[torch.Tensor] = None


@dataclass
class PredictionTrajectory:
    """Full prediction trajectory across all layers."""
    input_text: str
    layer_predictions: List[LayerPrediction]
    final_prediction: LayerPrediction
    
    def get_token_probability_curve(self, token_id: int) -> List[float]:
        """Get probability of specific token across all layers."""
        curve = []
        for lp in self.layer_predictions:
            if token_id in lp.top_tokens:
                idx = lp.top_tokens.index(token_id)
                curve.append(lp.top_probs[idx])
            else:
                curve.append(0.0)
        return curve
    
    def find_transition_layer(self, threshold: float = 0.5) -> Optional[int]:
        """Find layer where top prediction probability exceeds threshold."""
        for lp in self.layer_predictions:
            if lp.max_prob >= threshold:
                return lp.layer_idx
        return None


class LogitProjector:
    """
    Projects hidden states to vocabulary logits at any layer.
    
    Enables visualization of how predictions form across layers,
    useful for understanding where attacks succeed in changing model behavior.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        tokenizer: Any = None,
        use_tuned_transforms: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Get unembedding matrix
        self.unembed = get_unembedding_matrix(model)
        self.vocab_size = self.unembed.shape[0]
        
        # Optional layer-specific transforms
        self.tuned_transforms: Dict[int, nn.Linear] = {}
        self.use_tuned = use_tuned_transforms
        
        # Layer normalization (if exists)
        self.final_ln = self._get_final_layernorm()
    
    def _get_final_layernorm(self) -> Optional[nn.Module]:
        """Get final layer normalization module."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
            return self.model.transformer.ln_f
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            return self.model.model.norm
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'final_layer_norm'):
            return self.model.gpt_neox.final_layer_norm
        return None
    
    def project_to_vocab(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int = -1,
        apply_ln: bool = True,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Project hidden state to vocabulary logits.
        
        Args:
            hidden_state: Hidden state tensor [batch, seq_len, hidden_dim]
            layer_idx: Which layer this came from (-1 for final)
            apply_ln: Whether to apply layer normalization
            temperature: Temperature for softmax
            
        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        h = hidden_state.to(self.device)
        
        # Apply tuned transform if available
        if self.use_tuned and layer_idx in self.tuned_transforms:
            h = self.tuned_transforms[layer_idx](h)
        
        # Apply layer normalization
        if apply_ln and self.final_ln is not None:
            h = self.final_ln(h)
        
        # Project to vocabulary
        logits = h @ self.unembed.T
        
        if temperature != 1.0:
            logits = logits / temperature
        
        return logits
    
    def get_layer_prediction(
        self,
        hidden_state: torch.Tensor,
        layer_idx: int,
        position: int = -1,
        top_k: int = 10
    ) -> LayerPrediction:
        """
        Get prediction at specific layer for specific position.
        
        Args:
            hidden_state: Hidden state from layer
            layer_idx: Layer index
            position: Token position (-1 for last)
            top_k: Number of top tokens to return
            
        Returns:
            LayerPrediction with top tokens and probabilities
        """
        logits = self.project_to_vocab(hidden_state, layer_idx)
        
        # Get logits for specific position
        if position == -1:
            pos_logits = logits[0, -1, :]
        else:
            pos_logits = logits[0, position, :]
        
        probs = F.softmax(pos_logits, dim=-1)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        return LayerPrediction(
            layer_idx=layer_idx,
            top_tokens=top_indices.tolist(),
            top_probs=top_probs.tolist(),
            entropy=entropy,
            max_prob=top_probs[0].item(),
            logits=pos_logits.detach(),
        )
    
    def analyze_trajectory(
        self,
        input_ids: torch.Tensor,
        activations: ActivationCache,
        position: int = -1,
        top_k: int = 10
    ) -> PredictionTrajectory:
        """
        Analyze prediction trajectory across all layers.
        
        Args:
            input_ids: Input token IDs
            activations: Cached activations from forward pass
            position: Token position to analyze
            top_k: Number of top tokens per layer
            
        Returns:
            PredictionTrajectory with per-layer predictions
        """
        # Decode input text
        if self.tokenizer:
            input_text = self.tokenizer.decode(input_ids[0])
        else:
            input_text = f"[{len(input_ids[0])} tokens]"
        
        layer_predictions = []
        
        # Process each layer's residual stream
        for layer_idx in sorted(activations.residual.keys()):
            hidden = activations.residual[layer_idx]
            pred = self.get_layer_prediction(hidden, layer_idx, position, top_k)
            layer_predictions.append(pred)
        
        # Final prediction (from last layer)
        final_pred = layer_predictions[-1] if layer_predictions else None
        
        return PredictionTrajectory(
            input_text=input_text,
            layer_predictions=layer_predictions,
            final_prediction=final_pred,
        )
    
    def compare_trajectories(
        self,
        clean_trajectory: PredictionTrajectory,
        attack_trajectory: PredictionTrajectory
    ) -> Dict[str, Any]:
        """
        Compare prediction trajectories between clean and attack inputs.
        
        Returns dict with:
        - divergence_layer: First layer where predictions differ significantly
        - entropy_changes: Per-layer entropy differences
        - prediction_changes: Where top prediction changes
        """
        divergence_layer = None
        entropy_changes = []
        prediction_changes = []
        
        n_layers = min(
            len(clean_trajectory.layer_predictions),
            len(attack_trajectory.layer_predictions)
        )
        
        for i in range(n_layers):
            clean_pred = clean_trajectory.layer_predictions[i]
            attack_pred = attack_trajectory.layer_predictions[i]
            
            # Track entropy change
            entropy_diff = attack_pred.entropy - clean_pred.entropy
            entropy_changes.append(entropy_diff)
            
            # Check if top prediction changed
            clean_top = clean_pred.top_tokens[0] if clean_pred.top_tokens else -1
            attack_top = attack_pred.top_tokens[0] if attack_pred.top_tokens else -1
            
            if clean_top != attack_top:
                prediction_changes.append(i)
                if divergence_layer is None:
                    divergence_layer = i
        
        return {
            "divergence_layer": divergence_layer,
            "entropy_changes": entropy_changes,
            "prediction_changes": prediction_changes,
            "clean_final_entropy": clean_trajectory.final_prediction.entropy if clean_trajectory.final_prediction else 0,
            "attack_final_entropy": attack_trajectory.final_prediction.entropy if attack_trajectory.final_prediction else 0,
        }


class LogitLensVisualizer:
    """Visualization utilities for logit projections."""
    
    def __init__(self, projector: LogitProjector):
        self.projector = projector
    
    def format_trajectory_table(
        self,
        trajectory: PredictionTrajectory,
        show_n: int = 5
    ) -> str:
        """Format trajectory as text table."""
        lines = [
            f"Input: {trajectory.input_text[:50]}...",
            "-" * 60,
            f"{'Layer':<8} {'Top Token':<20} {'Prob':>8} {'Entropy':>10}",
            "-" * 60,
        ]
        
        for lp in trajectory.layer_predictions:
            if self.projector.tokenizer and lp.top_tokens:
                top_token = self.projector.tokenizer.decode([lp.top_tokens[0]])
            else:
                top_token = str(lp.top_tokens[0]) if lp.top_tokens else "N/A"
            
            lines.append(
                f"{lp.layer_idx:<8} {top_token:<20} {lp.max_prob:>8.3f} {lp.entropy:>10.3f}"
            )
        
        return "\n".join(lines)
    
    def get_trajectory_data(
        self,
        trajectory: PredictionTrajectory
    ) -> Dict[str, List]:
        """Extract trajectory data for plotting."""
        layers = []
        entropies = []
        max_probs = []
        top_tokens = []
        
        for lp in trajectory.layer_predictions:
            layers.append(lp.layer_idx)
            entropies.append(lp.entropy)
            max_probs.append(lp.max_prob)
            if self.projector.tokenizer and lp.top_tokens:
                top_tokens.append(self.projector.tokenizer.decode([lp.top_tokens[0]]))
            else:
                top_tokens.append(str(lp.top_tokens[0]) if lp.top_tokens else "")
        
        return {
            "layers": layers,
            "entropies": entropies,
            "max_probs": max_probs,
            "top_tokens": top_tokens,
        }


def run_logit_lens_analysis(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    hook_manager: Optional[ActivationHookManager] = None
) -> PredictionTrajectory:
    """
    Run complete logit lens analysis on input text.
    
    Args:
        model: The transformer model
        tokenizer: Model tokenizer
        text: Input text to analyze
        hook_manager: Optional pre-configured hook manager
        
    Returns:
        PredictionTrajectory for the input
    """
    # Create hook manager if not provided
    if hook_manager is None:
        hook_manager = ActivationHookManager(model)
        hook_manager.register_all_layers(["residual"])
    
    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(next(model.parameters()).device)
    
    # Run with cache
    logits, cache = hook_manager.run_with_cache(input_ids)
    
    # Create projector and analyze
    projector = LogitProjector(model, tokenizer)
    trajectory = projector.analyze_trajectory(input_ids, cache)
    
    return trajectory
