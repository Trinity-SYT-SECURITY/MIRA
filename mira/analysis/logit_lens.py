"""
Logit lens analysis for layer-wise prediction inspection.

This module implements the "logit lens" technique for:
- Projecting intermediate layer states to vocabulary space
- Tracking how predictions evolve across layers
- Identifying where safety decisions emerge
- Analyzing prediction uncertainty at each layer
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class LayerPrediction:
    """Prediction information at a specific layer."""
    layer_idx: int
    top_tokens: List[str]
    top_probs: List[float]
    target_prob: Optional[float]
    entropy: float
    confidence: float


@dataclass
class PredictionTrajectory:
    """How prediction evolves across layers."""
    token_position: int
    layer_predictions: List[LayerPrediction]
    emergence_layer: Optional[int]  # Where final prediction first appears in top-k
    stability_score: float  # How stable is the prediction across layers


class LogitLens:
    """
    Logit lens for analyzing layer-by-layer predictions.
    
    Enables inspection of:
    - What each layer "thinks" the next token should be
    - Where refusal/acceptance decisions emerge
    - How attacks change the prediction trajectory
    """
    
    def __init__(self, model_wrapper):
        """
        Initialize logit lens.
        
        Args:
            model_wrapper: ModelWrapper instance
        """
        self.model = model_wrapper
    
    def project_layer_to_vocab(
        self,
        hidden_state: torch.Tensor,
        apply_norm: bool = True,
    ) -> torch.Tensor:
        """
        Project hidden state to vocabulary logits.
        
        Args:
            hidden_state: Hidden state tensor
            apply_norm: Whether to apply layer normalization first
            
        Returns:
            Logits over vocabulary
        """
        return self.model.unembed(hidden_state)
    
    def get_layer_predictions(
        self,
        text: str,
        token_position: int = -1,
        top_k: int = 10,
        target_tokens: Optional[List[str]] = None,
    ) -> List[LayerPrediction]:
        """
        Get predictions at each layer for a specific token position.
        
        Args:
            text: Input text
            token_position: Which token position to analyze (-1 for last)
            top_k: Number of top predictions to return
            target_tokens: Specific tokens to track probability of
            
        Returns:
            List of LayerPrediction for each layer
        """
        _, cache = self.model.run_with_cache(text)
        
        predictions = []
        target_ids = None
        
        if target_tokens:
            target_ids = [
                self.model.tokenizer.encode(t, add_special_tokens=False)[0]
                for t in target_tokens
            ]
        
        for layer_idx in range(self.model.n_layers):
            hidden = cache.hidden_states.get(layer_idx)
            if hidden is None:
                continue
            
            # Get hidden state at specified position
            pos_hidden = hidden[0, token_position, :].unsqueeze(0)
            
            # Project to vocabulary
            logits = self.project_layer_to_vocab(pos_hidden)
            probs = F.softmax(logits, dim=-1)[0]
            
            # Get top-k predictions
            top_probs, top_indices = probs.topk(top_k)
            top_tokens = [
                self.model.tokenizer.decode([idx])
                for idx in top_indices.tolist()
            ]
            
            # Compute entropy
            entropy = float(-(probs * (probs + 1e-10).log()).sum())
            
            # Confidence: probability of top token
            confidence = float(top_probs[0])
            
            # Target token probability
            target_prob = None
            if target_ids:
                target_probs = [float(probs[tid]) for tid in target_ids]
                target_prob = max(target_probs)
            
            predictions.append(LayerPrediction(
                layer_idx=layer_idx,
                top_tokens=top_tokens,
                top_probs=[float(p) for p in top_probs.tolist()],
                target_prob=target_prob,
                entropy=entropy,
                confidence=confidence,
            ))
        
        return predictions
    
    def track_prediction_trajectory(
        self,
        text: str,
        token_position: int = -1,
        top_k: int = 5,
    ) -> PredictionTrajectory:
        """
        Track how prediction evolves across layers.
        
        Useful for identifying where safety decisions emerge.
        
        Args:
            text: Input text
            token_position: Token position to track
            top_k: Consider top-k predictions for emergence detection
            
        Returns:
            PredictionTrajectory with layer-by-layer analysis
        """
        predictions = self.get_layer_predictions(text, token_position, top_k)
        
        if not predictions:
            raise ValueError("No predictions collected")
        
        # Get final prediction
        final_pred = predictions[-1].top_tokens[0] if predictions else None
        
        # Find emergence layer (where final prediction first appears in top-k)
        emergence_layer = None
        for pred in predictions:
            if final_pred in pred.top_tokens:
                emergence_layer = pred.layer_idx
                break
        
        # Compute stability score (how consistent is top prediction across layers)
        if len(predictions) > 1:
            top_tokens = [p.top_tokens[0] for p in predictions]
            unique_tops = len(set(top_tokens))
            stability_score = 1.0 - (unique_tops - 1) / len(predictions)
        else:
            stability_score = 1.0
        
        return PredictionTrajectory(
            token_position=token_position,
            layer_predictions=predictions,
            emergence_layer=emergence_layer,
            stability_score=stability_score,
        )
    
    def compare_trajectories(
        self,
        text1: str,
        text2: str,
        token_position: int = -1,
    ) -> Dict[str, any]:
        """
        Compare prediction trajectories between two inputs.
        
        Useful for understanding how attacks change internal predictions.
        
        Args:
            text1: First input
            text2: Second input
            token_position: Token position to compare
            
        Returns:
            Dictionary with comparison metrics
        """
        traj1 = self.track_prediction_trajectory(text1, token_position)
        traj2 = self.track_prediction_trajectory(text2, token_position)
        
        # Compare emergence layers
        emergence_diff = None
        if traj1.emergence_layer is not None and traj2.emergence_layer is not None:
            emergence_diff = traj2.emergence_layer - traj1.emergence_layer
        
        # Compare layer-by-layer predictions
        divergence_layers = []
        for pred1, pred2 in zip(traj1.layer_predictions, traj2.layer_predictions):
            if pred1.top_tokens[0] != pred2.top_tokens[0]:
                divergence_layers.append(pred1.layer_idx)
        
        # Find first divergence
        first_divergence = divergence_layers[0] if divergence_layers else None
        
        return {
            "trajectory1": traj1,
            "trajectory2": traj2,
            "emergence_diff": emergence_diff,
            "divergence_layers": divergence_layers,
            "first_divergence": first_divergence,
            "stability_diff": traj2.stability_score - traj1.stability_score,
        }
    
    def find_refusal_emergence(
        self,
        harmful_texts: List[str],
        refusal_tokens: List[str] = None,
    ) -> Dict[str, float]:
        """
        Find at which layer refusal behavior typically emerges.
        
        Args:
            harmful_texts: List of harmful prompts that trigger refusal
            refusal_tokens: Tokens that indicate refusal (default: common refusals)
            
        Returns:
            Dictionary with emergence statistics
        """
        if refusal_tokens is None:
            refusal_tokens = ["I", "Sorry", "cannot", "can't", "won't", "As"]
        
        emergence_layers = []
        
        for text in harmful_texts:
            predictions = self.get_layer_predictions(text, -1, top_k=10)
            
            for pred in predictions:
                # Check if any refusal token appears in top predictions
                for refusal_tok in refusal_tokens:
                    if any(refusal_tok.lower() in t.lower() for t in pred.top_tokens):
                        emergence_layers.append(pred.layer_idx)
                        break
                else:
                    continue
                break
        
        if not emergence_layers:
            return {"mean_layer": None, "std_layer": None, "samples": 0}
        
        return {
            "mean_layer": np.mean(emergence_layers),
            "std_layer": np.std(emergence_layers),
            "min_layer": min(emergence_layers),
            "max_layer": max(emergence_layers),
            "samples": len(emergence_layers),
        }
    
    def layer_contribution_to_prediction(
        self,
        text: str,
        token_position: int = -1,
    ) -> List[Dict[str, float]]:
        """
        Measure each layer's contribution to the final prediction.
        
        Uses difference in target token probability between consecutive layers.
        
        Args:
            text: Input text
            token_position: Token position to analyze
            
        Returns:
            List of contribution info for each layer
        """
        predictions = self.get_layer_predictions(text, token_position, top_k=50)
        
        if len(predictions) < 2:
            return []
        
        # Get final top token
        final_token = predictions[-1].top_tokens[0]
        final_token_id = self.model.tokenizer.encode(
            final_token, add_special_tokens=False
        )[0]
        
        # Recompute with target token tracking
        predictions_with_target = self.get_layer_predictions(
            text, token_position, top_k=50, target_tokens=[final_token]
        )
        
        contributions = []
        prev_prob = 0.0
        
        for pred in predictions_with_target:
            curr_prob = pred.target_prob or 0.0
            contribution = curr_prob - prev_prob
            
            contributions.append({
                "layer_idx": pred.layer_idx,
                "target_prob": curr_prob,
                "contribution": contribution,
                "entropy": pred.entropy,
            })
            
            prev_prob = curr_prob
        
        return contributions
