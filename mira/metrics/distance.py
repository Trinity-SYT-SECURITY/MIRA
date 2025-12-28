"""
Subspace distance metrics.

Provides metrics for measuring distances between activations and subspaces,
useful for quantifying attack effectiveness in representation space.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class DistanceResult:
    """Container for distance computation results."""
    refusal_distance: float
    acceptance_distance: float
    boundary_distance: float  # Distance from decision boundary
    direction_projection: float
    normalized_shift: float


class SubspaceDistanceMetrics:
    """
    Metrics for measuring subspace-related distances.
    
    Provides methods for quantifying:
    - Distance from refusal/acceptance subspaces
    - Projection onto direction vectors
    - Boundary crossing magnitude
    """
    
    def __init__(
        self,
        refusal_direction: torch.Tensor,
        acceptance_direction: torch.Tensor,
        refusal_center: Optional[torch.Tensor] = None,
        acceptance_center: Optional[torch.Tensor] = None,
    ):
        """
        Initialize distance metrics.
        
        Args:
            refusal_direction: Normalized refusal direction
            acceptance_direction: Normalized acceptance direction
            refusal_center: Mean of refusal activations (optional)
            acceptance_center: Mean of acceptance activations (optional)
        """
        self.refusal_direction = refusal_direction / refusal_direction.norm()
        self.acceptance_direction = acceptance_direction / acceptance_direction.norm()
        self.refusal_center = refusal_center
        self.acceptance_center = acceptance_center
    
    def direction_projection(
        self,
        activation: torch.Tensor,
        direction: str = "refusal",
    ) -> float:
        """
        Compute projection of activation onto a direction.
        
        Args:
            activation: Activation vector
            direction: "refusal" or "acceptance"
            
        Returns:
            Scalar projection value
        """
        if direction == "refusal":
            d = self.refusal_direction
        elif direction == "acceptance":
            d = self.acceptance_direction
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        d = d.to(activation.device)
        return float(torch.dot(activation.flatten(), d.flatten()))
    
    def cosine_similarity(
        self,
        activation: torch.Tensor,
        direction: str = "refusal",
    ) -> float:
        """
        Compute cosine similarity with a direction.
        
        Args:
            activation: Activation vector
            direction: "refusal" or "acceptance"
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        if direction == "refusal":
            d = self.refusal_direction
        elif direction == "acceptance":
            d = self.acceptance_direction
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        d = d.to(activation.device)
        a_norm = activation.flatten() / (activation.flatten().norm() + 1e-10)
        d_norm = d / (d.norm() + 1e-10)
        
        return float(torch.dot(a_norm, d_norm))
    
    def boundary_distance(
        self,
        activation: torch.Tensor,
    ) -> float:
        """
        Compute signed distance from decision boundary.
        
        Positive = closer to acceptance
        Negative = closer to refusal
        
        Args:
            activation: Activation vector
            
        Returns:
            Signed distance from boundary
        """
        refusal_proj = self.direction_projection(activation, "refusal")
        acceptance_proj = self.direction_projection(activation, "acceptance")
        
        return acceptance_proj - refusal_proj
    
    def euclidean_to_center(
        self,
        activation: torch.Tensor,
        target: str = "refusal",
    ) -> float:
        """
        Compute Euclidean distance to subspace center.
        
        Requires center tensors to be set.
        
        Args:
            activation: Activation vector
            target: "refusal" or "acceptance"
            
        Returns:
            Euclidean distance
        """
        if target == "refusal":
            if self.refusal_center is None:
                raise ValueError("Refusal center not set")
            center = self.refusal_center
        elif target == "acceptance":
            if self.acceptance_center is None:
                raise ValueError("Acceptance center not set")
            center = self.acceptance_center
        else:
            raise ValueError(f"Unknown target: {target}")
        
        center = center.to(activation.device)
        return float((activation.flatten() - center.flatten()).norm())
    
    def compute_all(
        self,
        activation: torch.Tensor,
    ) -> DistanceResult:
        """
        Compute all distance metrics.
        
        Args:
            activation: Activation vector
            
        Returns:
            DistanceResult with all metrics
        """
        refusal_proj = self.direction_projection(activation, "refusal")
        acceptance_proj = self.direction_projection(activation, "acceptance")
        boundary_dist = acceptance_proj - refusal_proj
        
        # Normalized shift: boundary distance normalized by direction magnitude
        direction_norm = self.refusal_direction.norm()
        normalized_shift = boundary_dist / (float(direction_norm) + 1e-10)
        
        return DistanceResult(
            refusal_distance=refusal_proj,
            acceptance_distance=acceptance_proj,
            boundary_distance=boundary_dist,
            direction_projection=acceptance_proj,
            normalized_shift=normalized_shift,
        )
    
    def measure_shift(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Measure the shift in distances before and after an attack.
        
        Args:
            before: Activation before attack
            after: Activation after attack
            
        Returns:
            Dictionary with shift measurements
        """
        before_metrics = self.compute_all(before)
        after_metrics = self.compute_all(after)
        
        return {
            "refusal_shift": after_metrics.refusal_distance - before_metrics.refusal_distance,
            "acceptance_shift": after_metrics.acceptance_distance - before_metrics.acceptance_distance,
            "boundary_shift": after_metrics.boundary_distance - before_metrics.boundary_distance,
            "normalized_shift": after_metrics.normalized_shift - before_metrics.normalized_shift,
            "crossed_boundary": before_metrics.boundary_distance < 0 and after_metrics.boundary_distance > 0,
        }


def compute_subspace_shift(
    model_wrapper,
    subspace_result,
    prompt: str,
    suffix: str,
    layer_idx: Optional[int] = None,
) -> Dict[str, float]:
    """
    Convenience function to compute subspace shift for an attack.
    
    Args:
        model_wrapper: ModelWrapper instance
        subspace_result: SubspaceResult with directions
        prompt: Original prompt
        suffix: Adversarial suffix
        layer_idx: Layer to analyze
        
    Returns:
        Dictionary with shift metrics
    """
    layer = layer_idx or (model_wrapper.n_layers // 2)
    
    # Get activations before and after
    _, cache_before = model_wrapper.run_with_cache(prompt)
    _, cache_after = model_wrapper.run_with_cache(prompt + " " + suffix)
    
    act_before = cache_before.hidden_states.get(layer)[0, -1, :]
    act_after = cache_after.hidden_states.get(layer)[0, -1, :]
    
    metrics = SubspaceDistanceMetrics(
        subspace_result.refusal_direction,
        subspace_result.acceptance_direction,
    )
    
    return metrics.measure_shift(act_before, act_after)
