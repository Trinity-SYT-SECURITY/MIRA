"""
Enhanced Subspace Quantification Analysis.

Quantifies subspace differences between successful and failed attacks:
- KL divergence between success/failure distributions
- Cosine similarity with refusal/acceptance directions
- Subspace overlap analysis
- Probe response strength by attack type
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy


@dataclass
class SubspaceQuantification:
    """Quantitative subspace analysis results."""
    success_kl_div: float = 0.0
    failure_kl_div: float = 0.0
    success_cosine_sim: float = 0.0
    failure_cosine_sim: float = 0.0
    subspace_overlap: float = 0.0
    probe_response_strength: Dict[str, float] = None
    
    def __post_init__(self):
        if self.probe_response_strength is None:
            self.probe_response_strength = {}


class SubspaceQuantifier:
    """Quantify subspace differences for attack analysis."""
    
    def __init__(self, refusal_direction: Optional[torch.Tensor] = None, acceptance_direction: Optional[torch.Tensor] = None):
        """Initialize quantifier with subspace directions."""
        self.refusal_direction = refusal_direction
        self.acceptance_direction = acceptance_direction
    
    def compute_kl_divergence(
        self,
        activations1: torch.Tensor,
        activations2: torch.Tensor,
    ) -> float:
        """
        Compute KL divergence between two activation distributions.
        
        Args:
            activations1: First set of activations [N, hidden_dim]
            activations2: Second set of activations [M, hidden_dim]
            
        Returns:
            KL divergence value
        """
        if activations1.numel() == 0 or activations2.numel() == 0:
            return 0.0
        
        # Project to lower dimension using PCA or mean pooling
        if activations1.dim() > 2:
            activations1 = activations1.mean(dim=1)  # Average over sequence
        if activations2.dim() > 2:
            activations2 = activations2.mean(dim=1)
        
        # Flatten if needed
        if activations1.dim() > 2:
            activations1 = activations1.flatten(start_dim=1)
        if activations2.dim() > 2:
            activations2 = activations2.flatten(start_dim=1)
        
        # Normalize to probability distributions
        act1_norm = torch.softmax(activations1.mean(dim=0), dim=-1)
        act2_norm = torch.softmax(activations2.mean(dim=0), dim=-1)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        act1_norm = act1_norm + eps
        act2_norm = act2_norm + eps
        act1_norm = act1_norm / act1_norm.sum()
        act2_norm = act2_norm / act2_norm.sum()
        
        # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = float((act1_norm * torch.log(act1_norm / act2_norm)).sum())
        
        return kl_div
    
    def compute_cosine_similarity(
        self,
        activations: torch.Tensor,
        direction: torch.Tensor,
    ) -> float:
        """
        Compute cosine similarity between activations and a direction vector.
        
        Args:
            activations: Activation tensor [N, hidden_dim] or [N, seq_len, hidden_dim]
            direction: Direction vector [hidden_dim]
            
        Returns:
            Average cosine similarity
        """
        if activations.numel() == 0:
            return 0.0
        
        # Handle different activation shapes
        if activations.dim() == 3:
            activations = activations[:, -1, :]  # Last token
        elif activations.dim() > 2:
            activations = activations.flatten(start_dim=1)
        
        # Normalize
        direction = direction.to(activations.device)
        direction_norm = direction / (direction.norm() + 1e-10)
        
        # Compute cosine similarity for each activation
        activations_norm = activations / (activations.norm(dim=-1, keepdim=True) + 1e-10)
        similarities = torch.mm(activations_norm, direction_norm.unsqueeze(1)).squeeze(1)
        
        return float(similarities.mean().item())
    
    def compute_subspace_overlap(
        self,
        refusal_activations: torch.Tensor,
        acceptance_activations: torch.Tensor,
    ) -> float:
        """
        Compute overlap between refusal and acceptance subspaces.
        
        Args:
            refusal_activations: Activations in refusal subspace [N, hidden_dim]
            acceptance_activations: Activations in acceptance subspace [M, hidden_dim]
            
        Returns:
            Overlap score (0-1, higher = more overlap)
        """
        if refusal_activations.numel() == 0 or acceptance_activations.numel() == 0:
            return 0.0
        
        # Compute mean vectors
        refusal_mean = refusal_activations.mean(dim=0)
        acceptance_mean = acceptance_activations.mean(dim=0)
        
        # Compute cosine similarity between means
        refusal_norm = refusal_mean / (refusal_mean.norm() + 1e-10)
        acceptance_norm = acceptance_mean / (acceptance_mean.norm() + 1e-10)
        
        overlap = float(torch.dot(refusal_norm, acceptance_norm))
        
        # Convert to 0-1 range (cosine similarity is -1 to 1)
        overlap_normalized = (overlap + 1.0) / 2.0
        
        return overlap_normalized
    
    def quantify_attack_differences(
        self,
        success_activations: List[torch.Tensor],
        failure_activations: List[torch.Tensor],
        baseline_activations: Optional[List[torch.Tensor]] = None,
    ) -> SubspaceQuantification:
        """
        Quantify differences between successful and failed attacks.
        
        Args:
            success_activations: List of activation tensors from successful attacks
            failure_activations: List of activation tensors from failed attacks
            baseline_activations: Optional baseline activations for comparison
            
        Returns:
            SubspaceQuantification with all metrics
        """
        if not success_activations and not failure_activations:
            return SubspaceQuantification()
        
        # Concatenate activations
        success_cat = torch.cat(success_activations, dim=0) if success_activations else torch.empty(0)
        failure_cat = torch.cat(failure_activations, dim=0) if failure_activations else torch.empty(0)
        
        # Compute KL divergence
        success_kl_div = 0.0
        failure_kl_div = 0.0
        
        if baseline_activations and len(baseline_activations) > 0:
            baseline_cat = torch.cat(baseline_activations, dim=0)
            
            if success_cat.numel() > 0:
                success_kl_div = self.compute_kl_divergence(success_cat, baseline_cat)
            
            if failure_cat.numel() > 0:
                failure_kl_div = self.compute_kl_divergence(failure_cat, baseline_cat)
        elif success_cat.numel() > 0 and failure_cat.numel() > 0:
            # Compare success vs failure
            success_kl_div = self.compute_kl_divergence(success_cat, failure_cat)
            failure_kl_div = self.compute_kl_divergence(failure_cat, success_cat)
        
        # Compute cosine similarity with directions
        success_cosine_sim = 0.0
        failure_cosine_sim = 0.0
        
        if self.refusal_direction is not None:
            if success_cat.numel() > 0:
                success_cosine_sim = self.compute_cosine_similarity(success_cat, self.refusal_direction)
            if failure_cat.numel() > 0:
                failure_cosine_sim = self.compute_cosine_similarity(failure_cat, self.refusal_direction)
        
        # Compute subspace overlap
        subspace_overlap = 0.0
        if success_cat.numel() > 0 and failure_cat.numel() > 0:
            subspace_overlap = self.compute_subspace_overlap(success_cat, failure_cat)
        
        return SubspaceQuantification(
            success_kl_div=success_kl_div,
            failure_kl_div=failure_kl_div,
            success_cosine_sim=success_cosine_sim,
            failure_cosine_sim=failure_cosine_sim,
            subspace_overlap=subspace_overlap,
            probe_response_strength={},
        )
    
    def compute_probe_response_by_type(
        self,
        activations_by_type: Dict[str, List[torch.Tensor]],
        probe: Optional[torch.nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Compute probe response strength for different attack types.
        
        Args:
            activations_by_type: Dict mapping attack type to list of activations
            probe: Optional probe model for prediction
            
        Returns:
            Dict mapping attack type to average probe response strength
        """
        response_strength = {}
        
        for attack_type, activations in activations_by_type.items():
            if not activations:
                response_strength[attack_type] = 0.0
                continue
            
            # Concatenate activations
            act_cat = torch.cat(activations, dim=0)
            
            if probe is not None:
                # Use probe predictions
                with torch.no_grad():
                    if act_cat.dim() == 3:
                        act_cat = act_cat[:, -1, :]  # Last token
                    
                    probe_pred = torch.sigmoid(probe(act_cat))
                    response_strength[attack_type] = float(probe_pred.mean().item())
            else:
                # Use cosine similarity with refusal direction as proxy
                if self.refusal_direction is not None:
                    response_strength[attack_type] = self.compute_cosine_similarity(act_cat, self.refusal_direction)
                else:
                    response_strength[attack_type] = 0.0
        
        return response_strength

