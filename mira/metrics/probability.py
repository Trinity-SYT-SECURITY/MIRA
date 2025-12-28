"""
Probability distribution metrics.

Provides metrics for analyzing probability distributions during
model inference, useful for understanding attack effects on output.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class EntropyMetrics:
    """Container for entropy-based metrics."""
    raw_entropy: float
    normalized_entropy: float
    perplexity: float
    top_k_mass: float
    uniformity: float


@dataclass
class DistributionShift:
    """Metrics for comparing two probability distributions."""
    kl_divergence: float
    js_divergence: float
    total_variation: float
    cosine_similarity: float
    top_token_change: bool


class ProbabilityMetrics:
    """
    Metrics for analyzing probability distributions.
    
    Useful for:
    - Measuring uncertainty in model predictions
    - Comparing distributions before/after attacks
    - Quantifying probability mass shifts
    """
    
    def __init__(self, vocab_size: int):
        """
        Initialize probability metrics.
        
        Args:
            vocab_size: Size of model vocabulary
        """
        self.vocab_size = vocab_size
    
    def compute_entropy(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> EntropyMetrics:
        """
        Compute entropy-based metrics from logits.
        
        Args:
            logits: Raw model logits
            temperature: Temperature for softmax
            
        Returns:
            EntropyMetrics with various measures
        """
        # Apply temperature and softmax
        probs = F.softmax(logits / temperature, dim=-1)
        probs = probs.flatten()
        
        # Raw entropy
        eps = 1e-10
        raw_entropy = -float((probs * (probs + eps).log()).sum())
        
        # Normalized entropy (0 to 1)
        max_entropy = np.log(len(probs))
        normalized_entropy = raw_entropy / max_entropy
        
        # Perplexity
        perplexity = np.exp(raw_entropy)
        
        # Top-k probability mass
        top_k = min(100, len(probs))
        top_probs, _ = probs.topk(top_k)
        top_k_mass = float(top_probs.sum())
        
        # Uniformity (inverse concentration)
        uniformity = 1.0 - float((probs ** 2).sum())
        
        return EntropyMetrics(
            raw_entropy=raw_entropy,
            normalized_entropy=normalized_entropy,
            perplexity=perplexity,
            top_k_mass=top_k_mass,
            uniformity=uniformity,
        )
    
    def compare_distributions(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
    ) -> DistributionShift:
        """
        Compare two probability distributions.
        
        Args:
            logits1: First distribution logits
            logits2: Second distribution logits
            
        Returns:
            DistributionShift with comparison metrics
        """
        probs1 = F.softmax(logits1, dim=-1).flatten()
        probs2 = F.softmax(logits2, dim=-1).flatten()
        
        eps = 1e-10
        
        # KL divergence
        kl_div = float((probs1 * ((probs1 + eps) / (probs2 + eps)).log()).sum())
        
        # JS divergence (symmetric)
        m = (probs1 + probs2) / 2
        js_div = 0.5 * float((probs1 * ((probs1 + eps) / (m + eps)).log()).sum())
        js_div += 0.5 * float((probs2 * ((probs2 + eps) / (m + eps)).log()).sum())
        
        # Total variation distance
        tv_dist = 0.5 * float((probs1 - probs2).abs().sum())
        
        # Cosine similarity
        cos_sim = float(F.cosine_similarity(probs1.unsqueeze(0), probs2.unsqueeze(0)))
        
        # Top token change
        top1_idx1 = probs1.argmax().item()
        top1_idx2 = probs2.argmax().item()
        top_change = top1_idx1 != top1_idx2
        
        return DistributionShift(
            kl_divergence=kl_div,
            js_divergence=js_div,
            total_variation=tv_dist,
            cosine_similarity=cos_sim,
            top_token_change=top_change,
        )
    
    def target_probability(
        self,
        logits: torch.Tensor,
        target_tokens: List[int],
    ) -> Dict[str, float]:
        """
        Compute probabilities for specific target tokens.
        
        Args:
            logits: Model logits
            target_tokens: Token IDs to measure
            
        Returns:
            Dictionary with token probabilities
        """
        probs = F.softmax(logits, dim=-1).flatten()
        
        results = {
            "individual": {},
            "combined": 0.0,
            "max": 0.0,
        }
        
        for token_id in target_tokens:
            if token_id < len(probs):
                prob = float(probs[token_id])
                results["individual"][token_id] = prob
                results["combined"] += prob
                results["max"] = max(results["max"], prob)
        
        return results
    
    def probability_shift_for_tokens(
        self,
        logits_before: torch.Tensor,
        logits_after: torch.Tensor,
        token_ids: List[int],
    ) -> Dict[str, float]:
        """
        Measure probability shifts for specific tokens.
        
        Args:
            logits_before: Logits before attack
            logits_after: Logits after attack
            token_ids: Tokens to track
            
        Returns:
            Dictionary with shift measurements
        """
        before_probs = self.target_probability(logits_before, token_ids)
        after_probs = self.target_probability(logits_after, token_ids)
        
        shifts = {}
        for token_id in token_ids:
            before = before_probs["individual"].get(token_id, 0.0)
            after = after_probs["individual"].get(token_id, 0.0)
            shifts[token_id] = after - before
        
        return {
            "individual_shifts": shifts,
            "combined_shift": after_probs["combined"] - before_probs["combined"],
            "max_shift": after_probs["max"] - before_probs["max"],
            "relative_increase": (
                (after_probs["combined"] / (before_probs["combined"] + 1e-10))
                if before_probs["combined"] > 0 else float("inf")
            ),
        }


def compute_probability_metrics(
    model_wrapper,
    prompt: str,
    suffix: str,
    target_tokens: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Convenience function to compute probability metrics.
    
    Args:
        model_wrapper: ModelWrapper instance
        prompt: Original prompt
        suffix: Adversarial suffix
        target_tokens: Tokens to track (default: affirmative tokens)
        
    Returns:
        Dictionary with all metrics
    """
    if target_tokens is None:
        target_tokens = ["Sure", "Here", "Certainly", "Of"]
    
    # Encode target tokens
    target_ids = []
    for token in target_tokens:
        ids = model_wrapper.tokenizer.encode(token, add_special_tokens=False)
        if ids:
            target_ids.append(ids[0])
    
    # Get logits
    logits_before = model_wrapper.compute_token_logits(prompt)
    logits_after = model_wrapper.compute_token_logits(prompt + " " + suffix)
    
    metrics = ProbabilityMetrics(model_wrapper.vocab_size)
    
    # Compute all metrics
    entropy_before = metrics.compute_entropy(logits_before[0, -1, :])
    entropy_after = metrics.compute_entropy(logits_after[0, -1, :])
    
    dist_shift = metrics.compare_distributions(
        logits_before[0, -1, :],
        logits_after[0, -1, :],
    )
    
    prob_shift = metrics.probability_shift_for_tokens(
        logits_before[0, -1, :],
        logits_after[0, -1, :],
        target_ids,
    )
    
    return {
        "entropy_before": entropy_before,
        "entropy_after": entropy_after,
        "distribution_shift": dist_shift,
        "target_probability_shift": prob_shift,
    }
