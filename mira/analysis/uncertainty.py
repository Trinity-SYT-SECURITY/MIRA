"""
Uncertainty and entropy analysis for generation process.

Provides tools to analyze model confidence, entropy, and uncertainty
during text generation to detect anomalous or attack-prone behavior.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TokenMetrics:
    """Metrics for a single generated token."""
    token_id: int
    token_str: str
    probability: float
    entropy: float
    perplexity: float
    top_k_probs: List[float]
    rank: int  # Rank of chosen token in probability distribution


@dataclass  
class GenerationMetrics:
    """Aggregated metrics for entire generation."""
    tokens: List[TokenMetrics]
    mean_entropy: float
    max_entropy: float
    min_confidence: float
    mean_confidence: float
    entropy_variance: float
    risk_score: float
    risk_regions: List[Tuple[int, int]]  # (start, end) of high-risk spans
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "mean_entropy": self.mean_entropy,
            "max_entropy": self.max_entropy,
            "min_confidence": self.min_confidence,
            "mean_confidence": self.mean_confidence,
            "entropy_variance": self.entropy_variance,
            "risk_score": self.risk_score,
            "risk_regions": self.risk_regions,
            "num_tokens": len(self.tokens),
        }


class UncertaintyAnalyzer:
    """
    Analyzes model uncertainty during generation.
    
    Provides entropy, confidence, and perplexity metrics
    to understand model behavior and detect anomalies.
    """
    
    def __init__(self, vocab_size: Optional[int] = None):
        self.vocab_size = vocab_size
    
    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute Shannon entropy of probability distribution.
        
        Args:
            logits: Logits tensor [vocab_size]
            
        Returns:
            Entropy value (higher = more uncertain)
        """
        probs = F.softmax(logits, dim=-1)
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        return entropy
    
    def compute_confidence(self, logits: torch.Tensor) -> float:
        """
        Compute confidence as probability of top choice.
        
        Args:
            logits: Logits tensor [vocab_size]
            
        Returns:
            Confidence value (higher = more confident)
        """
        probs = F.softmax(logits, dim=-1)
        return probs.max().item()
    
    def compute_perplexity(self, logits: torch.Tensor) -> float:
        """
        Compute perplexity (exp of entropy).
        
        Args:
            logits: Logits tensor
            
        Returns:
            Perplexity value
        """
        entropy = self.compute_entropy(logits)
        return np.exp(entropy)
    
    def compute_top_k_entropy(self, logits: torch.Tensor, k: int = 10) -> float:
        """
        Compute entropy over top-k tokens only.
        
        Useful for focusing on the "decision space" rather than full distribution.
        """
        probs = F.softmax(logits, dim=-1)
        top_probs, _ = torch.topk(probs, k=min(k, len(probs)))
        top_probs = top_probs / top_probs.sum()  # Renormalize
        log_probs = torch.log(top_probs + 1e-10)
        entropy = -torch.sum(top_probs * log_probs).item()
        return entropy
    
    def get_token_rank(self, logits: torch.Tensor, token_id: int) -> int:
        """Get rank of specific token in probability distribution."""
        probs = F.softmax(logits, dim=-1)
        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0]
        return rank[0].item() + 1 if len(rank) > 0 else -1
    
    def analyze_logits(
        self, 
        logits: torch.Tensor, 
        chosen_token_id: int,
        tokenizer: Any = None
    ) -> TokenMetrics:
        """
        Analyze logits for a single generation step.
        
        Args:
            logits: Logits tensor [vocab_size]
            chosen_token_id: ID of token that was generated
            tokenizer: Optional tokenizer for decoding
            
        Returns:
            TokenMetrics for this step
        """
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=10)
        
        token_str = ""
        if tokenizer:
            token_str = tokenizer.decode([chosen_token_id])
        
        chosen_prob = probs[chosen_token_id].item()
        
        return TokenMetrics(
            token_id=chosen_token_id,
            token_str=token_str,
            probability=chosen_prob,
            entropy=self.compute_entropy(logits),
            perplexity=self.compute_perplexity(logits),
            top_k_probs=top_probs.tolist(),
            rank=self.get_token_rank(logits, chosen_token_id),
        )


class GenerationTracker:
    """
    Tracks uncertainty metrics across entire generation process.
    
    Monitors for anomalies, risk regions, and unusual patterns
    that may indicate successful attacks or model confusion.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        entropy_threshold: float = 5.0,
        confidence_threshold: float = 0.3
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.analyzer = UncertaintyAnalyzer()
        self.entropy_threshold = entropy_threshold
        self.confidence_threshold = confidence_threshold
    
    def track_generation(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0
    ) -> GenerationMetrics:
        """
        Track uncertainty metrics throughout generation.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            GenerationMetrics with full analysis
        """
        device = next(self.model.parameters()).device
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        token_metrics = []
        
        # Generate token by token
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Analyze this step
                metrics = self.analyzer.analyze_logits(
                    logits, 
                    next_token.item(),
                    self.tokenizer
                )
                token_metrics.append(metrics)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # Compute aggregate metrics
        return self._compute_generation_metrics(token_metrics)
    
    def _compute_generation_metrics(
        self, 
        token_metrics: List[TokenMetrics]
    ) -> GenerationMetrics:
        """Compute aggregate metrics from per-token metrics."""
        if not token_metrics:
            return GenerationMetrics(
                tokens=[],
                mean_entropy=0.0,
                max_entropy=0.0,
                min_confidence=0.0,
                mean_confidence=0.0,
                entropy_variance=0.0,
                risk_score=0.0,
                risk_regions=[],
            )
        
        entropies = [m.entropy for m in token_metrics]
        confidences = [m.probability for m in token_metrics]
        
        # Find risk regions (high entropy or low confidence)
        risk_regions = self._find_risk_regions(token_metrics)
        
        # Compute risk score
        risk_score = self._compute_risk_score(token_metrics, risk_regions)
        
        return GenerationMetrics(
            tokens=token_metrics,
            mean_entropy=np.mean(entropies),
            max_entropy=np.max(entropies),
            min_confidence=np.min(confidences),
            mean_confidence=np.mean(confidences),
            entropy_variance=np.var(entropies),
            risk_score=risk_score,
            risk_regions=risk_regions,
        )
    
    def _find_risk_regions(
        self, 
        token_metrics: List[TokenMetrics]
    ) -> List[Tuple[int, int]]:
        """Find regions with elevated risk indicators."""
        regions = []
        in_region = False
        region_start = 0
        
        for i, metrics in enumerate(token_metrics):
            is_risky = (
                metrics.entropy > self.entropy_threshold or
                metrics.probability < self.confidence_threshold
            )
            
            if is_risky and not in_region:
                region_start = i
                in_region = True
            elif not is_risky and in_region:
                regions.append((region_start, i))
                in_region = False
        
        # Close any open region
        if in_region:
            regions.append((region_start, len(token_metrics)))
        
        return regions
    
    def _compute_risk_score(
        self,
        token_metrics: List[TokenMetrics],
        risk_regions: List[Tuple[int, int]]
    ) -> float:
        """
        Compute overall risk score for generation.
        
        Factors:
        - Proportion of tokens in risk regions
        - Average entropy in risk regions
        - Entropy spikes
        """
        if not token_metrics:
            return 0.0
        
        # Proportion in risk regions
        total_risk_tokens = sum(end - start for start, end in risk_regions)
        risk_proportion = total_risk_tokens / len(token_metrics)
        
        # Entropy spikes
        entropies = [m.entropy for m in token_metrics]
        mean_entropy = np.mean(entropies)
        spikes = sum(1 for e in entropies if e > mean_entropy * 1.5)
        spike_proportion = spikes / len(token_metrics)
        
        # Low confidence tokens
        low_conf = sum(1 for m in token_metrics if m.probability < self.confidence_threshold)
        low_conf_proportion = low_conf / len(token_metrics)
        
        # Combined score
        risk_score = (
            0.4 * risk_proportion +
            0.3 * spike_proportion +
            0.3 * low_conf_proportion
        )
        
        return min(1.0, risk_score)


class RiskDetector:
    """
    Detects high-risk patterns in model generation.
    
    Identifies potential attack success indicators,
    anomalous behavior, and safety-relevant patterns.
    """
    
    def __init__(
        self,
        entropy_spike_threshold: float = 2.0,
        confidence_drop_threshold: float = 0.5,
        window_size: int = 5
    ):
        self.entropy_spike_threshold = entropy_spike_threshold
        self.confidence_drop_threshold = confidence_drop_threshold
        self.window_size = window_size
    
    def detect_entropy_spike(
        self, 
        metrics: GenerationMetrics
    ) -> List[int]:
        """Find positions with sudden entropy increases."""
        spikes = []
        entropies = [m.entropy for m in metrics.tokens]
        
        for i in range(1, len(entropies)):
            if entropies[i] > entropies[i-1] * self.entropy_spike_threshold:
                spikes.append(i)
        
        return spikes
    
    def detect_confidence_drop(
        self,
        metrics: GenerationMetrics
    ) -> List[int]:
        """Find positions where confidence suddenly drops."""
        drops = []
        
        for i in range(1, len(metrics.tokens)):
            prev_conf = metrics.tokens[i-1].probability
            curr_conf = metrics.tokens[i].probability
            
            if prev_conf > 0 and curr_conf / prev_conf < self.confidence_drop_threshold:
                drops.append(i)
        
        return drops
    
    def detect_sustained_uncertainty(
        self,
        metrics: GenerationMetrics,
        threshold: float = 5.0
    ) -> List[Tuple[int, int]]:
        """Find regions of sustained high uncertainty."""
        regions = []
        consecutive = 0
        start = -1
        
        for i, token_metrics in enumerate(metrics.tokens):
            if token_metrics.entropy > threshold:
                if start == -1:
                    start = i
                consecutive += 1
            else:
                if consecutive >= self.window_size:
                    regions.append((start, i))
                start = -1
                consecutive = 0
        
        # Check end of sequence
        if consecutive >= self.window_size:
            regions.append((start, len(metrics.tokens)))
        
        return regions
    
    def get_risk_summary(
        self,
        metrics: GenerationMetrics
    ) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        entropy_spikes = self.detect_entropy_spike(metrics)
        confidence_drops = self.detect_confidence_drop(metrics)
        sustained_uncertainty = self.detect_sustained_uncertainty(metrics)
        
        # Risk indicators
        has_entropy_spikes = len(entropy_spikes) > 0
        has_confidence_drops = len(confidence_drops) > 0
        has_sustained_uncertainty = len(sustained_uncertainty) > 0
        
        # Overall risk level
        risk_indicators = sum([
            has_entropy_spikes,
            has_confidence_drops,
            has_sustained_uncertainty,
            metrics.risk_score > 0.5
        ])
        
        if risk_indicators >= 3:
            risk_level = "HIGH"
        elif risk_indicators >= 2:
            risk_level = "MEDIUM"
        elif risk_indicators >= 1:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            "risk_level": risk_level,
            "risk_score": metrics.risk_score,
            "entropy_spikes": entropy_spikes,
            "confidence_drops": confidence_drops,
            "sustained_uncertainty": sustained_uncertainty,
            "mean_entropy": metrics.mean_entropy,
            "min_confidence": metrics.min_confidence,
        }


def analyze_generation_uncertainty(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Convenience function for full uncertainty analysis.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with metrics and risk analysis
    """
    tracker = GenerationTracker(model, tokenizer)
    metrics = tracker.track_generation(prompt, max_tokens)
    
    detector = RiskDetector()
    risk_summary = detector.get_risk_summary(metrics)
    
    return {
        "metrics": metrics.to_dict(),
        "risk": risk_summary,
        "tokens": [
            {
                "token": m.token_str,
                "probability": m.probability,
                "entropy": m.entropy,
                "rank": m.rank,
            }
            for m in metrics.tokens[:50]  # Limit for readability
        ]
    }
