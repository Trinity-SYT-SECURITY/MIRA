"""
Attention analysis for studying attention patterns and heads.

This module provides tools for:
- Analyzing attention patterns across layers and heads
- Identifying safety-relevant attention heads
- Measuring attention shifts during attacks
- Computing attention-based metrics
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class AttentionHeadInfo:
    """Information about an attention head."""
    layer_idx: int
    head_idx: int
    importance_score: float
    pattern_type: str  # "local", "global", "previous_token", "induction", etc.
    avg_entropy: float


@dataclass
class AttentionShift:
    """Quantifies attention shift between two inputs."""
    layer_idx: int
    head_idx: int
    kl_divergence: float
    js_divergence: float
    max_shift: float
    source_focus: List[int]
    target_focus: List[int]


class AttentionAnalyzer:
    """
    Analyzer for studying attention mechanisms.
    
    Enables analysis of:
    - Attention pattern visualization and statistics
    - Identification of important attention heads
    - Attention hijacking detection
    - Safety head identification
    """
    
    def __init__(self, model_wrapper):
        """
        Initialize attention analyzer.
        
        Args:
            model_wrapper: ModelWrapper instance
        """
        self.model = model_wrapper
        self.n_layers = model_wrapper.n_layers
        self.n_heads = model_wrapper.n_heads
    
    def get_attention_patterns(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Extract attention patterns for specified layers and heads.
        
        Args:
            text: Input text
            layers: Layers to extract (default: all)
            heads: Heads to extract (default: all)
            
        Returns:
            Dictionary mapping (layer, head) to attention pattern tensor
        """
        _, cache = self.model.run_with_cache(text)
        
        if layers is None:
            layers = list(range(self.n_layers))
        if heads is None:
            heads = list(range(self.n_heads))
        
        patterns = {}
        for layer_idx in layers:
            attn = cache.attention_weights.get(layer_idx)
            if attn is not None:
                for head_idx in heads:
                    if head_idx < attn.shape[1]:
                        patterns[(layer_idx, head_idx)] = attn[0, head_idx, :, :]
        
        return patterns
    
    def compute_attention_entropy(
        self,
        attention_pattern: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropy of attention distribution for each query position.
        
        Low entropy = focused attention
        High entropy = diffuse attention
        
        Args:
            attention_pattern: Attention weights (seq_len, seq_len)
            
        Returns:
            Entropy at each query position
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        attn = attention_pattern.clamp(min=eps)
        
        # Entropy: -sum(p * log(p))
        entropy = -(attn * attn.log()).sum(dim=-1)
        
        return entropy
    
    def identify_head_types(
        self,
        texts: List[str],
    ) -> List[AttentionHeadInfo]:
        """
        Classify attention heads by their behavior patterns.
        
        Args:
            texts: Sample texts to analyze
            
        Returns:
            List of AttentionHeadInfo for each head
        """
        head_info = []
        
        # Collect patterns across samples
        all_patterns = {(l, h): [] for l in range(self.n_layers) for h in range(self.n_heads)}
        
        for text in texts:
            patterns = self.get_attention_patterns(text)
            for key, pattern in patterns.items():
                all_patterns[key].append(pattern)
        
        for (layer_idx, head_idx), patterns in all_patterns.items():
            if not patterns:
                continue
            
            # Stack patterns and compute average
            avg_pattern = torch.stack(patterns).mean(dim=0)
            
            # Compute metrics
            entropy = self.compute_attention_entropy(avg_pattern).mean().item()
            
            # Determine pattern type based on attention distribution
            pattern_type = self._classify_pattern(avg_pattern)
            
            # Importance: heads with lower entropy are more focused
            importance = 1.0 / (entropy + 1e-3)
            
            head_info.append(AttentionHeadInfo(
                layer_idx=layer_idx,
                head_idx=head_idx,
                importance_score=importance,
                pattern_type=pattern_type,
                avg_entropy=entropy,
            ))
        
        # Sort by importance
        head_info.sort(key=lambda x: x.importance_score, reverse=True)
        return head_info
    
    def _classify_pattern(self, pattern: torch.Tensor) -> str:
        """Classify attention pattern type."""
        seq_len = pattern.shape[0]
        
        if seq_len < 2:
            return "trivial"
        
        # Check for local attention (attending to nearby tokens)
        diag_sum = 0
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                if j <= i:  # Causal
                    diag_sum += pattern[i, j].item()
        local_ratio = diag_sum / pattern.sum().item()
        
        if local_ratio > 0.7:
            return "local"
        
        # Check for previous token attention
        prev_token_attn = 0
        for i in range(1, seq_len):
            prev_token_attn += pattern[i, i-1].item()
        prev_ratio = prev_token_attn / pattern.sum().item()
        
        if prev_ratio > 0.5:
            return "previous_token"
        
        # Check for first token attention (BOS/instruction focusing)
        first_token_attn = pattern[:, 0].sum().item()
        first_ratio = first_token_attn / pattern.sum().item()
        
        if first_ratio > 0.5:
            return "first_token"
        
        # Check for global/diffuse attention
        entropy = self.compute_attention_entropy(pattern).mean().item()
        max_entropy = np.log(seq_len)
        
        if entropy > 0.8 * max_entropy:
            return "global"
        
        return "mixed"
    
    def find_safety_heads(
        self,
        safe_texts: List[str],
        unsafe_texts: List[str],
        top_k: int = 10,
    ) -> List[AttentionHeadInfo]:
        """
        Identify attention heads that behave differently for safe vs unsafe inputs.
        
        These heads are candidates for "safety heads" that detect harmful content.
        
        Args:
            safe_texts: Safe/benign inputs
            unsafe_texts: Unsafe/harmful inputs
            top_k: Number of heads to return
            
        Returns:
            List of most discriminative attention heads
        """
        head_scores = {}
        
        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                safe_entropies = []
                unsafe_entropies = []
                
                for text in safe_texts:
                    patterns = self.get_attention_patterns(text, [layer_idx], [head_idx])
                    if (layer_idx, head_idx) in patterns:
                        ent = self.compute_attention_entropy(patterns[(layer_idx, head_idx)])
                        safe_entropies.append(ent.mean().item())
                
                for text in unsafe_texts:
                    patterns = self.get_attention_patterns(text, [layer_idx], [head_idx])
                    if (layer_idx, head_idx) in patterns:
                        ent = self.compute_attention_entropy(patterns[(layer_idx, head_idx)])
                        unsafe_entropies.append(ent.mean().item())
                
                if safe_entropies and unsafe_entropies:
                    safe_mean = np.mean(safe_entropies)
                    unsafe_mean = np.mean(unsafe_entropies)
                    safe_std = np.std(safe_entropies) + 1e-6
                    unsafe_std = np.std(unsafe_entropies) + 1e-6
                    
                    # Discriminative score: normalized difference
                    score = abs(safe_mean - unsafe_mean) / ((safe_std + unsafe_std) / 2)
                    head_scores[(layer_idx, head_idx)] = score
        
        # Get top-k most discriminative
        sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for (layer_idx, head_idx), score in sorted_heads[:top_k]:
            results.append(AttentionHeadInfo(
                layer_idx=layer_idx,
                head_idx=head_idx,
                importance_score=score,
                pattern_type="safety_candidate",
                avg_entropy=0.0,  # Not computed here
            ))
        
        return results
    
    def measure_attention_shift(
        self,
        text1: str,
        text2: str,
        layer_idx: int,
        head_idx: int,
    ) -> AttentionShift:
        """
        Measure how attention changes between two inputs.
        
        Useful for detecting attention hijacking during attacks.
        
        Args:
            text1: First input (e.g., original)
            text2: Second input (e.g., with attack suffix)
            layer_idx: Layer to analyze
            head_idx: Head to analyze
            
        Returns:
            AttentionShift with various metrics
        """
        patterns1 = self.get_attention_patterns(text1, [layer_idx], [head_idx])
        patterns2 = self.get_attention_patterns(text2, [layer_idx], [head_idx])
        
        attn1 = patterns1.get((layer_idx, head_idx))
        attn2 = patterns2.get((layer_idx, head_idx))
        
        if attn1 is None or attn2 is None:
            raise ValueError(f"Could not get attention for layer {layer_idx}, head {head_idx}")
        
        # Use last query position for comparison
        p1 = attn1[-1, :]
        p2 = attn2[-1, :]
        
        # May have different lengths, truncate to shorter
        min_len = min(len(p1), len(p2))
        p1 = p1[:min_len]
        p2 = p2[:min_len]
        
        # Normalize
        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()
        
        # KL divergence
        eps = 1e-10
        kl_div = float((p1 * (p1 / (p2 + eps)).log()).sum())
        
        # JS divergence (symmetric)
        m = (p1 + p2) / 2
        js_div = 0.5 * float((p1 * (p1 / (m + eps)).log()).sum())
        js_div += 0.5 * float((p2 * (p2 / (m + eps)).log()).sum())
        
        # Max shift
        max_shift = float((p1 - p2).abs().max())
        
        # Top focused positions
        source_focus = p1.argsort(descending=True)[:3].tolist()
        target_focus = p2.argsort(descending=True)[:3].tolist()
        
        return AttentionShift(
            layer_idx=layer_idx,
            head_idx=head_idx,
            kl_divergence=kl_div,
            js_divergence=js_div,
            max_shift=max_shift,
            source_focus=source_focus,
            target_focus=target_focus,
        )
    
    def detect_hijacking(
        self,
        original_text: str,
        attacked_text: str,
        safety_heads: List[AttentionHeadInfo],
        threshold: float = 0.5,
    ) -> Dict[Tuple[int, int], bool]:
        """
        Detect if safety heads have been hijacked.
        
        Args:
            original_text: Original input
            attacked_text: Input with attack
            safety_heads: List of safety heads to check
            threshold: JS divergence threshold for hijacking
            
        Returns:
            Dictionary mapping (layer, head) to hijack detection
        """
        results = {}
        
        for head in safety_heads:
            shift = self.measure_attention_shift(
                original_text,
                attacked_text,
                head.layer_idx,
                head.head_idx,
            )
            
            # Consider hijacked if attention significantly changed
            hijacked = shift.js_divergence > threshold
            results[(head.layer_idx, head.head_idx)] = hijacked
        
        return results
