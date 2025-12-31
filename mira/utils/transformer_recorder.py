"""
Transformer State Recorder

Records all transformer internal state changes during forward passes.
Saves to JSON format for analysis and reporting.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import numpy as np


@dataclass
class LayerStateRecord:
    """Record of a single layer's state at one point in time."""
    layer_idx: int
    timestamp: str
    
    # Attention components
    attention_weights: Optional[List[List[float]]] = None  # [num_heads, seq_len, seq_len]
    attention_entropy: Optional[float] = None
    attention_max_weight: Optional[float] = None
    
    # Activation components
    residual_pre_norm: Optional[float] = None
    residual_post_norm: Optional[float] = None
    mlp_activation_norm: Optional[float] = None
    
    # Probe predictions (if available)
    probe_refusal_score: Optional[float] = None
    probe_acceptance_score: Optional[float] = None
    
    # Token predictions
    top_token: Optional[str] = None
    top_token_prob: Optional[float] = None


@dataclass
class ForwardPassRecord:
    """Complete record of one forward pass."""
    prompt: str
    timestamp: str
    is_attack: bool
    attack_type: Optional[str] = None
    success: Optional[bool] = None
    
    # Input
    tokens: List[str] = None
    token_ids: List[int] = None
    
    # Layer states
    layer_states: List[LayerStateRecord] = None
    
    # Final output
    final_logits_top5: List[Dict[str, Any]] = None
    response: Optional[str] = None
    
    # Metadata
    model_name: Optional[str] = None
    num_layers: Optional[int] = None


class TransformerRecorder:
    """
    Records all transformer state changes during analysis.
    
    Saves records to JSON files for later analysis and reporting.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize recorder.
        
        Args:
            output_dir: Directory to save JSON records
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.records: List[ForwardPassRecord] = []
        self.current_record: Optional[ForwardPassRecord] = None
    
    def start_forward_pass(
        self,
        prompt: str,
        is_attack: bool = False,
        attack_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Start recording a new forward pass.
        
        Args:
            prompt: Input prompt
            is_attack: Whether this is an attack prompt
            attack_type: Type of attack (if applicable)
            model_name: Name of model being used
        """
        self.current_record = ForwardPassRecord(
            prompt=prompt,
            timestamp=datetime.now().isoformat(),
            is_attack=is_attack,
            attack_type=attack_type,
            model_name=model_name,
            tokens=[],
            token_ids=[],
            layer_states=[],
            final_logits_top5=[],
        )
    
    def record_tokens(self, tokens: List[str], token_ids: torch.Tensor) -> None:
        """
        Record input tokens.
        
        Args:
            tokens: List of token strings
            token_ids: Token IDs tensor
        """
        if self.current_record:
            self.current_record.tokens = tokens
            if isinstance(token_ids, torch.Tensor):
                self.current_record.token_ids = token_ids.cpu().tolist()
            else:
                self.current_record.token_ids = list(token_ids)
    
    def record_layer_state(
        self,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None,
        residual_pre: Optional[torch.Tensor] = None,
        residual_post: Optional[torch.Tensor] = None,
        mlp_activation: Optional[torch.Tensor] = None,
        probe_refusal: Optional[float] = None,
        probe_acceptance: Optional[float] = None,
        top_token: Optional[str] = None,
        top_token_prob: Optional[float] = None,
    ) -> None:
        """
        Record state of one layer.
        
        Args:
            layer_idx: Layer index
            attention_weights: Attention weight tensor [num_heads, seq_len, seq_len]
            residual_pre: Residual stream before layer
            residual_post: Residual stream after layer
            mlp_activation: MLP intermediate activation
            probe_refusal: Probe refusal score (if available)
            probe_acceptance: Probe acceptance score (if available)
            top_token: Top predicted token at this layer
            top_token_prob: Probability of top token
        """
        if not self.current_record:
            return
        
        # Convert attention weights to list
        attn_list = None
        attn_entropy = None
        attn_max = None
        
        if attention_weights is not None:
            try:
                # Average over heads and convert to numpy
                if attention_weights.dim() >= 3:
                    attn_avg = attention_weights.mean(dim=0).detach().cpu().numpy()
                    attn_list = attn_avg.tolist()
                    
                    # Compute entropy
                    attn_probs = attn_avg / (attn_avg.sum(axis=-1, keepdims=True) + 1e-9)
                    entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-9), axis=-1)
                    attn_entropy = float(entropy.mean())
                    attn_max = float(attn_avg.max())
            except Exception:
                pass
        
        # Compute norms
        residual_pre_norm = None
        residual_post_norm = None
        mlp_norm = None
        
        if residual_pre is not None:
            try:
                residual_pre_norm = float(residual_pre.norm().item())
            except Exception:
                pass
        
        if residual_post is not None:
            try:
                residual_post_norm = float(residual_post.norm().item())
            except Exception:
                pass
        
        if mlp_activation is not None:
            try:
                mlp_norm = float(mlp_activation.norm().item())
            except Exception:
                pass
        
        layer_state = LayerStateRecord(
            layer_idx=layer_idx,
            timestamp=datetime.now().isoformat(),
            attention_weights=attn_list,
            attention_entropy=attn_entropy,
            attention_max_weight=attn_max,
            residual_pre_norm=residual_pre_norm,
            residual_post_norm=residual_post_norm,
            mlp_activation_norm=mlp_norm,
            probe_refusal_score=probe_refusal,
            probe_acceptance_score=probe_acceptance,
            top_token=top_token,
            top_token_prob=top_token_prob,
        )
        
        self.current_record.layer_states.append(layer_state)
    
    def record_final_output(
        self,
        logits: torch.Tensor,
        tokenizer: Any,
        response: Optional[str] = None,
    ) -> None:
        """
        Record final model output.
        
        Args:
            logits: Final logits tensor [seq_len, vocab_size]
            tokenizer: Tokenizer for decoding
            response: Generated response text (if available)
        """
        if not self.current_record:
            return
        
        # Get top 5 predictions
        try:
            if logits.dim() == 2:
                last_logits = logits[-1, :]
            else:
                last_logits = logits[-1] if logits.dim() == 1 else logits[0, -1, :]
            
            probs = torch.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5)
            
            top5 = []
            for prob, idx in zip(top_probs, top_indices):
                token = tokenizer.decode([idx.item()])
                top5.append({
                    "token": token,
                    "probability": float(prob.item()),
                    "logit": float(last_logits[idx].item()),
                })
            
            self.current_record.final_logits_top5 = top5
        except Exception:
            pass
        
        if response:
            self.current_record.response = response
    
    def finish_forward_pass(
        self,
        success: Optional[bool] = None,
        num_layers: Optional[int] = None,
    ) -> ForwardPassRecord:
        """
        Finish recording current forward pass.
        
        Args:
            success: Whether attack succeeded (if applicable)
            num_layers: Total number of layers in model
            
        Returns:
            The completed ForwardPassRecord
        """
        if not self.current_record:
            return None
        
        self.current_record.success = success
        self.current_record.num_layers = num_layers
        
        record = self.current_record
        self.records.append(record)
        self.current_record = None
        
        return record
    
    def save_record(self, record: ForwardPassRecord, filename: Optional[str] = None) -> Path:
        """
        Save a single record to JSON file.
        
        Args:
            record: ForwardPassRecord to save
            filename: Optional custom filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = record.timestamp.replace(":", "-").replace(".", "-")
            prefix = "attack" if record.is_attack else "baseline"
            filename = f"{prefix}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert to dict, handling nested dataclasses
        def convert_to_dict(obj):
            if isinstance(obj, (ForwardPassRecord, LayerStateRecord)):
                return asdict(obj)
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        data = convert_to_dict(record)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def save_all_records(self, filename: str = "all_transformer_records.json") -> Path:
        """
        Save all records to a single JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        
        def convert_to_dict(obj):
            if isinstance(obj, (ForwardPassRecord, LayerStateRecord)):
                return asdict(obj)
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        data = {
            "metadata": {
                "total_records": len(self.records),
                "generated_at": datetime.now().isoformat(),
                "output_dir": str(self.output_dir),
            },
            "records": [convert_to_dict(r) for r in self.records],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def compare_baseline_vs_attack(
        self,
        baseline_records: List[ForwardPassRecord],
        attack_records: List[ForwardPassRecord],
    ) -> Dict[str, Any]:
        """
        Compare baseline and attack records to compute differences.
        
        Args:
            baseline_records: List of baseline forward pass records
            attack_records: List of attack forward pass records
            
        Returns:
            Dictionary with comparison statistics
        """
        if not baseline_records or not attack_records:
            return {}
        
        comparison = {
            "num_baseline": len(baseline_records),
            "num_attacks": len(attack_records),
            "layer_differences": [],
        }
        
        # Compare layer by layer
        max_layers = max(
            max(len(r.layer_states) for r in baseline_records),
            max(len(r.layer_states) for r in attack_records),
        )
        
        for layer_idx in range(max_layers):
            baseline_states = [
                r.layer_states[layer_idx]
                for r in baseline_records
                if layer_idx < len(r.layer_states)
            ]
            attack_states = [
                r.layer_states[layer_idx]
                for r in attack_records
                if layer_idx < len(r.layer_states)
            ]
            
            if not baseline_states or not attack_states:
                continue
            
            # Compute average differences
            baseline_attn_entropy = np.mean([
                s.attention_entropy for s in baseline_states if s.attention_entropy is not None
            ]) if any(s.attention_entropy is not None for s in baseline_states) else None
            
            attack_attn_entropy = np.mean([
                s.attention_entropy for s in attack_states if s.attention_entropy is not None
            ]) if any(s.attention_entropy is not None for s in attack_states) else None
            
            baseline_residual = np.mean([
                s.residual_post_norm for s in baseline_states if s.residual_post_norm is not None
            ]) if any(s.residual_post_norm is not None for s in baseline_states) else None
            
            attack_residual = np.mean([
                s.residual_post_norm for s in attack_states if s.residual_post_norm is not None
            ]) if any(s.residual_post_norm is not None for s in attack_states) else None
            
            layer_diff = {
                "layer_idx": layer_idx,
                "attention_entropy": {
                    "baseline_mean": float(baseline_attn_entropy) if baseline_attn_entropy is not None else None,
                    "attack_mean": float(attack_attn_entropy) if attack_attn_entropy is not None else None,
                    "difference": float(attack_attn_entropy - baseline_attn_entropy) if (baseline_attn_entropy is not None and attack_attn_entropy is not None) else None,
                },
                "residual_norm": {
                    "baseline_mean": float(baseline_residual) if baseline_residual is not None else None,
                    "attack_mean": float(attack_residual) if attack_residual is not None else None,
                    "difference": float(attack_residual - baseline_residual) if (baseline_residual is not None and attack_residual is not None) else None,
                },
            }
            
            comparison["layer_differences"].append(layer_diff)
        
        return comparison

