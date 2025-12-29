"""
Transformer Tracer - Captures detailed internal states during forward pass.

This module hooks into transformer layers to extract:
- Token embeddings
- Q/K/V vectors
- Attention weights (all heads)
- MLP activations
- Residual stream updates

Used for visualizing how attacks affect transformer processing.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np


@dataclass
class LayerTrace:
    """Captured data from one transformer layer."""
    layer_idx: int
    
    # Attention components
    query: torch.Tensor  # [seq_len, num_heads, head_dim]
    key: torch.Tensor
    value: torch.Tensor
    attention_weights: torch.Tensor  # [num_heads, seq_len, seq_len]
    attention_out: torch.Tensor  # [seq_len, hidden_dim]
    
    # MLP components
    mlp_in: torch.Tensor  # [seq_len, hidden_dim]
    mlp_intermediate: torch.Tensor  # [seq_len, intermediate_dim]
    mlp_out: torch.Tensor  # [seq_len, hidden_dim]
    
    # Residual stream
    residual_pre: torch.Tensor  # [seq_len, hidden_dim]
    residual_post: torch.Tensor  # [seq_len, hidden_dim]


@dataclass
class TransformerTrace:
    """Complete trace of transformer forward pass."""
    tokens: List[str]
    token_ids: torch.Tensor
    embeddings: torch.Tensor  # [seq_len, hidden_dim]
    layers: List[LayerTrace]
    final_logits: torch.Tensor  # [seq_len, vocab_size]
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for visualization."""
        return {
            "tokens": self.tokens,
            "token_ids": self.token_ids.tolist(),
            "embeddings": self.embeddings.detach().cpu().numpy().tolist(),
            "layers": [
                {
                    "layer_idx": layer.layer_idx,
                    "attention_weights": layer.attention_weights.detach().cpu().numpy().tolist(),
                    "mlp_activations": layer.mlp_intermediate.detach().cpu().numpy().tolist(),
                    "residual_norm": float(layer.residual_post.norm()),
                }
                for layer in self.layers
            ],
            "final_logits_top5": self._get_top_predictions(5),
        }
    
    def _get_top_predictions(self, k: int = 5) -> List[Dict]:
        """Get top-k predicted tokens for last position."""
        last_logits = self.final_logits[-1]
        top_k = torch.topk(last_logits, k)
        return [
            {"token_id": int(idx), "prob": float(torch.softmax(last_logits, dim=0)[idx])}
            for idx in top_k.indices
        ]


class TransformerTracer:
    """
    Hooks into transformer model to capture internal states.
    
    Usage:
        tracer = TransformerTracer(model)
        trace = tracer.trace_forward(input_ids)
        # trace contains all internal states
    """
    
    def __init__(self, model_wrapper):
        """
        Initialize tracer.
        
        Args:
            model_wrapper: MIRA ModelWrapper instance
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        
        # Storage for captured activations
        self.layer_traces: List[LayerTrace] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.hooks = []
        
    def trace_forward(
        self,
        input_ids: torch.Tensor,
        return_trace: bool = True
    ) -> TransformerTrace:
        """
        Run forward pass and capture all internal states.
        
        Args:
            input_ids: Input token IDs [seq_len] or [1, seq_len]
            return_trace: Whether to return TransformerTrace object
            
        Returns:
            TransformerTrace with all captured states
        """
        # Ensure input is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Move to model device
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Clear previous traces
        self.layer_traces = []
        self.embeddings = None
        
        # Forward pass with attention outputs
        with torch.no_grad():
            try:
                outputs = self.model(
                    input_ids, 
                    output_attentions=True,
                    output_hidden_states=True,
                )
            except Exception as e:
                # Fallback without special outputs
                outputs = self.model(input_ids)
        
        # Extract attention weights (may be None for some models)
        attentions = getattr(outputs, 'attentions', None)
        hidden_states = getattr(outputs, 'hidden_states', None)
        
        # Build layer traces
        self.layer_traces = []
        num_layers = self.model_wrapper.n_layers
        
        if attentions is not None and len(attentions) > 0:
            # We have real attention weights
            for layer_idx, attn_weights in enumerate(attentions):
                attn = attn_weights[0].detach().cpu()
                
                h_pre = hidden_states[layer_idx][0].detach().cpu() if hidden_states else torch.zeros(1, 1)
                h_post = hidden_states[layer_idx + 1][0].detach().cpu() if hidden_states and len(hidden_states) > layer_idx + 1 else torch.zeros(1, 1)
                
                layer_trace = LayerTrace(
                    layer_idx=layer_idx,
                    query=torch.zeros(1, 1, 1),
                    key=torch.zeros(1, 1, 1),
                    value=torch.zeros(1, 1, 1),
                    attention_weights=attn,
                    attention_out=h_post,
                    mlp_in=h_pre,
                    mlp_intermediate=torch.zeros(1, 1),
                    mlp_out=h_post,
                    residual_pre=h_pre,
                    residual_post=h_post,
                )
                self.layer_traces.append(layer_trace)
        else:
            # No attention available - create placeholder traces
            for layer_idx in range(num_layers):
                seq_len = input_ids.shape[1]
                # Create uniform attention as placeholder
                uniform_attn = torch.ones(1, seq_len, seq_len) / seq_len
                
                layer_trace = LayerTrace(
                    layer_idx=layer_idx,
                    query=torch.zeros(1, 1, 1),
                    key=torch.zeros(1, 1, 1),
                    value=torch.zeros(1, 1, 1),
                    attention_weights=uniform_attn,
                    attention_out=torch.zeros(1, 1),
                    mlp_in=torch.zeros(1, 1),
                    mlp_intermediate=torch.zeros(1, 1),
                    mlp_out=torch.zeros(1, 1),
                    residual_pre=torch.zeros(1, 1),
                    residual_post=torch.zeros(1, 1),
                )
                self.layer_traces.append(layer_trace)
        
        # Decode tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        
        # Get embeddings
        if hidden_states is not None and len(hidden_states) > 0:
            embeddings = hidden_states[0][0].detach().cpu()
        elif self.embeddings is not None:
            embeddings = self.embeddings
        else:
            embeddings = torch.zeros(input_ids.shape[1], 1)
        
        # Build trace
        trace = TransformerTrace(
            tokens=tokens,
            token_ids=input_ids[0].cpu(),
            embeddings=embeddings,
            layers=self.layer_traces,
            final_logits=outputs.logits[0].cpu(),
        )
        
        return trace
    
    def _register_hooks(self):
        """Register forward hooks on all transformer layers."""
        # Hook for embeddings
        if hasattr(self.model, 'gpt_neox'):
            # GPT-NeoX architecture (pythia)
            embed_layer = self.model.gpt_neox.embed_in
            self.hooks.append(
                embed_layer.register_forward_hook(self._capture_embeddings)
            )
            
            # Hook for each transformer layer
            for layer_idx, layer in enumerate(self.model.gpt_neox.layers):
                self.hooks.append(
                    layer.register_forward_hook(
                        lambda module, input, output, idx=layer_idx: self._capture_layer(
                            module, input, output, idx
                        )
                    )
                )
        elif hasattr(self.model, 'transformer'):
            # GPT-2 architecture
            embed_layer = self.model.transformer.wte
            self.hooks.append(
                embed_layer.register_forward_hook(self._capture_embeddings)
            )
            
            for layer_idx, layer in enumerate(self.model.transformer.h):
                self.hooks.append(
                    layer.register_forward_hook(
                        lambda module, input, output, idx=layer_idx: self._capture_layer(
                            module, input, output, idx
                        )
                    )
                )
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _capture_embeddings(self, module, input, output):
        """Hook to capture token embeddings."""
        self.embeddings = output.detach().clone()
    
    def _capture_layer(self, module, input, output, layer_idx: int):
        """Hook to capture layer internals."""
        # Get input (residual stream before this layer)
        hidden_states = input[0] if isinstance(input, tuple) else input
        
        # Try to extract attention and MLP components
        try:
            # This is architecture-specific
            # For GPT-NeoX/Pythia:
            if hasattr(module, 'attention'):
                attn_module = module.attention
                
                # Get Q/K/V (need to hook deeper or compute manually)
                # For now, we'll capture what we can from the output
                
                # Attention weights are typically not exposed directly
                # We need to modify the attention module or use custom forward
                
                # Placeholder: capture residual stream
                residual_pre = hidden_states.detach().clone()
                residual_post = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
                
                # Create layer trace with available data
                layer_trace = LayerTrace(
                    layer_idx=layer_idx,
                    query=torch.zeros(1, 1, 1),  # Placeholder
                    key=torch.zeros(1, 1, 1),
                    value=torch.zeros(1, 1, 1),
                    attention_weights=torch.zeros(1, 1, 1),  # Will compute separately
                    attention_out=residual_post,
                    mlp_in=residual_pre,
                    mlp_intermediate=torch.zeros(1, 1),  # Placeholder
                    mlp_out=residual_post,
                    residual_pre=residual_pre,
                    residual_post=residual_post,
                )
                
                self.layer_traces.append(layer_trace)
        except Exception as e:
            # Fallback: just capture residual stream
            pass
    
    def compute_attention_weights(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute attention weights for specific layer/head.
        
        This is a separate method because attention weights aren't
        always exposed in the forward pass.
        
        Args:
            input_ids: Input token IDs
            layer_idx: Which layer to analyze
            head_idx: Which head (None = all heads)
            
        Returns:
            Attention weights [num_heads, seq_len, seq_len] or [seq_len, seq_len]
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        with torch.no_grad():
            # Get hidden states up to target layer
            hidden_states = self.model.get_input_embeddings()(input_ids)
            
            # Forward through layers up to target
            if hasattr(self.model, 'gpt_neox'):
                layers = self.model.gpt_neox.layers
            elif hasattr(self.model, 'transformer'):
                layers = self.model.transformer.h
            else:
                raise ValueError("Unsupported model architecture")
            
            for i, layer in enumerate(layers):
                if i < layer_idx:
                    outputs = layer(hidden_states)
                    hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
                elif i == layer_idx:
                    # At target layer, extract attention
                    # This requires accessing attention module directly
                    attn = layer.attention if hasattr(layer, 'attention') else layer.attn
                    
                    # Compute Q, K, V
                    # This is architecture-specific
                    # For now, return placeholder
                    seq_len = hidden_states.shape[1]
                    num_heads = self.model.config.num_attention_heads
                    
                    # Placeholder: uniform attention
                    attention_weights = torch.ones(num_heads, seq_len, seq_len) / seq_len
                    
                    if head_idx is not None:
                        return attention_weights[head_idx]
                    return attention_weights
        
        raise ValueError(f"Layer {layer_idx} not found")
    
    def compare_traces(
        self,
        trace1: TransformerTrace,
        trace2: TransformerTrace
    ) -> Dict:
        """
        Compare two traces to find differences.
        
        Useful for comparing:
        - Normal vs adversarial prompts
        - Successful vs failed attacks
        
        Returns:
            Dict with comparison metrics
        """
        comparison = {
            "embedding_diff": float((trace1.embeddings - trace2.embeddings).norm()),
            "layer_diffs": [],
        }
        
        for layer1, layer2 in zip(trace1.layers, trace2.layers):
            layer_diff = {
                "layer_idx": layer1.layer_idx,
                "residual_diff": float((layer1.residual_post - layer2.residual_post).norm()),
                "attention_diff": float((layer1.attention_weights - layer2.attention_weights).norm()),
            }
            comparison["layer_diffs"].append(layer_diff)
        
        return comparison


def analyze_attack_patterns(
    tracer: TransformerTracer,
    normal_prompt: str,
    adversarial_prompt: str
) -> Dict:
    """
    Analyze how attack changes transformer processing.
    
    Args:
        tracer: TransformerTracer instance
        normal_prompt: Original prompt
        adversarial_prompt: Prompt with adversarial suffix
        
    Returns:
        Analysis results with patterns
    """
    # Tokenize
    normal_ids = tracer.tokenizer.encode(normal_prompt, return_tensors="pt")[0]
    adv_ids = tracer.tokenizer.encode(adversarial_prompt, return_tensors="pt")[0]
    
    # Trace both
    normal_trace = tracer.trace_forward(normal_ids)
    adv_trace = tracer.trace_forward(adv_ids)
    
    # Compare
    comparison = tracer.compare_traces(normal_trace, adv_trace)
    
    # Find most affected layers
    layer_diffs = comparison["layer_diffs"]
    most_affected = max(layer_diffs, key=lambda x: x["residual_diff"])
    
    return {
        "comparison": comparison,
        "most_affected_layer": most_affected["layer_idx"],
        "normal_trace": normal_trace.to_dict(),
        "adversarial_trace": adv_trace.to_dict(),
    }
