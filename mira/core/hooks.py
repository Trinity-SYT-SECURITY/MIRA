"""
Core activation hook system for capturing internal model states.

Provides mechanisms to intercept and cache activations at any layer
during forward pass, enabling mechanistic analysis of model behavior.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import threading


@dataclass
class ActivationCache:
    """Container for cached activations from a forward pass."""
    
    residual: Dict[int, torch.Tensor] = field(default_factory=dict)
    attention_out: Dict[int, torch.Tensor] = field(default_factory=dict)
    attention_patterns: Dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_out: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    def clear(self):
        """Clear all cached activations."""
        self.residual.clear()
        self.attention_out.clear()
        self.attention_patterns.clear()
        self.mlp_out.clear()
    
    def get_layer(self, layer_idx: int, activation_type: str = "residual") -> Optional[torch.Tensor]:
        """Get activation for specific layer and type."""
        cache_map = {
            "residual": self.residual,
            "attention_out": self.attention_out,
            "attention_patterns": self.attention_patterns,
            "mlp_out": self.mlp_out,
        }
        return cache_map.get(activation_type, {}).get(layer_idx)
    
    def to_dict(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """Convert to dictionary format."""
        return {
            "residual": dict(self.residual),
            "attention_out": dict(self.attention_out),
            "attention_patterns": dict(self.attention_patterns),
            "mlp_out": dict(self.mlp_out),
        }


class HookHandle:
    """Handle for managing a registered hook."""
    
    def __init__(self, hook_id: str, remove_fn: Callable):
        self.hook_id = hook_id
        self._remove_fn = remove_fn
        self._active = True
    
    def remove(self):
        """Remove this hook."""
        if self._active:
            self._remove_fn(self.hook_id)
            self._active = False
    
    @property
    def active(self) -> bool:
        return self._active


class ActivationHookManager:
    """
    Manages activation hooks for transformer models.
    
    Supports capturing:
    - Residual stream (hidden states after each layer)
    - Attention outputs
    - Attention patterns (weights)
    - MLP outputs
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: Dict[str, Any] = {}
        self._cache = ActivationCache()
        self._hook_counter = 0
        self._lock = threading.Lock()
        self._capture_enabled = False
        
        # Detect model architecture
        self._detect_architecture()
    
    def _detect_architecture(self):
        """Detect model architecture and layer structure."""
        model_name = type(self.model).__name__.lower()
        
        # Common architectures
        if hasattr(self.model, 'transformer'):
            self._layers = self.model.transformer.h if hasattr(self.model.transformer, 'h') else []
            self._arch = "gpt2"
        elif hasattr(self.model, 'gpt_neox'):
            self._layers = self.model.gpt_neox.layers
            self._arch = "neox"
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self._layers = self.model.model.layers
            self._arch = "llama"
        else:
            # Fallback: try to find layers
            self._layers = []
            for name, module in self.model.named_modules():
                if 'layer' in name.lower() or 'block' in name.lower():
                    if not any(l == module for l in self._layers):
                        self._layers.append(module)
            self._arch = "generic"
        
        self.n_layers = len(self._layers)
    
    def _generate_hook_id(self) -> str:
        """Generate unique hook ID."""
        with self._lock:
            self._hook_counter += 1
            return f"hook_{self._hook_counter}"
    
    def _create_capture_hook(
        self, 
        layer_idx: int, 
        activation_type: str,
        callback: Optional[Callable] = None
    ) -> Callable:
        """Create a forward hook that captures activations."""
        def hook_fn(module, input_tensor, output):
            if not self._capture_enabled:
                return
            
            # Handle different output formats
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # Store in cache
            if activation_type == "residual":
                self._cache.residual[layer_idx] = activation.detach().clone()
            elif activation_type == "attention_out":
                self._cache.attention_out[layer_idx] = activation.detach().clone()
            elif activation_type == "mlp_out":
                self._cache.mlp_out[layer_idx] = activation.detach().clone()
            
            # Call custom callback if provided
            if callback:
                callback(layer_idx, activation_type, activation)
        
        return hook_fn
    
    def _create_attention_pattern_hook(
        self,
        layer_idx: int,
        callback: Optional[Callable] = None
    ) -> Callable:
        """Create hook specifically for attention patterns."""
        def hook_fn(module, input_tensor, output):
            if not self._capture_enabled:
                return
            
            # Attention patterns are typically in the second element of output tuple
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn_weights = output[1]
                self._cache.attention_patterns[layer_idx] = attn_weights.detach().clone()
                
                if callback:
                    callback(layer_idx, "attention_patterns", attn_weights)
        
        return hook_fn
    
    def register_residual_hook(
        self, 
        layer_idx: int,
        callback: Optional[Callable] = None
    ) -> HookHandle:
        """Register hook for residual stream at specific layer."""
        if layer_idx >= self.n_layers:
            raise ValueError(f"Layer {layer_idx} out of range (max: {self.n_layers-1})")
        
        hook_id = self._generate_hook_id()
        layer = self._layers[layer_idx]
        
        hook_fn = self._create_capture_hook(layer_idx, "residual", callback)
        handle = layer.register_forward_hook(hook_fn)
        
        self._hooks[hook_id] = handle
        return HookHandle(hook_id, lambda hid: self._remove_hook(hid))
    
    def register_attention_hook(
        self,
        layer_idx: int,
        callback: Optional[Callable] = None
    ) -> HookHandle:
        """Register hook for attention output at specific layer."""
        if layer_idx >= self.n_layers:
            raise ValueError(f"Layer {layer_idx} out of range")
        
        hook_id = self._generate_hook_id()
        layer = self._layers[layer_idx]
        
        # Find attention module within layer
        attn_module = None
        for name, module in layer.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if not 'norm' in name.lower():
                    attn_module = module
                    break
        
        if attn_module is None:
            attn_module = layer  # Fallback to layer itself
        
        hook_fn = self._create_capture_hook(layer_idx, "attention_out", callback)
        handle = attn_module.register_forward_hook(hook_fn)
        
        self._hooks[hook_id] = handle
        return HookHandle(hook_id, lambda hid: self._remove_hook(hid))
    
    def register_mlp_hook(
        self,
        layer_idx: int,
        callback: Optional[Callable] = None
    ) -> HookHandle:
        """Register hook for MLP output at specific layer."""
        if layer_idx >= self.n_layers:
            raise ValueError(f"Layer {layer_idx} out of range")
        
        hook_id = self._generate_hook_id()
        layer = self._layers[layer_idx]
        
        # Find MLP module within layer
        mlp_module = None
        for name, module in layer.named_modules():
            if 'mlp' in name.lower() or 'ff' in name.lower() or 'feedforward' in name.lower():
                mlp_module = module
                break
        
        if mlp_module is None:
            mlp_module = layer
        
        hook_fn = self._create_capture_hook(layer_idx, "mlp_out", callback)
        handle = mlp_module.register_forward_hook(hook_fn)
        
        self._hooks[hook_id] = handle
        return HookHandle(hook_id, lambda hid: self._remove_hook(hid))
    
    def register_all_layers(
        self,
        activation_types: List[str] = None
    ) -> List[HookHandle]:
        """Register hooks for all layers."""
        if activation_types is None:
            activation_types = ["residual"]
        
        handles = []
        for layer_idx in range(self.n_layers):
            for act_type in activation_types:
                if act_type == "residual":
                    handles.append(self.register_residual_hook(layer_idx))
                elif act_type == "attention_out":
                    handles.append(self.register_attention_hook(layer_idx))
                elif act_type == "mlp_out":
                    handles.append(self.register_mlp_hook(layer_idx))
        
        return handles
    
    def _remove_hook(self, hook_id: str):
        """Remove a hook by ID."""
        if hook_id in self._hooks:
            self._hooks[hook_id].remove()
            del self._hooks[hook_id]
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook_id in list(self._hooks.keys()):
            self._remove_hook(hook_id)
    
    def clear_cache(self):
        """Clear activation cache."""
        self._cache.clear()
    
    def enable_capture(self):
        """Enable activation capture."""
        self._capture_enabled = True
    
    def disable_capture(self):
        """Disable activation capture."""
        self._capture_enabled = False
    
    def run_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> Tuple[Optional[torch.Tensor], ActivationCache]:
        """
        Run forward pass and capture all registered activations.
        
        Returns:
            Tuple of (logits or None, ActivationCache)
        """
        self.clear_cache()
        self.enable_capture()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
            
            logits = outputs.logits if return_logits else None
            
            # Create a copy of the cache
            cache_copy = ActivationCache(
                residual=dict(self._cache.residual),
                attention_out=dict(self._cache.attention_out),
                attention_patterns=dict(self._cache.attention_patterns),
                mlp_out=dict(self._cache.mlp_out),
            )
            
            return logits, cache_copy
        
        finally:
            self.disable_capture()
    
    def get_cache(self) -> ActivationCache:
        """Get current activation cache."""
        return self._cache


class ActivationEditor:
    """
    Enables intervention on model activations during forward pass.
    
    Supports:
    - Replacing activations with custom values
    - Adding perturbations to activations
    - Steering activations toward/away from directions
    """
    
    def __init__(self, hook_manager: ActivationHookManager):
        self.hook_manager = hook_manager
        self._interventions: Dict[str, Any] = {}
    
    def _create_intervention_hook(
        self,
        layer_idx: int,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable:
        """Create a forward hook that modifies activations."""
        def hook_fn(module, input_tensor, output):
            if isinstance(output, tuple):
                modified = (intervention_fn(output[0]),) + output[1:]
                return modified
            else:
                return intervention_fn(output)
        
        return hook_fn
    
    def add_value(
        self, 
        layer_idx: int,
        direction: torch.Tensor,
        strength: float = 1.0
    ):
        """Add a value to activations at specified layer."""
        def intervention(activation):
            return activation + strength * direction.to(activation.device)
        
        # Register intervention hook
        layer = self.hook_manager._layers[layer_idx]
        hook_fn = self._create_intervention_hook(layer_idx, intervention)
        handle = layer.register_forward_hook(hook_fn)
        
        hook_id = f"intervention_{layer_idx}_{len(self._interventions)}"
        self._interventions[hook_id] = handle
    
    def subtract_direction(
        self,
        layer_idx: int,
        direction: torch.Tensor,
        strength: float = 1.0
    ):
        """Remove projection onto direction from activations."""
        direction_norm = direction / torch.norm(direction)
        
        def intervention(activation):
            # Project onto direction and subtract
            proj = torch.sum(activation * direction_norm, dim=-1, keepdim=True)
            return activation - strength * proj * direction_norm
        
        layer = self.hook_manager._layers[layer_idx]
        hook_fn = self._create_intervention_hook(layer_idx, intervention)
        handle = layer.register_forward_hook(hook_fn)
        
        hook_id = f"steer_{layer_idx}_{len(self._interventions)}"
        self._interventions[hook_id] = handle
    
    def replace_activation(
        self,
        layer_idx: int,
        replacement: torch.Tensor
    ):
        """Replace activations completely at specified layer."""
        def intervention(activation):
            return replacement.to(activation.device).expand_as(activation)
        
        layer = self.hook_manager._layers[layer_idx]
        hook_fn = self._create_intervention_hook(layer_idx, intervention)
        handle = layer.register_forward_hook(hook_fn)
        
        hook_id = f"replace_{layer_idx}_{len(self._interventions)}"
        self._interventions[hook_id] = handle
    
    def clear_interventions(self):
        """Remove all interventions."""
        for hook_id, handle in self._interventions.items():
            handle.remove()
        self._interventions.clear()


def get_model_layers(model: nn.Module) -> List[nn.Module]:
    """Extract transformer layers from model."""
    manager = ActivationHookManager(model)
    return list(manager._layers)


def get_embedding_matrix(model: nn.Module) -> torch.Tensor:
    """Get input embedding matrix from model."""
    if hasattr(model, 'get_input_embeddings'):
        return model.get_input_embeddings().weight
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte.weight
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight
    else:
        raise ValueError("Could not find embedding matrix")


def get_unembedding_matrix(model: nn.Module) -> torch.Tensor:
    """Get output unembedding matrix from model."""
    if hasattr(model, 'get_output_embeddings'):
        return model.get_output_embeddings().weight
    elif hasattr(model, 'lm_head'):
        return model.lm_head.weight
    else:
        raise ValueError("Could not find unembedding matrix")
