"""
Hook manager for model activation intervention.

Provides utilities for registering, managing, and applying hooks
for activation patching, steering, and ablation studies.
"""

from typing import Callable, Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn


class HookType(Enum):
    """Types of hooks that can be registered."""
    FORWARD = "forward"
    BACKWARD = "backward"
    FORWARD_PRE = "forward_pre"


@dataclass
class HookConfig:
    """Configuration for a registered hook."""
    
    name: str
    module_name: str
    hook_type: HookType
    callback: Callable
    enabled: bool = True


class ActivationModifier:
    """
    Base class for activation modification functions.
    
    Subclass this to create custom intervention behaviors.
    """
    
    def __call__(
        self,
        activation: torch.Tensor,
        layer_idx: int,
        token_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply modification to activation tensor."""
        raise NotImplementedError


class ZeroAblation(ActivationModifier):
    """Set activations to zero (ablation study)."""
    
    def __init__(
        self,
        neuron_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
    ):
        self.neuron_indices = neuron_indices
        self.head_indices = head_indices
    
    def __call__(
        self,
        activation: torch.Tensor,
        layer_idx: int,
        token_idx: Optional[int] = None,
    ) -> torch.Tensor:
        modified = activation.clone()
        
        if self.neuron_indices is not None:
            if token_idx is not None:
                modified[:, token_idx, self.neuron_indices] = 0
            else:
                modified[:, :, self.neuron_indices] = 0
        
        if self.head_indices is not None and activation.dim() == 4:
            for head_idx in self.head_indices:
                if token_idx is not None:
                    modified[:, head_idx, token_idx, :] = 0
                else:
                    modified[:, head_idx, :, :] = 0
        
        return modified


class AdditiveSteering(ActivationModifier):
    """Add a steering vector to activations."""
    
    def __init__(self, steering_vector: torch.Tensor, scale: float = 1.0):
        self.steering_vector = steering_vector
        self.scale = scale
    
    def __call__(
        self,
        activation: torch.Tensor,
        layer_idx: int,
        token_idx: Optional[int] = None,
    ) -> torch.Tensor:
        steering = self.steering_vector.to(activation.device)
        
        if token_idx is not None:
            modified = activation.clone()
            modified[:, token_idx, :] = modified[:, token_idx, :] + self.scale * steering
            return modified
        else:
            return activation + self.scale * steering


class ProjectionRemoval(ActivationModifier):
    """Remove the projection onto a direction (e.g., refusal direction)."""
    
    def __init__(self, direction: torch.Tensor):
        # Normalize the direction
        self.direction = direction / direction.norm()
    
    def __call__(
        self,
        activation: torch.Tensor,
        layer_idx: int,
        token_idx: Optional[int] = None,
    ) -> torch.Tensor:
        direction = self.direction.to(activation.device)
        
        # Compute projection: proj = (x . d) * d
        if token_idx is not None:
            x = activation[:, token_idx, :]
            proj = torch.einsum("bh,h->b", x, direction).unsqueeze(-1) * direction
            modified = activation.clone()
            modified[:, token_idx, :] = x - proj
            return modified
        else:
            proj = torch.einsum("bth,h->bt", activation, direction).unsqueeze(-1) * direction
            return activation - proj


class HookManager:
    """
    Manager for model hooks enabling activation intervention.
    
    Provides an interface for:
    - Registering hooks on specific modules
    - Enabling/disabling hooks dynamically
    - Applying activation modifications during forward pass
    - Collecting activations for analysis
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize hook manager.
        
        Args:
            model: The PyTorch model to apply hooks to
        """
        self.model = model
        self.hooks: Dict[str, HookConfig] = {}
        self.handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.collected_activations: Dict[str, List[torch.Tensor]] = {}
    
    def _get_module(self, module_name: str) -> nn.Module:
        """Get a module by its name path (e.g., 'transformer.h.0.attn')."""
        parts = module_name.split(".")
        current = self.model
        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        return current
    
    def register_hook(
        self,
        name: str,
        module_name: str,
        callback: Callable,
        hook_type: HookType = HookType.FORWARD,
    ) -> None:
        """
        Register a hook on a module.
        
        Args:
            name: Unique identifier for this hook
            module_name: Dot-separated path to the module
            callback: Function to call when hook fires
            hook_type: Type of hook to register
        """
        module = self._get_module(module_name)
        
        if hook_type == HookType.FORWARD:
            handle = module.register_forward_hook(callback)
        elif hook_type == HookType.BACKWARD:
            handle = module.register_full_backward_hook(callback)
        elif hook_type == HookType.FORWARD_PRE:
            handle = module.register_forward_pre_hook(callback)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
        
        config = HookConfig(
            name=name,
            module_name=module_name,
            hook_type=hook_type,
            callback=callback,
            enabled=True,
        )
        
        self.hooks[name] = config
        self.handles[name] = handle
    
    def remove_hook(self, name: str) -> None:
        """Remove a registered hook."""
        if name in self.handles:
            self.handles[name].remove()
            del self.handles[name]
            del self.hooks[name]
    
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles.values():
            handle.remove()
        self.hooks.clear()
        self.handles.clear()
        self.collected_activations.clear()
    
    def enable_hook(self, name: str) -> None:
        """Enable a hook."""
        if name in self.hooks:
            self.hooks[name].enabled = True
    
    def disable_hook(self, name: str) -> None:
        """Disable a hook."""
        if name in self.hooks:
            self.hooks[name].enabled = False
    
    def create_collection_hook(self, name: str) -> Callable:
        """Create a hook that collects activations."""
        self.collected_activations[name] = []
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            self.collected_activations[name].append(activation.detach().cpu())
        
        return hook
    
    def create_intervention_hook(
        self,
        modifier: ActivationModifier,
        layer_idx: int,
        token_idx: Optional[int] = None,
    ) -> Callable:
        """Create a hook that modifies activations."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
                modified = modifier(activation, layer_idx, token_idx)
                return (modified,) + output[1:]
            else:
                return modifier(output, layer_idx, token_idx)
        
        return hook
    
    def get_collected(self, name: str) -> List[torch.Tensor]:
        """Get collected activations for a hook."""
        return self.collected_activations.get(name, [])
    
    def clear_collected(self) -> None:
        """Clear all collected activations."""
        for key in self.collected_activations:
            self.collected_activations[key] = []
    
    def apply_steering(
        self,
        layer_idx: int,
        steering_vector: torch.Tensor,
        scale: float = 1.0,
    ) -> str:
        """
        Apply additive steering to a layer.
        
        Args:
            layer_idx: Index of the layer to steer
            steering_vector: Direction to steer activations
            scale: Scaling factor for steering
            
        Returns:
            Name of the registered hook
        """
        modifier = AdditiveSteering(steering_vector, scale)
        hook = self.create_intervention_hook(modifier, layer_idx)
        
        name = f"steering_layer_{layer_idx}"
        module_name = self._get_layer_module_name(layer_idx)
        self.register_hook(name, module_name, hook)
        
        return name
    
    def apply_ablation(
        self,
        layer_idx: int,
        neuron_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
    ) -> str:
        """
        Apply ablation (zeroing) to specific neurons or attention heads.
        
        Args:
            layer_idx: Index of the layer
            neuron_indices: Indices of neurons to ablate
            head_indices: Indices of attention heads to ablate
            
        Returns:
            Name of the registered hook
        """
        modifier = ZeroAblation(neuron_indices, head_indices)
        hook = self.create_intervention_hook(modifier, layer_idx)
        
        name = f"ablation_layer_{layer_idx}"
        module_name = self._get_layer_module_name(layer_idx)
        self.register_hook(name, module_name, hook)
        
        return name
    
    def remove_direction(
        self,
        layer_idx: int,
        direction: torch.Tensor,
    ) -> str:
        """
        Remove projection onto a direction from layer activations.
        
        Useful for removing refusal direction or other semantic directions.
        
        Args:
            layer_idx: Index of the layer
            direction: Direction to remove
            
        Returns:
            Name of the registered hook
        """
        modifier = ProjectionRemoval(direction)
        hook = self.create_intervention_hook(modifier, layer_idx)
        
        name = f"projection_removal_layer_{layer_idx}"
        module_name = self._get_layer_module_name(layer_idx)
        self.register_hook(name, module_name, hook)
        
        return name
    
    def _get_layer_module_name(self, layer_idx: int) -> str:
        """Get the module name for a transformer layer."""
        # Try common module naming patterns
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return f"transformer.h.{layer_idx}"
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                return f"model.layers.{layer_idx}"
        if hasattr(self.model, "gpt_neox"):
            return f"gpt_neox.layers.{layer_idx}"
        
        raise ValueError(f"Cannot determine layer module name for model: {type(self.model)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks."""
        self.remove_all_hooks()
        return False
