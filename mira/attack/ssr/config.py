"""
Configuration classes for SSR attacks.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SSRConfig(BaseModel):
    """Base configuration for Subspace Rerouting attacks."""
    
    model_name: str = Field(description="Model identifier")
    
    # Optimization parameters
    search_width: int = Field(default=256, description="Number of candidates to try per iteration")
    search_topk: int = Field(default=64, description="Top-k tokens to sample from gradients")
    buffer_size: int = Field(default=10, description="Number of best candidates to keep")
    
    # Adaptive replacement
    replace_coefficient: float = Field(default=1.8, description="Controls how n_replace decreases")
    n_replace: Optional[int] = Field(default=None, description="Number of tokens to replace (computed dynamically)")
    
    # Layer targeting
    max_layer: int = Field(default=-1, description="Maximum layer to compute gradients (negative for reverse indexing)")
    
    # Stopping criteria
    patience: int = Field(default=10, description="Iterations without improvement before jumping")
    early_stop_loss: float = Field(default=0.05, description="Stop if loss below this threshold")
    max_iterations: int = Field(default=60, description="Maximum optimization iterations")
    
    # Token filtering
    filter_tokens: bool = Field(default=True, description="Filter tokens that don't re-encode correctly")
    restrict_nonascii: bool = Field(default=True, description="Restrict to ASCII printable tokens")
    
    class Config:
        arbitrary_types_allowed = True


class ProbeSSRConfig(SSRConfig):
    """Configuration for Probe-based SSR."""
    
    layers: List[int] = Field(description="Layers to target with probes")
    alphas: List[float] = Field(description="Weight for each layer's probe loss")
    pattern: str = Field(default="resid_post", description="Activation pattern to hook")
    
    # Probe training
    probe_hidden_dim: Optional[int] = Field(default=None, description="Hidden dimension for probe (None = linear)")
    probe_epochs: int = Field(default=10, description="Training epochs for probes")
    probe_lr: float = Field(default=0.001, description="Learning rate for probe training")
    probe_batch_size: int = Field(default=32, description="Batch size for probe training")


class SteeringSSRConfig(SSRConfig):
    """Configuration for Steering-based SSR."""
    
    layers: List[int] = Field(description="Layers to target with steering")
    alphas: List[float] = Field(description="Weight for each layer's steering loss")
    pattern: str = Field(default="resid_post", description="Activation pattern to hook")
    
    # Direction computation
    num_samples: int = Field(default=100, description="Number of samples for direction computation")
    normalize_directions: bool = Field(default=True, description="Normalize refusal directions")

