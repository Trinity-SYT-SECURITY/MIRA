"""
Steering-based Subspace Rerouting (SSR).

Computes refusal direction vectors from activation differences between
safe and harmful prompts, then optimizes adversarial tokens to move away
from refusal directions.
"""

from typing import Dict, Optional, Callable
import torch
import torch.nn.functional as F

from mira.core.model_wrapper import ModelWrapper
from mira.attack.ssr.core import SSRAttack
from mira.attack.ssr.config import SteeringSSRConfig


class SteeringSSR(SSRAttack):
    """
    Steering-based Subspace Rerouting attack.
    
    Computes refusal direction vectors for each layer by analyzing
    activation differences, then optimizes to minimize projection onto
    these directions.
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        config: SteeringSSRConfig,
        refusal_directions: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, float, str], None]] = None,
    ):
        """
        Initialize Steering SSR.
        
        Args:
            model: Model wrapper
            config: Steering SSR configuration
            refusal_directions: Optional pre-computed refusal directions [n_layers, d_model]
            callback: Optional progress callback
        """
        super().__init__(model, config, callback)
        self.config: SteeringSSRConfig = config
        
        # Refusal directions: [n_layers, d_model]
        self.refusal_directions: Optional[torch.Tensor] = refusal_directions
        
        # Verify layers and alphas match
        if len(config.layers) != len(config.alphas):
            raise ValueError(f"layers ({len(config.layers)}) and alphas ({len(config.alphas)}) must have same length")
        
        # Hook handles
        self._hook_handles = []
    
    def _register_hooks(self) -> None:
        """Register forward hooks to capture activations."""
        self.activation_cache = {}
        
        def make_hook(layer_idx: int, pattern: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                self.activation_cache[f"layer_{layer_idx}_{pattern}"] = act.float()
            return hook
        
        # Get transformer blocks
        blocks = self.model._get_transformer_blocks()
        
        # Register hooks for target layers
        for layer_idx in self.config.layers:
            if layer_idx < len(blocks):
                block = blocks[layer_idx]
                
                # Hook on block output for residual post
                if self.config.pattern == "resid_post":
                    handle = block.register_forward_hook(
                        make_hook(layer_idx, self.config.pattern)
                    )
                    self._hook_handles.append(handle)
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
    
    def compute_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute steering-based loss.
        
        Loss is the sum of projections onto refusal directions across layers.
        We want to maximize negative projection (move away from refusal).
        
        Args:
            embeddings: [batch_size, seq_len, d_model] input embeddings
            
        Returns:
            [batch_size] losses
        """
        if self.refusal_directions is None:
            raise ValueError("Refusal directions not computed. Call compute_refusal_directions() first.")
        
        # Register hooks
        self._register_hooks()
        
        try:
            # Forward pass through model
            with torch.set_grad_enabled(embeddings.requires_grad):
                outputs = self.model.model(
                    inputs_embeds=embeddings,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            batch_size = embeddings.shape[0]
            total_loss = torch.zeros(batch_size, device=self.device)
            
            # Compute projection onto refusal directions for each layer
            for i, layer_idx in enumerate(self.config.layers):
                alpha = self.config.alphas[i]
                
                # Get activation for this layer
                act_key = f"layer_{layer_idx}_{self.config.pattern}"
                if act_key in self.activation_cache:
                    act = self.activation_cache[act_key]
                else:
                    # Fallback: use hidden states from output
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        if layer_idx < len(outputs.hidden_states):
                            act = outputs.hidden_states[layer_idx]
                        else:
                            continue
                    else:
                        continue
                
                # Use last token position
                last_token_act = act[:, -1, :]  # [batch_size, d_model]
                
                # Get refusal direction for this layer
                refusal_dir = self.refusal_directions[layer_idx].to(self.device)  # [d_model]
                
                # Compute projection: dot product with refusal direction
                # Positive projection = moving toward refusal (bad)
                # Negative projection = moving away from refusal (good)
                projection = torch.sum(last_token_act * refusal_dir, dim=-1)  # [batch_size]
                
                # Add to loss (we want to minimize positive projection)
                total_loss += alpha * projection
            
            return total_loss
        
        finally:
            # Always remove hooks
            self._remove_hooks()
    
    def compute_refusal_directions(
        self,
        safe_prompts: list[str],
        harmful_prompts: list[str],
        save_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute refusal direction vectors from prompt differences.
        
        Refusal direction = mean(harmful_acts) - mean(safe_acts)
        
        Args:
            safe_prompts: List of safe prompts (should be accepted)
            harmful_prompts: List of harmful prompts (should be refused)
            save_path: Optional path to save directions
            
        Returns:
            [n_layers, d_model] refusal direction vectors
        """
        print(f"Computing refusal directions from {len(safe_prompts)} safe and {len(harmful_prompts)} harmful prompts...")
        
        # Limit number of samples if specified
        if self.config.num_samples > 0:
            safe_prompts = safe_prompts[:self.config.num_samples]
            harmful_prompts = harmful_prompts[:self.config.num_samples]
        
        # Collect activations
        safe_acts = self._collect_activations(safe_prompts)
        harmful_acts = self._collect_activations(harmful_prompts)
        
        # Compute directions for each layer
        refusal_directions = []
        
        for layer_idx in self.config.layers:
            # Get mean activations
            safe_mean = safe_acts[layer_idx].mean(dim=0)  # [d_model]
            harmful_mean = harmful_acts[layer_idx].mean(dim=0)  # [d_model]
            
            # Refusal direction = harmful - safe
            direction = harmful_mean - safe_mean
            
            # Normalize if specified
            if self.config.normalize_directions:
                direction = direction / (direction.norm() + 1e-8)
            
            refusal_directions.append(direction)
            
            # Print statistics
            norm = direction.norm().item()
            print(f"Layer {layer_idx}: direction norm = {norm:.4f}")
        
        # Stack into tensor
        self.refusal_directions = torch.stack(refusal_directions)  # [n_layers, d_model]
        
        # Save if path provided
        if save_path is not None:
            import os
            import json
            
            os.makedirs(save_path, exist_ok=True)
            
            # Save directions
            torch.save(self.refusal_directions, f"{save_path}/refusal_directions.pt")
            
            # Save metadata
            metadata = {
                "model_name": self.model.model_name,
                "layers": self.config.layers,
                "alphas": self.config.alphas,
                "normalize_directions": self.config.normalize_directions,
                "num_samples": len(safe_prompts),
            }
            with open(f"{save_path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved refusal directions to {save_path}")
        
        return self.refusal_directions
    
    def load_refusal_directions(self, load_path: str) -> None:
        """
        Load pre-computed refusal directions from disk.
        
        Args:
            load_path: Directory containing refusal_directions.pt and metadata.json
        """
        import json
        import os
        
        # Load directions
        directions_path = f"{load_path}/refusal_directions.pt"
        if not os.path.exists(directions_path):
            raise FileNotFoundError(f"Refusal directions not found at {directions_path}")
        
        self.refusal_directions = torch.load(directions_path, map_location=self.device)
        
        # Load metadata
        metadata_path = f"{load_path}/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"Loaded refusal directions from {load_path}")
            print(f"Model: {metadata.get('model_name')}")
            print(f"Layers: {metadata.get('layers')}")
        else:
            print(f"Loaded refusal directions from {load_path} (no metadata)")
    
    def _collect_activations(self, prompts: list[str]) -> Dict[int, torch.Tensor]:
        """
        Collect activations for given prompts.
        
        Args:
            prompts: List of prompts
            
        Returns:
            Dictionary of {layer_idx: [n_prompts, d_model] activations}
        """
        layer_acts = {layer: [] for layer in self.config.layers}
        
        self._register_hooks()
        
        try:
            for prompt in prompts:
                # Tokenize
                inputs = self.model.tokenize(prompt)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                
                # Extract activations for each layer
                for layer_idx in self.config.layers:
                    act_key = f"layer_{layer_idx}_{self.config.pattern}"
                    if act_key in self.activation_cache:
                        act = self.activation_cache[act_key]
                    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        if layer_idx < len(outputs.hidden_states):
                            act = outputs.hidden_states[layer_idx]
                        else:
                            continue
                    else:
                        continue
                    
                    # Use last token
                    last_token_act = act[0, -1, :].cpu()  # [d_model]
                    layer_acts[layer_idx].append(last_token_act)
        
        finally:
            self._remove_hooks()
        
        # Stack activations
        for layer_idx in self.config.layers:
            if layer_acts[layer_idx]:
                layer_acts[layer_idx] = torch.stack(layer_acts[layer_idx])
        
        return layer_acts
    
    def visualize_directions(self, save_path: Optional[str] = None) -> None:
        """
        Visualize refusal directions using PCA.
        
        Args:
            save_path: Optional path to save visualization
        """
        if self.refusal_directions is None:
            raise ValueError("No refusal directions to visualize")
        
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
        except ImportError:
            print("sklearn and matplotlib required for visualization")
            return
        
        # Apply PCA to reduce to 2D
        directions_np = self.refusal_directions.cpu().numpy()
        pca = PCA(n_components=2)
        directions_2d = pca.fit_transform(directions_np)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(directions_2d[:, 0], directions_2d[:, 1], s=100, alpha=0.6)
        
        # Label each point with layer index
        for i, layer_idx in enumerate(self.config.layers):
            plt.annotate(
                f"L{layer_idx}",
                (directions_2d[i, 0], directions_2d[i, 1]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.title("Refusal Direction Vectors (PCA Projection)")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()

