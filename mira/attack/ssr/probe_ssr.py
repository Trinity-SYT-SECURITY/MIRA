"""
Probe-based Subspace Rerouting (SSR).

Uses trained linear classifiers to detect refusal vs acceptance patterns
in model activations, then optimizes adversarial tokens to minimize refusal probability.
"""

from typing import Dict, Tuple, Optional, Callable
import torch
import torch.nn as nn

from mira.core.model_wrapper import ModelWrapper
from mira.attack.ssr.core import SSRAttack
from mira.attack.ssr.config import ProbeSSRConfig


class LinearProbe(nn.Module):
    """Linear classifier for refusal detection."""
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        """
        Initialize probe.
        
        Args:
            input_dim: Dimension of input activations
            hidden_dim: If provided, use 2-layer MLP; otherwise, use linear classifier
        """
        super().__init__()
        
        if hidden_dim is not None:
            # 2-layer MLP
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            # Linear classifier
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch_size, input_dim] activations
            
        Returns:
            [batch_size, 1] refusal probability (0 = acceptance, 1 = refusal)
        """
        return self.net(x)


class ProbeSSR(SSRAttack):
    """
    Probe-based Subspace Rerouting attack.
    
    Trains linear probes on each layer to classify refusal vs acceptance,
    then optimizes adversarial tokens to minimize refusal probability across layers.
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        config: ProbeSSRConfig,
        probes: Optional[Dict[int, Tuple[nn.Module, float, nn.Module]]] = None,
        callback: Optional[Callable[[int, float, str], None]] = None,
    ):
        """
        Initialize Probe SSR.
        
        Args:
            model: Model wrapper
            config: Probe SSR configuration
            probes: Optional pre-trained probes dict: {layer: (probe, alpha, loss_fn)}
            callback: Optional progress callback
        """
        super().__init__(model, config, callback)
        self.config: ProbeSSRConfig = config
        
        # Probes: {layer_idx: (probe_model, alpha_weight, loss_function)}
        self.probes: Dict[int, Tuple[nn.Module, float, nn.Module]] = probes or {}
        
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
                
                # Can add more patterns here (e.g., "mlp_out", "attn_out")
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
    
    def compute_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute probe-based loss.
        
        Args:
            embeddings: [batch_size, seq_len, d_model] input embeddings
            
        Returns:
            [batch_size] losses
        """
        # Register hooks
        self._register_hooks()
        
        try:
            # Forward pass through model
            with torch.set_grad_enabled(embeddings.requires_grad):
                # Run model up to max_layer
                # We need to manually run through layers since we're using embeddings
                outputs = self.model.model(
                    inputs_embeds=embeddings,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            batch_size = embeddings.shape[0]
            total_loss = torch.zeros(batch_size, device=self.device)
            
            # Compute loss from each probe
            for layer_idx, alpha in zip(self.config.layers, self.config.alphas):
                if layer_idx not in self.probes:
                    continue
                
                probe, weight, loss_fn = self.probes[layer_idx]
                
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
                
                # Get probe prediction
                pred = probe(last_token_act)  # [batch_size, 1]
                
                # Target is 0 (acceptance, not refusal)
                target = torch.zeros_like(pred)
                
                # Compute loss
                layer_loss = loss_fn(pred, target)
                if layer_loss.dim() > 1:
                    layer_loss = layer_loss.squeeze(-1)
                
                # Add weighted loss
                total_loss += alpha * layer_loss
            
            return total_loss
        
        finally:
            # Always remove hooks
            self._remove_hooks()
    
    def train_probes(
        self,
        safe_prompts: list[str],
        harmful_prompts: list[str],
        save_path: Optional[str] = None,
    ) -> Dict[int, float]:
        """
        Train linear probes to classify refusal vs acceptance.
        
        Args:
            safe_prompts: List of safe prompts (should be accepted)
            harmful_prompts: List of harmful prompts (should be refused)
            save_path: Optional path to save trained probes
            
        Returns:
            Dictionary of {layer_idx: accuracy}
        """
        print(f"Training probes on {len(safe_prompts)} safe and {len(harmful_prompts)} harmful prompts...")
        
        # Collect activations
        safe_acts = self._collect_activations(safe_prompts)
        harmful_acts = self._collect_activations(harmful_prompts)
        
        accuracies = {}
        
        # Train probe for each layer
        for layer_idx in self.config.layers:
            print(f"\nTraining probe for layer {layer_idx}...")
            
            # Get activations for this layer
            safe_layer_acts = safe_acts[layer_idx]  # [n_safe, d_model]
            harmful_layer_acts = harmful_acts[layer_idx]  # [n_harmful, d_model]
            
            # Create dataset
            X = torch.cat([safe_layer_acts, harmful_layer_acts], dim=0)
            y = torch.cat([
                torch.zeros(len(safe_layer_acts), 1),  # 0 = acceptance
                torch.ones(len(harmful_layer_acts), 1),  # 1 = refusal
            ], dim=0)
            
            # Shuffle
            perm = torch.randperm(len(X))
            X, y = X[perm], y[perm]
            
            # Split train/val
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create probe
            probe = LinearProbe(
                input_dim=self.model.hidden_size,
                hidden_dim=self.config.probe_hidden_dim
            ).to(self.device)
            
            # Train
            optimizer = torch.optim.Adam(probe.parameters(), lr=self.config.probe_lr)
            loss_fn = nn.BCELoss()
            
            best_val_acc = 0.0
            best_probe_state = None
            
            for epoch in range(self.config.probe_epochs):
                # Training
                probe.train()
                train_loss = 0.0
                
                for i in range(0, len(X_train), self.config.probe_batch_size):
                    batch_X = X_train[i:i+self.config.probe_batch_size].to(self.device)
                    batch_y = y_train[i:i+self.config.probe_batch_size].to(self.device)
                    
                    optimizer.zero_grad()
                    pred = probe(batch_X)
                    loss = loss_fn(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                probe.eval()
                with torch.no_grad():
                    val_pred = probe(X_val.to(self.device))
                    val_acc = ((val_pred > 0.5).float() == y_val.to(self.device)).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_probe_state = probe.state_dict().copy()
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.config.probe_epochs}, "
                          f"Train Loss: {train_loss/len(X_train):.4f}, "
                          f"Val Acc: {val_acc:.4f}")
            
            # Load best probe
            probe.load_state_dict(best_probe_state)
            probe.eval()
            
            # Store probe with loss function
            alpha = self.config.alphas[self.config.layers.index(layer_idx)]
            self.probes[layer_idx] = (probe, alpha, nn.BCELoss(reduction='none'))
            
            accuracies[layer_idx] = best_val_acc
            print(f"Layer {layer_idx} best accuracy: {best_val_acc:.4f}")
            
            # Save if path provided
            if save_path is not None:
                import os
                os.makedirs(save_path, exist_ok=True)
                torch.save(probe.state_dict(), f"{save_path}/probe_layer_{layer_idx}.pt")
        
        # Save metadata
        if save_path is not None:
            import json
            metadata = {
                "model_name": self.model.model_name,
                "layers": self.config.layers,
                "alphas": self.config.alphas,
                "accuracies": accuracies,
                "hidden_dim": self.config.probe_hidden_dim,
            }
            with open(f"{save_path}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        
        return accuracies
    
    def load_probes(self, load_path: str) -> None:
        """
        Load pre-trained probes from disk.
        
        Args:
            load_path: Directory containing probe weights and metadata
        """
        import json
        import os
        
        # Load metadata
        with open(f"{load_path}/metadata.json", "r") as f:
            metadata = json.load(f)
        
        print(f"Loading probes from {load_path}...")
        
        # Load each probe
        for layer_idx in self.config.layers:
            probe_path = f"{load_path}/probe_layer_{layer_idx}.pt"
            if not os.path.exists(probe_path):
                print(f"Warning: Probe for layer {layer_idx} not found")
                continue
            
            # Create probe
            probe = LinearProbe(
                input_dim=self.model.hidden_size,
                hidden_dim=metadata.get("hidden_dim")
            ).to(self.device)
            
            # Load weights
            probe.load_state_dict(torch.load(probe_path, map_location=self.device))
            probe.eval()
            
            # Freeze parameters
            for param in probe.parameters():
                param.requires_grad = False
            
            # Store probe
            alpha = self.config.alphas[self.config.layers.index(layer_idx)]
            self.probes[layer_idx] = (probe, alpha, nn.BCELoss(reduction='none'))
            
            print(f"Loaded probe for layer {layer_idx} (accuracy: {metadata['accuracies'].get(str(layer_idx), 'N/A')})")
    
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

