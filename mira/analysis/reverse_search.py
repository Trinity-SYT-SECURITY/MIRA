"""
Reverse activation search for finding inputs that produce target activations.

Enables discovering which inputs trigger specific internal states,
useful for understanding defense mechanisms and generating targeted attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    """Result of reverse activation search."""
    target_type: str  # "direction", "pattern", "activation"
    best_tokens: List[int]
    best_text: str
    final_similarity: float
    optimization_steps: int
    history: List[float]


class ReverseActivationSearch:
    """
    Finds inputs that produce target activation patterns.
    
    Given a target activation (e.g., "refusal direction"),
    searches for input tokens that maximize similarity to target.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        target_layer: int = -1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Get target layer index
        self._detect_layers()
        if target_layer < 0:
            target_layer = self.n_layers + target_layer
        self.target_layer = target_layer
        
        # Get embedding matrix
        self.embed_matrix = self._get_embedding_matrix()
    
    def _detect_layers(self):
        """Detect model layers."""
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'h'):
                self._layers = list(self.model.transformer.h)
            else:
                self._layers = []
        elif hasattr(self.model, 'gpt_neox'):
            self._layers = list(self.model.gpt_neox.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self._layers = list(self.model.model.layers)
        else:
            self._layers = []
        
        self.n_layers = len(self._layers)
    
    def _get_embedding_matrix(self) -> torch.Tensor:
        """Get input embedding matrix."""
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings().weight
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte.weight
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens.weight
        raise ValueError("Could not find embedding matrix")
    
    def _get_layer_output(
        self,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Get output of specific layer."""
        activations = {}
        
        def hook_fn(module, input_tensor, output):
            if isinstance(output, tuple):
                activations['output'] = output[0]
            else:
                activations['output'] = output
        
        handle = self._layers[layer_idx].register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                self.model(input_ids)
            return activations['output']
        finally:
            handle.remove()
    
    def search_for_direction(
        self,
        direction: torch.Tensor,
        num_tokens: int = 10,
        num_steps: int = 500,
        lr: float = 0.1,
        maximize: bool = True,
        verbose: bool = True
    ) -> SearchResult:
        """
        Search for tokens that maximize/minimize projection onto direction.
        
        Args:
            direction: Target direction vector [hidden_dim]
            num_tokens: Number of tokens to optimize
            num_steps: Optimization steps
            lr: Learning rate
            maximize: True to maximize, False to minimize projection
            verbose: Print progress
            
        Returns:
            SearchResult with best tokens found
        """
        direction = direction.to(self.device)
        direction_norm = direction / torch.norm(direction)
        
        # Initialize with random embeddings
        embed_dim = self.embed_matrix.shape[1]
        soft_tokens = nn.Parameter(
            torch.randn(1, num_tokens, embed_dim, device=self.device) * 0.1
        )
        
        optimizer = torch.optim.Adam([soft_tokens], lr=lr)
        history = []
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass with soft embeddings
            hidden = self._forward_with_embeddings(soft_tokens)
            
            # Compute projection onto direction
            # Use mean of sequence positions
            mean_hidden = hidden.mean(dim=1)  # [1, hidden_dim]
            projection = torch.sum(mean_hidden * direction_norm, dim=-1)
            
            # Loss (negate if maximizing)
            if maximize:
                loss = -projection
            else:
                loss = projection
            
            loss.backward()
            optimizer.step()
            
            history.append(projection.item())
            
            if verbose and (step + 1) % 100 == 0:
                print(f"  Step {step+1}: projection = {projection.item():.4f}")
        
        # Find nearest tokens for each soft embedding
        best_tokens = self._soft_to_tokens(soft_tokens)
        best_text = self.tokenizer.decode(best_tokens)
        
        return SearchResult(
            target_type="direction",
            best_tokens=best_tokens,
            best_text=best_text,
            final_similarity=history[-1] if history else 0.0,
            optimization_steps=num_steps,
            history=history,
        )
    
    def _forward_with_embeddings(
        self,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass using soft embeddings instead of token IDs."""
        # This is architecture-specific
        # For now, use the target layer hook approach
        
        # Create dummy input IDs (will be replaced by embeddings)
        batch_size, seq_len = embeddings.shape[:2]
        dummy_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        
        # Hook to replace input embeddings
        replaced = {}
        
        def embed_hook(module, input_tensor, output):
            replaced['original'] = output
            return embeddings
        
        # Register hooks
        embed_layer = self.model.get_input_embeddings()
        embed_handle = embed_layer.register_forward_hook(embed_hook)
        
        layer_output = {}
        
        def layer_hook(module, input_tensor, output):
            if isinstance(output, tuple):
                layer_output['hidden'] = output[0]
            else:
                layer_output['hidden'] = output
        
        layer_handle = self._layers[self.target_layer].register_forward_hook(layer_hook)
        
        try:
            self.model(dummy_ids)
            return layer_output['hidden']
        finally:
            embed_handle.remove()
            layer_handle.remove()
    
    def _soft_to_tokens(self, soft_embeddings: torch.Tensor) -> List[int]:
        """Convert soft embeddings to nearest token IDs."""
        tokens = []
        
        for pos in range(soft_embeddings.shape[1]):
            soft_embed = soft_embeddings[0, pos, :]  # [embed_dim]
            
            # Compute similarity to all tokens
            similarities = F.cosine_similarity(
                soft_embed.unsqueeze(0),
                self.embed_matrix,
                dim=1
            )
            
            # Get most similar token
            best_token = torch.argmax(similarities).item()
            tokens.append(best_token)
        
        return tokens
    
    def find_refusal_triggers(
        self,
        refusal_direction: torch.Tensor,
        num_tokens: int = 10,
        num_steps: int = 300
    ) -> SearchResult:
        """
        Find inputs that maximally activate refusal direction.
        
        Args:
            refusal_direction: The refusal direction vector
            num_tokens: Length of trigger sequence
            num_steps: Optimization steps
            
        Returns:
            SearchResult with tokens that trigger refusal
        """
        return self.search_for_direction(
            direction=refusal_direction,
            num_tokens=num_tokens,
            num_steps=num_steps,
            maximize=True,
            verbose=True
        )
    
    def find_acceptance_triggers(
        self,
        refusal_direction: torch.Tensor,
        num_tokens: int = 10,
        num_steps: int = 300
    ) -> SearchResult:
        """
        Find inputs that minimally activate refusal direction.
        
        Args:
            refusal_direction: The refusal direction vector
            num_tokens: Length of trigger sequence
            num_steps: Optimization steps
            
        Returns:
            SearchResult with tokens that avoid refusal
        """
        return self.search_for_direction(
            direction=refusal_direction,
            num_tokens=num_tokens,
            num_steps=num_steps,
            maximize=False,
            verbose=True
        )


class SSROptimizer:
    """
    Subspace Steering Routing Optimizer.
    
    Optimizes adversarial suffixes to steer model activations
    away from refusal subspace toward acceptance subspace.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        refusal_direction: torch.Tensor,
        target_layer: int = -1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        self.refusal_dir = refusal_direction.to(self.device)
        self.refusal_dir_norm = self.refusal_dir / torch.norm(self.refusal_dir)
        
        # Detect layers
        self._detect_layers()
        if target_layer < 0:
            target_layer = self.n_layers + target_layer
        self.target_layer = target_layer
        
        # Get embeddings
        self.embed_matrix = self._get_embedding_matrix()
    
    def _detect_layers(self):
        """Detect model layers."""
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'h'):
                self._layers = list(self.model.transformer.h)
            else:
                self._layers = []
        elif hasattr(self.model, 'gpt_neox'):
            self._layers = list(self.model.gpt_neox.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self._layers = list(self.model.model.layers)
        else:
            self._layers = []
        
        self.n_layers = len(self._layers)
    
    def _get_embedding_matrix(self) -> torch.Tensor:
        """Get input embedding matrix."""
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings().weight
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte.weight
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens.weight
        raise ValueError("Could not find embedding matrix")
    
    def compute_ssr_loss(
        self,
        input_ids: torch.Tensor,
        suffix_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute SSR loss: projection onto refusal direction.
        
        Lower is better (further from refusal).
        
        Args:
            input_ids: Input token IDs
            suffix_mask: Boolean mask for suffix positions
            
        Returns:
            (loss tensor for backprop, loss value)
        """
        # Get layer activations
        activations = []
        
        def hook_fn(module, input_tensor, output):
            if isinstance(output, tuple):
                activations.append(output[0])
            else:
                activations.append(output)
        
        handle = self._layers[self.target_layer].register_forward_hook(hook_fn)
        
        try:
            outputs = self.model(input_ids)
            hidden = activations[0]
        finally:
            handle.remove()
        
        # Get mean activation over suffix positions
        suffix_hidden = hidden[0, suffix_mask, :]
        mean_hidden = suffix_hidden.mean(dim=0)
        
        # Compute cosine similarity with refusal direction
        similarity = F.cosine_similarity(
            mean_hidden.unsqueeze(0),
            self.refusal_dir_norm.unsqueeze(0),
            dim=1
        )
        
        return similarity, similarity.item()
    
    def optimize_suffix(
        self,
        prompt: str,
        suffix_length: int = 10,
        num_steps: int = 200,
        batch_size: int = 16,
        top_k: int = 128,
        verbose: bool = True
    ) -> Tuple[str, List[float]]:
        """
        Optimize adversarial suffix using SSR loss.
        
        Args:
            prompt: Base prompt
            suffix_length: Number of suffix tokens
            num_steps: Optimization steps
            batch_size: Candidates per step
            top_k: Top-k token selection
            verbose: Print progress
            
        Returns:
            (optimized suffix, loss history)
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids.shape[1]
        
        # Initialize random suffix
        suffix_ids = torch.randint(
            0, self.tokenizer.vocab_size, 
            (1, suffix_length), 
            device=self.device
        )
        
        # Create full sequence and mask
        input_ids = torch.cat([prompt_ids, suffix_ids], dim=1)
        suffix_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=self.device)
        suffix_mask[prompt_len:] = True
        
        # Optimization loop
        history = []
        best_loss = float('inf')
        best_suffix = suffix_ids.clone()
        
        for step in range(num_steps):
            # Compute current loss
            with torch.no_grad():
                _, loss_val = self.compute_ssr_loss(input_ids, suffix_mask)
            history.append(loss_val)
            
            if loss_val < best_loss:
                best_loss = loss_val
                best_suffix = input_ids[0, prompt_len:].clone()
            
            # Generate candidates by flipping one token
            candidates = []
            for pos in range(suffix_length):
                for _ in range(batch_size // suffix_length):
                    new_ids = input_ids.clone()
                    # Random token from embedding space
                    new_token = torch.randint(
                        0, min(top_k, self.tokenizer.vocab_size),
                        (1,), device=self.device
                    )
                    new_ids[0, prompt_len + pos] = new_token
                    candidates.append(new_ids)
            
            # Evaluate candidates
            best_candidate = input_ids
            best_cand_loss = loss_val
            
            for cand in candidates:
                with torch.no_grad():
                    _, cand_loss = self.compute_ssr_loss(cand, suffix_mask)
                
                if cand_loss < best_cand_loss:
                    best_cand_loss = cand_loss
                    best_candidate = cand
            
            input_ids = best_candidate
            
            if verbose and (step + 1) % 20 == 0:
                current_suffix = self.tokenizer.decode(input_ids[0, prompt_len:])
                print(f"  Step {step+1}: loss = {best_cand_loss:.4f} | suffix: {current_suffix[:30]}...")
        
        # Return best suffix found
        best_suffix_text = self.tokenizer.decode(best_suffix)
        return best_suffix_text, history


def extract_refusal_direction(
    model: nn.Module,
    tokenizer: Any,
    safe_prompts: List[str],
    harmful_prompts: List[str],
    layer_idx: int = -1
) -> torch.Tensor:
    """
    Extract refusal direction from contrastive prompts.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        safe_prompts: List of safe/benign prompts
        harmful_prompts: List of harmful prompts
        layer_idx: Which layer to extract direction from
        
    Returns:
        Refusal direction vector [hidden_dim]
    """
    device = next(model.parameters()).device
    
    # Get layer for extraction
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = list(model.transformer.h)
    elif hasattr(model, 'gpt_neox'):
        layers = list(model.gpt_neox.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = list(model.model.layers)
    else:
        raise ValueError("Could not detect model layers")
    
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    
    # Collect activations
    safe_activations = []
    harmful_activations = []
    
    def hook_fn(activation_list):
        def fn(module, input_tensor, output):
            if isinstance(output, tuple):
                activation_list.append(output[0].detach())
            else:
                activation_list.append(output.detach())
        return fn
    
    # Process safe prompts
    handle = layers[layer_idx].register_forward_hook(hook_fn(safe_activations))
    for prompt in safe_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(input_ids)
    handle.remove()
    
    # Process harmful prompts
    handle = layers[layer_idx].register_forward_hook(hook_fn(harmful_activations))
    for prompt in harmful_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(input_ids)
    handle.remove()
    
    # Compute mean activations
    safe_mean = torch.cat([a[:, -1, :] for a in safe_activations], dim=0).mean(dim=0)
    harmful_mean = torch.cat([a[:, -1, :] for a in harmful_activations], dim=0).mean(dim=0)
    
    # Refusal direction: safe - harmful
    refusal_direction = safe_mean - harmful_mean
    
    return refusal_direction
