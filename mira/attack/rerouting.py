"""
Subspace rerouting attack implementation.

This attack leverages mechanistic understanding to redirect model activations
from the refusal subspace to the acceptance subspace, enabling jailbreaks
through internal representation manipulation rather than output targeting.
"""

from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F

from mira.attack.base import BaseAttack, AttackResult, AttackType
from mira.analysis.subspace import SubspaceAnalyzer, SubspaceResult


class ReroutingAttack(BaseAttack):
    """
    Subspace rerouting attack.
    
    This attack works by:
    1. Identifying refusal and acceptance subspaces using linear probes
    2. Optimizing an adversarial suffix to move activations toward acceptance
    3. Using gradient-based search to find optimal token replacements
    
    The key insight is that we're not optimizing for specific output text,
    but for internal representation shift, making the attack more robust.
    """
    
    def __init__(
        self,
        model_wrapper,
        subspace_result: Optional[SubspaceResult] = None,
        target_layer: Optional[int] = None,
        suffix_length: int = 20,
        batch_size: int = 512,
        top_k: int = 256,
        device: Optional[str] = None,
    ):
        """
        Initialize rerouting attack.
        
        Args:
            model_wrapper: ModelWrapper instance
            subspace_result: Pre-computed subspace analysis (optional)
            target_layer: Layer to optimize activations at
            suffix_length: Number of tokens in adversarial suffix
            batch_size: Batch size for candidate evaluation
            top_k: Top-k candidates to consider in gradient search
            device: Device for computation
        """
        super().__init__(
            model_wrapper,
            AttackType.SUBSPACE_REROUTING,
            suffix_length,
            batch_size,
            device,
        )
        
        self.subspace_result = subspace_result
        self.target_layer = target_layer or (model_wrapper.n_layers // 2)
        self.top_k = top_k
        
        # Will be computed if not provided
        self._acceptance_direction: Optional[torch.Tensor] = None
        self._refusal_direction: Optional[torch.Tensor] = None
    
    def set_subspaces(self, subspace_result: SubspaceResult) -> None:
        """Set subspace definitions for the attack."""
        self.subspace_result = subspace_result
        self._acceptance_direction = subspace_result.acceptance_direction.to(self.device)
        self._refusal_direction = subspace_result.refusal_direction.to(self.device)
    
    def compute_subspaces(
        self,
        safe_prompts: List[str],
        unsafe_prompts: List[str],
    ) -> SubspaceResult:
        """
        Compute subspaces from sample prompts.
        
        Args:
            safe_prompts: Examples of safe/benign prompts
            unsafe_prompts: Examples of unsafe/harmful prompts
            
        Returns:
            SubspaceResult with directions and bases
        """
        analyzer = SubspaceAnalyzer(
            self.model,
            layer_idx=self.target_layer,
        )
        
        # Use linear probe for better accuracy
        result = analyzer.train_probe(safe_prompts, unsafe_prompts, self.target_layer)
        self.set_subspaces(result)
        
        return result
    
    def get_activation_at_layer(
        self,
        prompt: str,
        suffix: str,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Get the activation at target layer for prompt + suffix."""
        full_input = prompt + " " + suffix
        _, cache = self.model.run_with_cache(full_input)
        
        layer = layer_idx or self.target_layer
        hidden = cache.hidden_states.get(layer)
        
        if hidden is None:
            raise ValueError(f"No activation found for layer {layer}")
        
        # Return last token activation
        return hidden[0, -1, :]
    
    def compute_rerouting_loss(
        self,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the rerouting loss.
        
        Loss is the projection onto the refusal direction (we want to minimize)
        minus the projection onto acceptance direction (we want to maximize).
        
        Args:
            activation: Current activation vector
            
        Returns:
            Scalar loss tensor
        """
        if self._refusal_direction is None or self._acceptance_direction is None:
            raise ValueError("Subspaces not set. Call set_subspaces() first.")
        
        # Projection onto refusal direction (positive = more refusal-like)
        refusal_proj = torch.dot(activation, self._refusal_direction)
        
        # Projection onto acceptance direction (positive = more acceptance-like)
        acceptance_proj = torch.dot(activation, self._acceptance_direction)
        
        # We want to maximize acceptance and minimize refusal
        # So loss = refusal_proj - acceptance_proj
        loss = refusal_proj - acceptance_proj
        
        return loss
    
    def compute_loss(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
        target: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute the attack loss.
        
        Args:
            prompt: Original prompt
            suffix_tokens: Current suffix token IDs
            target: Not used in rerouting attack
            
        Returns:
            Scalar loss tensor
        """
        suffix_text = self.decode_suffix(suffix_tokens)
        activation = self.get_activation_at_layer(prompt, suffix_text)
        return self.compute_rerouting_loss(activation)
    
    def compute_token_gradients(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradients with respect to suffix token choices.
        
        Uses the embedding gradient approach to estimate which tokens
        would reduce the loss if substituted.
        
        Args:
            prompt: Original prompt
            suffix_tokens: Current suffix token IDs
            
        Returns:
            Gradient tensor of shape (suffix_length, vocab_size)
        """
        vocab_size = self.model.vocab_size
        
        # Create one-hot representation
        one_hot = torch.zeros(
            len(suffix_tokens), vocab_size,
            device=self.device,
            dtype=torch.float32,
        )
        one_hot.scatter_(1, suffix_tokens.unsqueeze(1), 1)
        one_hot.requires_grad = True
        
        # Get embedding matrix
        embed_matrix = self.model.get_embedding_matrix()
        
        # Compute suffix embeddings
        suffix_embeds = torch.matmul(one_hot, embed_matrix)
        
        # Get prompt embeddings
        prompt_tokens = self.model.tokenize(prompt)
        prompt_embeds = self.model.embed_tokens(prompt_tokens["input_ids"])
        
        # Concatenate and run through model
        full_embeds = torch.cat([prompt_embeds, suffix_embeds.unsqueeze(0)], dim=1)
        
        # Forward pass to get hidden states
        # We need to hook into the model to get gradients through to the embedding
        activation = self._forward_with_embeds(full_embeds, self.target_layer)
        
        # Compute rerouting loss
        loss = self.compute_rerouting_loss(activation)
        
        # Backpropagate
        loss.backward()
        
        return one_hot.grad
    
    def _forward_with_embeds(
        self,
        input_embeds: torch.Tensor,
        target_layer: int,
    ) -> torch.Tensor:
        """Run forward pass with embeddings and extract layer activation."""
        # Simplified forward - in practice need to handle model-specific details
        with torch.enable_grad():
            outputs = self.model.model(inputs_embeds=input_embeds, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            if target_layer < len(hidden_states):
                return hidden_states[target_layer][0, -1, :]
            else:
                return hidden_states[-1][0, -1, :]
    
    def sample_token_candidates(
        self,
        gradients: torch.Tensor,
        current_tokens: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """
        Sample candidate tokens based on gradients.
        
        Uses the gradient to find tokens that would reduce the loss.
        
        Args:
            gradients: Token gradients (suffix_length, vocab_size)
            current_tokens: Current suffix tokens
            position: Position to sample candidates for
            
        Returns:
            Tensor of candidate token IDs
        """
        # Get gradient for this position
        pos_grad = gradients[position]
        
        # Negative gradient indicates loss reduction
        # Get top-k tokens with most negative gradient
        _, candidates = (-pos_grad).topk(self.top_k)
        
        return candidates
    
    def evaluate_candidates(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
        position: int,
        candidates: torch.Tensor,
    ) -> int:
        """
        Evaluate candidate tokens and return best one.
        
        Args:
            prompt: Original prompt
            suffix_tokens: Current suffix tokens
            position: Position being modified
            candidates: Candidate token IDs
            
        Returns:
            Best candidate token ID
        """
        best_loss = float("inf")
        best_token = suffix_tokens[position].item()
        
        # Try each candidate
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i+self.batch_size]
            
            for candidate in batch_candidates:
                # Create modified suffix
                modified = suffix_tokens.clone()
                modified[position] = candidate
                
                # Compute loss
                loss = self.compute_loss(prompt, modified)
                loss_val = float(loss)
                
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_token = candidate.item()
        
        return best_token
    
    def optimize_step(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform one optimization step.
        
        Uses gradient-guided token sampling and replacement.
        
        Args:
            prompt: Original prompt
            suffix_tokens: Current suffix token IDs
            
        Returns:
            Updated suffix token IDs
        """
        try:
            # Compute token gradients
            gradients = self.compute_token_gradients(prompt, suffix_tokens)
            
            # Choose random positions to modify
            n_positions = min(3, len(suffix_tokens))
            positions = torch.randperm(len(suffix_tokens))[:n_positions]
            
            new_suffix = suffix_tokens.clone()
            
            for pos in positions:
                pos = pos.item()
                
                # Get candidate tokens
                candidates = self.sample_token_candidates(gradients, suffix_tokens, pos)
                
                # Evaluate and select best
                best_token = self.evaluate_candidates(prompt, new_suffix, pos, candidates)
                new_suffix[pos] = best_token
            
            return new_suffix
            
        except Exception:
            # Fallback to random replacement if gradient computation fails
            new_suffix = suffix_tokens.clone()
            pos = torch.randint(0, len(suffix_tokens), (1,)).item()
            new_token = torch.randint(0, self.model.vocab_size, (1,)).item()
            new_suffix[pos] = new_token
            return new_suffix
    
    def optimize(
        self,
        prompt: str,
        num_steps: int = 100,
        target: Optional[str] = None,
        init_method: str = "exclamation",
        verbose: bool = False,
        **kwargs,
    ) -> AttackResult:
        """
        Run full rerouting attack optimization.
        
        Args:
            prompt: Original prompt to attack
            num_steps: Number of optimization steps
            target: Not used in rerouting attack
            init_method: Suffix initialization method
            verbose: Print progress
            
        Returns:
            AttackResult with attack outcome and subspace metrics
        """
        # Run base optimization
        result = super().optimize(prompt, num_steps, target, init_method, verbose, **kwargs)
        
        # Add subspace-specific metrics
        if self.subspace_result is not None:
            # Compute initial distance
            initial_act = self.get_activation_at_layer(prompt, "")
            initial_dist = float(self.compute_rerouting_loss(initial_act))
            
            # Compute final distance
            final_act = self.get_activation_at_layer(prompt, result.adversarial_suffix)
            final_dist = float(self.compute_rerouting_loss(final_act))
            
            result.initial_subspace_distance = initial_dist
            result.final_subspace_distance = final_dist
        
        return result
