"""
GCG (Greedy Coordinate Gradient) Attack Implementation.

This attack optimizes adversarial suffixes by:
1. Computing gradients w.r.t. token embeddings via one-hot trick
2. Sampling candidate replacements from top-k gradient positions
3. Evaluating batch of candidates and selecting best
"""

import gc
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mira.attack.base import BaseAttack, AttackResult, AttackType


@dataclass
class GCGConfig:
    """Configuration for GCG attack."""
    suffix_length: int = 20
    batch_size: int = 256  # Number of candidate suffixes per step
    top_k: int = 256  # Top-k tokens to sample from
    num_steps: int = 500
    allow_non_ascii: bool = False
    target_weight: float = 1.0
    filter_candidates: bool = True


class GCGAttack(BaseAttack):
    """
    Greedy Coordinate Gradient attack for adversarial suffix optimization.
    
    Key insight: Instead of gradient descent on continuous embeddings,
    GCG uses gradients to identify promising discrete token replacements,
    then evaluates a batch of candidates to find the best one.
    """
    
    def __init__(
        self,
        model_wrapper,
        config: Optional[GCGConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize GCG attack.
        
        Args:
            model_wrapper: ModelWrapper instance
            config: GCG configuration
            device: Device for computation
        """
        self.config = config or GCGConfig()
        
        super().__init__(
            model_wrapper,
            AttackType.GRADIENT,
            self.config.suffix_length,
            self.config.batch_size,
            device,
        )
        
        # Cache embedding matrix
        self.embed_weights = self.model.get_embedding_matrix()
        
        # Find non-ASCII tokens to exclude
        self._not_allowed_tokens = self._get_non_ascii_tokens() if not self.config.allow_non_ascii else None
    
    def _get_non_ascii_tokens(self) -> torch.Tensor:
        """Get token IDs that don't decode to ASCII."""
        non_ascii = []
        for i in range(self.model.vocab_size):
            try:
                decoded = self.model.tokenizer.decode([i])
                if not decoded.isascii() or decoded.strip() == '':
                    non_ascii.append(i)
            except:
                non_ascii.append(i)
        return torch.tensor(non_ascii, device=self.device)
    
    def get_input_slices(
        self,
        prompt: str,
        suffix: str,
        target: str = "Sure, here is",
    ) -> Tuple[torch.Tensor, slice, slice, slice]:
        """
        Tokenize and compute slices for prompt, control (suffix), and target.
        
        Returns:
            input_ids: Full input token IDs
            control_slice: Slice for suffix tokens
            target_slice: Slice for target tokens  
            loss_slice: Slice for loss computation
        """
        # Tokenize components
        prompt_toks = self.model.tokenizer.encode(prompt, add_special_tokens=True)
        suffix_toks = self.model.tokenizer.encode(suffix, add_special_tokens=False)
        target_toks = self.model.tokenizer.encode(target, add_special_tokens=False)
        
        # Build full input
        full_input = prompt_toks + suffix_toks + target_toks
        input_ids = torch.tensor(full_input, device=self.device)
        
        # Compute slices
        prompt_len = len(prompt_toks)
        suffix_len = len(suffix_toks)
        target_len = len(target_toks)
        
        control_slice = slice(prompt_len, prompt_len + suffix_len)
        target_slice = slice(prompt_len + suffix_len, prompt_len + suffix_len + target_len)
        loss_slice = slice(prompt_len + suffix_len - 1, prompt_len + suffix_len + target_len - 1)
        
        return input_ids, control_slice, target_slice, loss_slice
    
    def token_gradients(
        self,
        input_ids: torch.Tensor,
        control_slice: slice,
        target_slice: slice,
        loss_slice: slice,
    ) -> torch.Tensor:
        """
        Compute gradients of loss w.r.t. token positions in control_slice.
        
        Uses one-hot encoding trick: gradients flow through one-hot × embedding matrix.
        
        Args:
            input_ids: Full input token IDs
            control_slice: Slice for suffix/control tokens
            target_slice: Slice for target tokens
            loss_slice: Slice for loss computation
            
        Returns:
            Gradient tensor of shape (control_length, vocab_size)
        """
        # Create one-hot encoding for control tokens
        control_tokens = input_ids[control_slice]
        one_hot = torch.zeros(
            control_tokens.shape[0],
            self.embed_weights.shape[0],
            device=self.device,
            dtype=self.embed_weights.dtype,
        )
        one_hot.scatter_(
            1,
            control_tokens.unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.device, dtype=self.embed_weights.dtype),
        )
        one_hot.requires_grad_(True)
        
        # Compute embeddings via one-hot @ embedding matrix
        control_embeds = (one_hot @ self.embed_weights).unsqueeze(0)
        
        # Get embeddings for rest of input
        all_embeds = self.model.embed_tokens(input_ids.unsqueeze(0)).detach()
        
        # Stitch together: [before_control, control_embeds, after_control]
        full_embeds = torch.cat([
            all_embeds[:, :control_slice.start, :],
            control_embeds,
            all_embeds[:, control_slice.stop:, :],
        ], dim=1)
        
        # Forward pass
        outputs = self.model.model(inputs_embeds=full_embeds)
        logits = outputs.logits
        
        # Compute cross-entropy loss for target prediction
        targets = input_ids[target_slice]
        loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
        
        # Backpropagate
        loss.backward()
        
        # Return normalized gradients
        grad = one_hot.grad.clone()
        grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-10)
        
        return grad
    
    def sample_candidates(
        self,
        control_tokens: torch.Tensor,
        grad: torch.Tensor,
        batch_size: int,
        top_k: int = 256,
    ) -> torch.Tensor:
        """
        Sample candidate control token sequences based on gradients.
        
        For each position, sample from top-k tokens with most negative gradient
        (indicating largest loss reduction).
        
        Args:
            control_tokens: Current control token IDs
            grad: Gradient tensor (control_length, vocab_size)
            batch_size: Number of candidates to generate
            top_k: Number of top candidates per position
            
        Returns:
            Candidate token IDs of shape (batch_size, control_length)
        """
        # Exclude non-ASCII tokens
        if self._not_allowed_tokens is not None:
            grad = grad.clone()
            grad[:, self._not_allowed_tokens] = float('inf')
        
        # Get top-k token indices with most negative gradient
        top_indices = (-grad).topk(top_k, dim=1).indices
        
        # Create batch of candidates by sampling replacements
        control_tokens = control_tokens.to(grad.device)
        original = control_tokens.repeat(batch_size, 1)
        
        # Choose which position to modify for each candidate
        positions = torch.arange(
            0, len(control_tokens),
            len(control_tokens) / batch_size,
            device=grad.device,
        ).long()
        
        # Sample random token from top-k for each position
        replacements = torch.gather(
            top_indices[positions],
            1,
            torch.randint(0, top_k, (batch_size, 1), device=grad.device),
        )
        
        # Apply replacements
        candidates = original.scatter_(1, positions.unsqueeze(-1), replacements)
        
        return candidates
    
    def evaluate_candidates(
        self,
        prompt: str,
        candidates: List[str],
        target: str = "Sure, here is",
    ) -> Tuple[int, float]:
        """
        Evaluate candidate suffixes and return best one.
        
        Args:
            prompt: Original prompt
            candidates: List of candidate suffix strings
            target: Target completion
            
        Returns:
            Index of best candidate and its loss
        """
        losses = []
        
        for candidate in candidates:
            try:
                # Get slices for this candidate
                input_ids, _, target_slice, loss_slice = self.get_input_slices(
                    prompt, candidate, target
                )
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model.model(input_ids.unsqueeze(0))
                    logits = outputs.logits
                
                # Compute loss
                targets = input_ids[target_slice]
                loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
                losses.append(float(loss))
            except Exception as e:
                losses.append(float('inf'))
        
        # Return best
        best_idx = np.argmin(losses)
        return best_idx, losses[best_idx]
    
    def optimize(
        self,
        prompt: str,
        num_steps: Optional[int] = None,
        target: str = "Sure, here is",
        init_method: str = "exclamation",
        verbose: bool = True,
        **kwargs,
    ) -> AttackResult:
        """
        Run GCG attack optimization.
        
        Args:
            prompt: Original prompt to attack
            num_steps: Number of optimization steps
            target: Target completion to optimize for
            init_method: Suffix initialization method
            verbose: Print progress
            
        Returns:
            AttackResult with best suffix and metrics
        """
        num_steps = num_steps or self.config.num_steps
        
        # Initialize suffix
        if init_method == "exclamation":
            suffix = "!" * self.config.suffix_length
        else:
            suffix = " " * self.config.suffix_length
        
        best_suffix = suffix
        best_loss = float('inf')
        loss_history = []
        
        iterator = tqdm(range(num_steps), desc="GCG Attack") if verbose else range(num_steps)
        
        for step in iterator:
            try:
                # Get current slices
                input_ids, control_slice, target_slice, loss_slice = self.get_input_slices(
                    prompt, suffix, target
                )
                control_tokens = input_ids[control_slice]
                
                # Compute gradients
                grad = self.token_gradients(input_ids, control_slice, target_slice, loss_slice)
                
                # Sample candidates
                candidate_tokens = self.sample_candidates(
                    control_tokens,
                    grad,
                    batch_size=self.config.batch_size,
                    top_k=self.config.top_k,
                )
                
                # Decode candidates
                candidates = []
                for i in range(candidate_tokens.shape[0]):
                    decoded = self.model.tokenizer.decode(candidate_tokens[i], skip_special_tokens=True)
                    candidates.append(decoded)
                
                # Filter candidates
                if self.config.filter_candidates:
                    candidates = [c for c in candidates if c != suffix and len(c.strip()) > 0]
                    if len(candidates) == 0:
                        candidates = [suffix]
                
                # Evaluate and select best
                best_idx, loss = self.evaluate_candidates(prompt, candidates, target)
                
                if loss < best_loss:
                    best_loss = loss
                    best_suffix = candidates[best_idx]
                    suffix = best_suffix
                
                loss_history.append(loss)
                
                if verbose and isinstance(iterator, tqdm):
                    iterator.set_postfix({"loss": f"{loss:.4f}", "best": f"{best_loss:.4f}"})
                
                # Clear memory
                del grad, candidate_tokens
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if verbose:
                    print(f"Step {step} error: {e}")
                continue
        
        # Generate response with best suffix
        full_prompt = prompt + " " + best_suffix
        try:
            response = self.model.generate(full_prompt, max_new_tokens=100)[0]
            response = response[len(full_prompt):].strip()
        except:
            response = ""
        
        # Check success
        success = target.lower() in response.lower() if response else False
        
        # Collect internal state data for research analysis
        layer_data = None
        attention_data = None
        subspace_metrics = None
        
        try:
            # Tokenize final prompt
            final_input_ids = self.model.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            
            # Run with cache to collect internal states
            with torch.no_grad():
                outputs = self.model.model(
                    final_input_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                
                # Collect layer-wise activation norms
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    layer_norms = {}
                    for i, hidden in enumerate(outputs.hidden_states):
                        if hidden is not None:
                            layer_norms[i] = float(hidden.norm().item())
                    layer_data = {"activation_norms": layer_norms}
                
                # Collect attention statistics
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    num_heads = outputs.attentions[0].shape[1] if len(outputs.attentions) > 0 else 0
                    # Compute mean attention entropy across layers
                    attention_entropies = []
                    for attn in outputs.attentions:
                        if attn is not None:
                            # Compute entropy: -sum(p * log(p))
                            attn_probs = attn.mean(dim=1)  # Average over heads
                            entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum(dim=-1).mean()
                            attention_entropies.append(float(entropy.item()))
                    
                    attention_data = {
                        "num_heads": num_heads,
                        "mean_entropy": float(np.mean(attention_entropies)) if attention_entropies else 0.0,
                        "entropy_by_layer": attention_entropies,
                    }
                
                # Collect subspace metrics if available
                subspace_metrics = {
                    "final_loss": best_loss,
                    "loss_reduction": loss_history[0] - best_loss if loss_history else 0.0,
                }
        except Exception as e:
            # If data collection fails, continue without it
            if verbose:
                print(f"Warning: Could not collect internal state data: {e}")
        
        return AttackResult(
            success=success,
            adversarial_suffix=best_suffix,
            final_loss=best_loss,
            generated_response=response,
            num_steps=num_steps,
            loss_history=loss_history,
            layer_activations=layer_data,
            attention_patterns=attention_data,
            subspace_metrics=subspace_metrics,
        )
    
    def optimize_step(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Single optimization step (for compatibility with base class)."""
        # GCG uses its own optimization loop
        target = kwargs.get("target", "Sure, here is")
        suffix = self.decode_suffix(suffix_tokens)
        
        input_ids, control_slice, target_slice, loss_slice = self.get_input_slices(
            prompt, suffix, target
        )
        
        grad = self.token_gradients(input_ids, control_slice, target_slice, loss_slice)
        candidates = self.sample_candidates(suffix_tokens, grad, batch_size=1, top_k=self.config.top_k)
        
        return candidates[0]
