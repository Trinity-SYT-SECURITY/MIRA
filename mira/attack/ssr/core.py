"""
Core Subspace Rerouting (SSR) algorithm.

Implements gradient-based optimization to craft adversarial prompts by
analyzing and manipulating model internal activations.
"""

import re
from typing import List, Optional, Callable, Dict, Any, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mira.core.model_wrapper import ModelWrapper
from mira.attack.ssr.config import SSRConfig


class SSRAttack(ABC):
    """
    Base class for Subspace Rerouting attacks.
    
    The SSR algorithm optimizes adversarial tokens by:
    1. Masking parts of the input with [MASK] tokens
    2. Computing gradients through a loss function (implemented by subclasses)
    3. Sampling replacement tokens from top-k gradients
    4. Maintaining a buffer of best candidates
    5. Adaptively reducing the number of replaced tokens as loss decreases
    """
    
    def __init__(
        self,
        model: ModelWrapper,
        config: SSRConfig,
        callback: Optional[Callable[[int, float, str], None]] = None,
    ):
        """
        Initialize SSR attack.
        
        Args:
            model: Model wrapper instance
            config: SSR configuration
            callback: Optional callback(iteration, loss, candidate_text) for progress tracking
        """
        self.model = model
        self.config = config
        self.callback = callback
        self.device = model.device
        
        # Adjust max_layer for negative indexing
        if self.config.max_layer < 0:
            self.config.max_layer += self.model.n_layers
        
        # Prompt components (set by init_prompt)
        self.full_tokens: Optional[torch.Tensor] = None  # [seq_len]
        self.full_embeds: Optional[torch.Tensor] = None  # [seq_len, d_model]
        self.mask_positions: Optional[torch.Tensor] = None  # [mask_len]
        
        # Optimization state
        self.candidate_ids: Optional[torch.Tensor] = None  # [buffer_size, mask_len]
        self.candidate_losses: Optional[torch.Tensor] = None  # [buffer_size]
        self.archive_ids: Optional[torch.Tensor] = None  # [archive_size, mask_len]
        self.archive_losses: Optional[torch.Tensor] = None  # [archive_size]
        
        self.initial_loss: float = 0.0
        self.n_replace: int = 1
        
        # Activation storage (used by subclasses)
        self.activation_cache: Dict[str, torch.Tensor] = {}
    
    def init_prompt(self, sentence: str, mask_str: str = "[MASK]") -> None:
        """
        Initialize the prompt with masked positions.
        
        Args:
            sentence: Input sentence with [MASK] tokens, e.g.,
                     "How to create a bomb? [MASK][MASK][MASK]"
            mask_str: Mask token string (default: "[MASK]")
        """
        # Split sentence by mask tokens
        parts = re.split(f"({re.escape(mask_str)})", sentence)
        
        fixed_tokens = []
        fixed_positions: List[int] = []
        mask_positions = []
        current_pos = 0
        
        for part in parts:
            if part == mask_str:
                mask_positions.append(current_pos)
                current_pos += 1
            elif len(part) > 0:
                # Tokenize the fixed part
                tokens = self.model.tokenizer.encode(
                    part,
                    add_special_tokens=False,
                    return_tensors="pt"
                )[0]
                fixed_tokens.append(tokens)
                fixed_positions.extend(range(current_pos, current_pos + len(tokens)))
                current_pos += len(tokens)
        
        # Create full token tensor with zeros at mask positions
        self.full_tokens = torch.zeros(current_pos, dtype=torch.long, device=self.device)
        if fixed_tokens:
            self.full_tokens[fixed_positions] = torch.cat(fixed_tokens).to(self.device)
        
        # Get embeddings
        embed_matrix = self.model.get_embedding_matrix()
        self.full_embeds = F.embedding(self.full_tokens, embed_matrix)
        
        # Store mask positions
        self.mask_positions = torch.tensor(mask_positions, dtype=torch.long, device=self.device)
        
        print(f"Initialized prompt with {len(mask_positions)} mask positions")
        print(f"Total sequence length: {current_pos}")
    
    def get_full_tokens(self, masked_tokens: torch.Tensor) -> torch.Tensor:
        """
        Get full token sequence by filling in masked positions.
        
        Args:
            masked_tokens: [batch_size, mask_len] tokens for masked positions
            
        Returns:
            [batch_size, seq_len] full token sequence
        """
        batch_size = masked_tokens.shape[0]
        full_tokens = self.full_tokens.clone().unsqueeze(0).repeat(batch_size, 1)
        full_tokens[:, self.mask_positions] = masked_tokens
        return full_tokens
    
    def buffer_init_random(self) -> None:
        """Initialize candidate buffer with random tokens."""
        # Sample random tokens for mask positions
        candidate_ids = torch.randint(
            0,
            self.model.vocab_size,
            (self.config.buffer_size, len(self.mask_positions)),
            dtype=torch.long,
            device=self.device
        )
        
        # Filter invalid tokens
        if self.config.filter_tokens:
            candidate_ids = self._filter_tokens(candidate_ids)
        
        # Compute losses for initial candidates
        candidate_losses = self._compute_candidate_losses(candidate_ids)
        
        # Initialize buffers
        self.candidate_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.candidate_losses = torch.empty(0, dtype=torch.float32, device=self.device)
        self.archive_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self.archive_losses = torch.empty(0, dtype=torch.float32, device=self.device)
        
        self._buffer_add(candidate_ids, candidate_losses, update_n_replace=False)
        
        self.initial_loss = self.candidate_losses[0].item()
        self.n_replace = len(self.mask_positions)
        
        print(f"Initial loss: {self.initial_loss:.4f}")
        print(f"Initial candidate: {self.model.tokenizer.decode(self.candidate_ids[0])}")
    
    def _buffer_add(
        self,
        new_ids: torch.Tensor,
        new_losses: torch.Tensor,
        update_n_replace: bool = True,
    ) -> None:
        """
        Add new candidates to buffer, keeping only the best ones.
        
        Args:
            new_ids: [search_width, mask_len] new candidate tokens
            new_losses: [search_width] corresponding losses
            update_n_replace: Whether to update n_replace if loss improved
        """
        best_loss_before = (
            self.candidate_losses[0].item()
            if len(self.candidate_losses) > 0
            else float('inf')
        )
        
        # Combine with existing candidates - ensure all on same device
        # Move new tensors to the same device as existing ones
        new_losses = new_losses.to(self.device)
        new_ids = new_ids.to(self.device)
        all_losses = torch.cat([self.candidate_losses, new_losses], dim=0)
        all_ids = torch.cat([self.candidate_ids, new_ids], dim=0)
        
        # Sort by loss
        sorted_indices = torch.argsort(all_losses)
        
        # Remove duplicate losses (keep first occurrence)
        unique_mask = torch.cat([
            torch.tensor([True], device=self.device),
            all_losses[sorted_indices[:-1]] != all_losses[sorted_indices[1:]]
        ])
        
        filtered_indices = sorted_indices[unique_mask][:self.config.buffer_size]
        
        self.candidate_losses = all_losses[filtered_indices]
        self.candidate_ids = all_ids[filtered_indices]
        
        # Check if loss improved
        if self.candidate_losses[0] < best_loss_before:
            best_text = self.model.tokenizer.decode(self.candidate_ids[0])
            print(f"\n[IMPROVED] Loss: {self.candidate_losses[0]:.4f}")
            print(f"Candidate: {best_text}")
            
            if update_n_replace:
                self._update_n_replace()
    
    def _buffer_jump(self) -> None:
        """
        Jump to a different candidate when stuck.
        
        Moves the best candidate(s) to archive and samples a new starting point.
        """
        # Sample jump index based on softmax of negative losses
        jump_probs = F.softmax(-self.candidate_losses, dim=0)
        jump_idx = torch.multinomial(jump_probs, 1).item()
        
        print(f"\n[JUMP] Patience exceeded. Jumping from rank 0 (loss={self.candidate_losses[0]:.4f}) "
              f"to rank {jump_idx} (loss={self.candidate_losses[jump_idx]:.4f})")
        
        # Archive the skipped candidates - keep on same device
        self.archive_ids = torch.cat([
            self.archive_ids,
            self.candidate_ids[:jump_idx]
        ], dim=0)
        self.archive_losses = torch.cat([
            self.archive_losses,
            self.candidate_losses[:jump_idx]
        ], dim=0)
        
        # Keep only candidates from jump_idx onwards
        self.candidate_ids = self.candidate_ids[jump_idx:]
        self.candidate_losses = self.candidate_losses[jump_idx:]
    
    def _update_n_replace(self) -> None:
        """
        Update the number of tokens to replace based on current loss.
        
        As loss decreases, we replace fewer tokens for fine-tuning.
        Formula: n_replace = (current_loss / initial_loss) ^ (1 / replace_coefficient)
        """
        loss_ratio = min(
            1.0,
            max(0.0, self.candidate_losses[0].item() / (self.initial_loss + 1e-5))
        )
        loss_ratio = loss_ratio ** (1.0 / self.config.replace_coefficient)
        
        new_n_replace = max(1, int(loss_ratio * len(self.mask_positions)))
        
        if new_n_replace != self.n_replace:
            print(f"[UPDATE] n_replace: {self.n_replace} -> {new_n_replace}")
            self.n_replace = new_n_replace
    
    def _sample_ids_from_grad(
        self,
        current_ids: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample new candidate tokens based on gradients.
        
        Args:
            current_ids: [mask_len] current tokens at mask positions
            grad: [mask_len, vocab_size] gradients for each position
            
        Returns:
            [search_width, mask_len] new candidate tokens
        """
        # Start with current ids repeated
        new_ids = current_ids.unsqueeze(0).repeat(self.config.search_width, 1)
        
        # Get top-k tokens with lowest gradient (steepest descent)
        topk_ids = torch.topk(-grad, k=self.config.search_topk, dim=1).indices  # [mask_len, topk]
        
        # For each candidate, randomly select n_replace positions to modify
        for i in range(self.config.search_width):
            # Randomly select positions to replace
            replace_positions = torch.randperm(len(self.mask_positions), device=self.device)[:self.n_replace]
            
            # For each position, sample from top-k
            for pos in replace_positions:
                sampled_token = topk_ids[pos, torch.randint(0, self.config.search_topk, (1,), device=self.device)]
                new_ids[i, pos] = sampled_token
        
        return new_ids
    
    def _filter_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Filter out tokens that don't re-encode correctly.
        
        Args:
            tokens: [batch_size, seq_len] token sequences
            
        Returns:
            [new_batch_size, seq_len] filtered sequences
        """
        valid_sequences = []
        
        for i in range(tokens.shape[0]):
            # Decode and re-encode
            text = self.model.tokenizer.decode(tokens[i], skip_special_tokens=False)
            re_encoded = self.model.tokenizer.encode(
                text,
                add_special_tokens=False,
                return_tensors="pt"
            )[0].to(self.device)
            
            # Check if re-encoding matches
            if len(re_encoded) == len(tokens[i]) and torch.equal(tokens[i], re_encoded):
                valid_sequences.append(tokens[i])
        
        if not valid_sequences:
            # Fallback: return original if all filtered out
            print("[WARNING] All tokens filtered out, returning original")
            return tokens[:1]
        
        return torch.stack(valid_sequences)
    
    def _compute_candidate_losses(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute losses for candidate tokens.
        
        Args:
            candidate_ids: [batch_size, mask_len] candidate tokens
            
        Returns:
            [batch_size] losses
        """
        # Get full embeddings
        embed_matrix = self.model.get_embedding_matrix()
        candidate_embeds = F.embedding(candidate_ids, embed_matrix)  # [batch, mask_len, d_model]
        
        # Create full embedding sequences
        batch_size = candidate_ids.shape[0]
        full_embeds = self.full_embeds.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, seq_len, d_model]
        full_embeds[:, self.mask_positions, :] = candidate_embeds
        
        # Forward pass with hooks (implemented by subclasses)
        with torch.no_grad():
            losses = self.compute_loss(full_embeds)
        
        return losses
    
    def _compute_gradients(self, current_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients for current candidate.
        
        Args:
            current_ids: [mask_len] current tokens at mask positions
            
        Returns:
            [mask_len, vocab_size] gradients
        """
        # Get embeddings matrix first to determine correct dtype
        embed_matrix = self.model.get_embedding_matrix()
        
        # Create one-hot encoding with same dtype as embed_matrix
        one_hot = F.one_hot(current_ids, num_classes=self.model.vocab_size)
        one_hot = one_hot.to(dtype=embed_matrix.dtype)  # Match embed_matrix dtype (float16 or float32)
        one_hot.requires_grad = True
        
        # Get embeddings from one-hot
        adv_embeds = torch.matmul(one_hot, embed_matrix)  # [mask_len, d_model]
        
        # Create full embedding sequence
        full_embeds = self.full_embeds.clone()
        full_embeds[self.mask_positions] = adv_embeds
        full_embeds = full_embeds.unsqueeze(0)  # [1, seq_len, d_model]
        
        # Forward pass with gradient
        loss = self.compute_loss(full_embeds)
        loss.backward()
        
        # Get gradients
        grad = one_hot.grad.detach()  # [mask_len, vocab_size]
        
        return grad
    
    @abstractmethod
    def compute_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for given embeddings.
        
        This method must be implemented by subclasses to define the
        optimization objective (e.g., probe-based, steering-based).
        
        Args:
            embeddings: [batch_size, seq_len, d_model] input embeddings
            
        Returns:
            [batch_size] losses
        """
        raise NotImplementedError
    
    def generate(self) -> Tuple[str, float]:
        """
        Run the SSR optimization loop.
        
        Returns:
            (best_adversarial_text, final_loss)
        """
        if self.full_tokens is None or self.mask_positions is None:
            raise ValueError("Must call init_prompt() before generate()")
        
        if self.candidate_ids is None:
            raise ValueError("Must call buffer_init_random() before generate()")
        
        last_improvement = 0
        
        for iteration in tqdm(range(self.config.max_iterations), desc="SSR Optimization"):
            # Get current best candidate
            current_ids = self.candidate_ids[0].clone()
            current_loss = self.candidate_losses[0].item()
            
            # Compute gradients
            grad = self._compute_gradients(current_ids)
            
            # Sample new candidates from gradients
            new_ids = self._sample_ids_from_grad(current_ids, grad)
            
            # Filter tokens if enabled
            if self.config.filter_tokens:
                new_ids = self._filter_tokens(new_ids)
            
            # Compute losses for new candidates
            new_losses = self._compute_candidate_losses(new_ids)
            
            # Add to buffer
            old_best_loss = self.candidate_losses[0].item()
            self._buffer_add(new_ids, new_losses)
            
            # Check if improved
            if self.candidate_losses[0] < old_best_loss:
                last_improvement = iteration
            
            # Callback for visualization
            if self.callback is not None:
                best_text = self.model.tokenizer.decode(self.candidate_ids[0])
                self.callback(iteration, self.candidate_losses[0].item(), best_text)
            
            # Check stopping criteria
            if self.candidate_losses[0] < self.config.early_stop_loss:
                print(f"\n[EARLY STOP] Loss below threshold: {self.candidate_losses[0]:.4f}")
                break
            
            # Jump if stuck
            if iteration - last_improvement > self.config.patience:
                self._buffer_jump()
                last_improvement = iteration
        
        # Return best result
        best_text = self.model.tokenizer.decode(self.candidate_ids[0])
        final_loss = self.candidate_losses[0].item()
        
        print(f"\n[FINAL] Loss: {final_loss:.4f}")
        print(f"Adversarial prompt: {best_text}")
        
        return best_text, final_loss
    
    def get_best_candidate(self) -> Tuple[str, float]:
        """Get the current best candidate."""
        if self.candidate_ids is None or len(self.candidate_ids) == 0:
            raise ValueError("No candidates available")
        
        best_text = self.model.tokenizer.decode(self.candidate_ids[0])
        best_loss = self.candidate_losses[0].item()
        
        return best_text, best_loss
    
    def get_all_candidates(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k candidates from buffer and archive."""
        if self.candidate_ids is None:
            return []
        
        # Combine buffer and archive
        all_ids = torch.cat([self.candidate_ids, self.archive_ids], dim=0)
        all_losses = torch.cat([self.candidate_losses, self.archive_losses], dim=0)
        
        # Sort and get top-k
        sorted_indices = torch.argsort(all_losses)[:top_k]
        
        results = []
        for idx in sorted_indices:
            text = self.model.tokenizer.decode(all_ids[idx])
            loss = all_losses[idx].item()
            results.append((text, loss))
        
        return results

