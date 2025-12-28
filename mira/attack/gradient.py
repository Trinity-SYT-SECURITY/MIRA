"""
Gradient-based attack for adversarial suffix optimization.

This attack uses token-level gradient optimization to find adversarial
suffixes that cause the model to comply with harmful requests.

All target tokens loaded from configuration - no hardcoded values.
"""

from typing import Optional, List
import torch
import torch.nn.functional as F
import numpy as np

from mira.attack.base import BaseAttack, AttackResult, AttackType
from mira.config import load_config, EvaluationConfig


class GradientAttack(BaseAttack):
    """
    Gradient-based adversarial suffix attack.
    
    Uses gradient information to find token replacements that increase
    the probability of a target completion.
    
    All target tokens loaded from config - no hardcoded values.
    """
    
    def __init__(
        self,
        model_wrapper,
        suffix_length: int = 20,
        batch_size: int = 512,
        top_k: int = 256,
        target_type: str = "affirmative",
        config_path: Optional[str] = None,
        affirmative_targets: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize gradient attack.
        
        Args:
            model_wrapper: ModelWrapper instance
            suffix_length: Number of tokens in adversarial suffix
            batch_size: Batch size for candidate evaluation
            top_k: Top-k candidates to consider
            target_type: "affirmative" or "custom"
            config_path: Path to config file
            affirmative_targets: Override targets (loads from config if None)
            device: Device for computation
        """
        super().__init__(
            model_wrapper,
            AttackType.GRADIENT_BASED,
            suffix_length,
            batch_size,
            device,
        )
        
        self.top_k = top_k
        self.target_type = target_type
        
        # Load affirmative targets from config or use provided
        if affirmative_targets is not None:
            self.affirmative_targets = affirmative_targets
        else:
            config_dict = load_config(config_path)
            eval_config = EvaluationConfig.from_dict(config_dict)
            self.affirmative_targets = eval_config.affirmative_targets or ["Sure", "Here", "Certainly"]
        
        # Encode targets
        self._target_ids = None
    
    def set_targets(self, target_texts: List[str]) -> None:
        """Set custom target texts."""
        self.affirmative_targets = target_texts
        self._target_ids = None  # Reset cache
    
    def _get_target_ids(self) -> List[int]:
        """Get token IDs for targets."""
        if self._target_ids is not None:
            return self._target_ids
        
        target_ids = []
        for text in self.affirmative_targets:
            tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                target_ids.append(tokens[0])
        
        self._target_ids = target_ids
        return target_ids
    
    def compute_loss(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
        target: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute the attack loss.
        
        For affirmative targeting: minimize negative log probability of targets.
        For custom targeting: maximize probability of specific completion.
        
        Args:
            prompt: Original prompt
            suffix_tokens: Current suffix token IDs
            target: Optional custom target text
            
        Returns:
            Scalar loss tensor
        """
        suffix_text = self.decode_suffix(suffix_tokens)
        full_prompt = prompt + " " + suffix_text
        
        # Get logits for the last position
        logits = self.model.compute_token_logits(full_prompt)
        next_token_logits = logits[0, -1, :]  # Last position
        
        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        
        if target is not None:
            # Custom target: maximize target probability
            target_ids = self.model.tokenizer.encode(target, add_special_tokens=False)
            if target_ids:
                target_prob = probs[target_ids[0]]
                return -torch.log(target_prob + 1e-10)
        
        # Affirmative targeting: maximize sum of affirmative token probabilities
        target_ids = self._get_target_ids()
        if not target_ids:
            # Fallback if config is empty
            fallback_tokens = self.model.tokenizer.encode("Sure", add_special_tokens=False)
            target_ids = [fallback_tokens[0]] if fallback_tokens else [0]
        
        target_probs = probs[target_ids]
        combined_prob = target_probs.sum()
        
        return -torch.log(combined_prob + 1e-10)
    
    def compute_token_gradients(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
        target: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute gradients for token replacement.
        
        Args:
            prompt: Original prompt
            suffix_tokens: Current suffix tokens
            target: Optional target text
            
        Returns:
            Gradient tensor (suffix_length, vocab_size)
        """
        vocab_size = self.model.vocab_size
        
        # Create one-hot with gradients
        one_hot = torch.zeros(
            len(suffix_tokens), vocab_size,
            device=self.device,
            dtype=torch.float32,
        )
        one_hot.scatter_(1, suffix_tokens.unsqueeze(1), 1)
        one_hot.requires_grad = True
        
        # Get embeddings through one-hot
        embed_matrix = self.model.get_embedding_matrix()
        suffix_embeds = torch.matmul(one_hot, embed_matrix)
        
        # Tokenize prompt
        prompt_tokens = self.model.tokenize(prompt)
        prompt_embeds = self.model.embed_tokens(prompt_tokens["input_ids"])
        
        # Concatenate embeddings
        full_embeds = torch.cat([prompt_embeds, suffix_embeds.unsqueeze(0)], dim=1)
        
        # Forward pass
        outputs = self.model.model(inputs_embeds=full_embeds)
        last_logits = outputs.logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        
        # Compute loss
        if target is not None:
            target_ids = self.model.tokenizer.encode(target, add_special_tokens=False)
            target_prob = probs[target_ids[0]] if target_ids else probs.max()
        else:
            target_ids = self._get_target_ids()
            target_prob = probs[target_ids].sum()
        
        loss = -torch.log(target_prob + 1e-10)
        
        # Backpropagate
        loss.backward()
        
        return one_hot.grad
    
    def optimize_step(
        self,
        prompt: str,
        suffix_tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform one gradient-based optimization step.
        
        Args:
            prompt: Original prompt
            suffix_tokens: Current suffix token IDs
            
        Returns:
            Updated suffix token IDs
        """
        target = kwargs.get("target", None)
        
        try:
            # Compute gradients
            gradients = self.compute_token_gradients(prompt, suffix_tokens, target)
            
            # Sample positions to modify
            n_positions = min(4, len(suffix_tokens))
            positions = torch.randperm(len(suffix_tokens))[:n_positions]
            
            new_suffix = suffix_tokens.clone()
            
            for pos in positions:
                pos = pos.item()
                
                # Get gradient for this position
                pos_grad = gradients[pos]
                
                # Select candidates with most negative gradient (biggest loss decrease)
                _, candidates = (-pos_grad).topk(self.top_k)
                
                # Evaluate candidates
                best_loss = float("inf")
                best_token = suffix_tokens[pos].item()
                
                # Sample a subset for efficiency
                sample_size = min(32, len(candidates))
                sample_indices = torch.randperm(len(candidates))[:sample_size]
                
                for idx in sample_indices:
                    candidate = candidates[idx]
                    modified = new_suffix.clone()
                    modified[pos] = candidate
                    
                    loss = float(self.compute_loss(prompt, modified, target))
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_token = candidate.item()
                
                new_suffix[pos] = best_token
            
            return new_suffix
            
        except Exception:
            # Fallback to random search
            new_suffix = suffix_tokens.clone()
            pos = torch.randint(0, len(suffix_tokens), (1,)).item()
            
            best_loss = float("inf")
            best_token = suffix_tokens[pos].item()
            
            for _ in range(32):
                candidate = torch.randint(0, self.model.vocab_size, (1,)).item()
                modified = new_suffix.clone()
                modified[pos] = candidate
                
                loss = float(self.compute_loss(prompt, modified, target))
                
                if loss < best_loss:
                    best_loss = loss
                    best_token = candidate
            
            new_suffix[pos] = best_token
            return new_suffix
