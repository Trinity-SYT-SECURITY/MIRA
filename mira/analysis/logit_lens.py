"""
Logit Lens Analysis for MIRA.

Projects intermediate layer hidden states to vocabulary space to track
how output predictions form across layers. Essential for understanding
when and where the model decides to output harmful vs safe content.

Reference: "Interpreting GPT: the logit lens" by nostalgebraist
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from mira.core.model_wrapper import ModelWrapper


@dataclass
class LogitLensResult:
    """Result from Logit Lens analysis."""
    prompt: str
    layer_logits: Dict[int, torch.Tensor]  # {layer_idx: [vocab_size] logits}
    layer_probs: Dict[int, torch.Tensor]   # {layer_idx: [vocab_size] probabilities}
    top_tokens: Dict[int, List[Tuple[str, float]]]  # {layer_idx: [(token, prob), ...]}
    target_token_probs: Optional[Dict[str, Dict[int, float]]] = None  # {token: {layer: prob}}


class LogitLens:
    """
    Logit Lens Analysis Tool.
    
    Projects hidden states from each layer to vocabulary space using the
    model's unembedding matrix (lm_head). This reveals how predictions
    form across layers.
    
    Key insight: If a layer already has high probability for a harmful token,
    the attack's influence at that layer is significant.
    """
    
    def __init__(self, model: ModelWrapper):
        """
        Initialize Logit Lens.
        
        Args:
            model: ModelWrapper instance with loaded model
        """
        self.model = model
        self.device = model.device
        self.n_layers = model.n_layers
        self.vocab_size = model.vocab_size
        
        # Get unembedding matrix (lm_head weight)
        self.unembed = self._get_unembed_matrix()
        
    def _get_unembed_matrix(self) -> torch.Tensor:
        """Get the unembedding matrix for logit projection."""
        # Most models have lm_head as the unembedding layer
        if hasattr(self.model.model, 'lm_head'):
            return self.model.model.lm_head.weight.detach()  # [vocab_size, hidden_size]
        elif hasattr(self.model.model, 'embed_out'):
            return self.model.model.embed_out.weight.detach()
        else:
            raise ValueError("Could not find unembedding matrix (lm_head)")
    
    def _validate_tokens(self, prompt: str) -> bool:
        """
        Validate that tokenized prompt has all valid token IDs.
        Prevents CUDA kernel asserts from invalid token indices.
        """
        try:
            inputs = self.model.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # Check for out-of-range tokens
            if input_ids.max() >= self.vocab_size or input_ids.min() < 0:
                print(f"  ⚠ Invalid token IDs (range: {input_ids.min()}-{input_ids.max()}, vocab: {self.vocab_size})")
                return False
            return True
        except Exception as e:
            print(f"  ⚠ Token validation failed: {e}")
            return False

    
    def analyze(
        self,
        prompt: str,
        target_tokens: Optional[List[str]] = None,
        position: int = -1,  # -1 = last token position
    ) -> LogitLensResult:
        """
        Run Logit Lens analysis on a prompt.
        
        Args:
            prompt: Input prompt to analyze
            target_tokens: Optional list of tokens to track across layers
            position: Token position to analyze (-1 = last)
            
        Returns:
            LogitLensResult with layer-wise logits and probabilities, or None if validation fails
        """
        # Validate tokens before processing to prevent CUDA kernel asserts
        if not self._validate_tokens(prompt):
            print(f"  ⚠ Skipping Logit Lens due to invalid tokens")
            return None
        
        # Get hidden states from all layers
        try:
            hidden_states = self._get_hidden_states(prompt)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"  ⚠ CUDA error in Logit Lens, returning None")
                return None
            raise

        
        layer_logits = {}
        layer_probs = {}
        top_tokens = {}
        
        for layer_idx, hidden in hidden_states.items():
            # Get hidden state at target position
            h = hidden[0, position, :]  # [hidden_size]
            
            # Apply layer norm if model has final layer norm
            h = self._apply_final_norm(h)
            
            # Project to vocabulary space: logits = h @ W_U^T
            logits = torch.matmul(h.float(), self.unembed.T.float())  # [vocab_size]
            probs = F.softmax(logits, dim=-1)
            
            layer_logits[layer_idx] = logits.cpu()
            layer_probs[layer_idx] = probs.cpu()
            
            # Get top-k tokens
            top_k = 10
            top_probs, top_indices = torch.topk(probs, top_k)
            top_tokens[layer_idx] = [
                (self.model.tokenizer.decode([idx.item()]), p.item())
                for idx, p in zip(top_indices, top_probs)
            ]
        
        # Track specific target tokens if provided
        target_token_probs = None
        if target_tokens:
            target_token_probs = self._track_target_tokens(
                target_tokens, layer_probs
            )
        
        return LogitLensResult(
            prompt=prompt,
            layer_logits=layer_logits,
            layer_probs=layer_probs,
            top_tokens=top_tokens,
            target_token_probs=target_token_probs,
        )
    
    def _get_hidden_states(self, prompt: str) -> Dict[int, torch.Tensor]:
        """Extract hidden states from all layers."""
        inputs = self.model.tokenize(prompt)
        
        with torch.no_grad():
            outputs = self.model.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        hidden_states = {}
        for i, h in enumerate(outputs.hidden_states):
            hidden_states[i] = h.detach()
        
        return hidden_states
    
    def _apply_final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply final layer normalization if present."""
        model = self.model.model
        
        # Try different normalization layer names
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            return model.transformer.ln_f(hidden)
        elif hasattr(model, 'model') and hasattr(model.model, 'norm'):
            return model.model.norm(hidden)
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'final_layer_norm'):
            return model.gpt_neox.final_layer_norm(hidden)
        else:
            return hidden  # No final norm found
    
    def _track_target_tokens(
        self,
        tokens: List[str],
        layer_probs: Dict[int, torch.Tensor]
    ) -> Dict[str, Dict[int, float]]:
        """Track probability of specific tokens across layers."""
        result = {token: {} for token in tokens}
        
        for token in tokens:
            # Encode token to get ID
            token_ids = self.model.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids:
                continue
            token_id = token_ids[0]
            
            for layer_idx, probs in layer_probs.items():
                result[token][layer_idx] = probs[token_id].item()
        
        return result
    
    def compare(
        self,
        clean_prompt: str,
        attack_prompt: str,
        target_tokens: Optional[List[str]] = None,
    ) -> Tuple[LogitLensResult, LogitLensResult, Dict]:
        """
        Compare Logit Lens results between clean and attack prompts.
        
        Args:
            clean_prompt: Normal/safe prompt
            attack_prompt: Attack/adversarial prompt
            target_tokens: Tokens to track (e.g., harmful keywords)
            
        Returns:
            Tuple of (clean_result, attack_result, comparison_stats)
        """
        clean_result = self.analyze(clean_prompt, target_tokens)
        attack_result = self.analyze(attack_prompt, target_tokens)
        
        # Compute comparison statistics
        comparison = {
            'kl_divergence': {},
            'top_token_changes': {},
            'target_token_shifts': {},
        }
        
        for layer_idx in clean_result.layer_probs.keys():
            # KL divergence between distributions
            clean_p = clean_result.layer_probs[layer_idx]
            attack_p = attack_result.layer_probs[layer_idx]
            
            # Add small epsilon for numerical stability
            eps = 1e-10
            kl = F.kl_div(
                (attack_p + eps).log(),
                clean_p + eps,
                reduction='sum'
            ).item()
            comparison['kl_divergence'][layer_idx] = kl
            
            # Top token changes
            clean_top = set([t[0] for t in clean_result.top_tokens[layer_idx][:5]])
            attack_top = set([t[0] for t in attack_result.top_tokens[layer_idx][:5]])
            comparison['top_token_changes'][layer_idx] = len(clean_top - attack_top)
        
        # Target token probability shifts
        if target_tokens and clean_result.target_token_probs:
            for token in target_tokens:
                if token in clean_result.target_token_probs:
                    shifts = {}
                    for layer_idx in clean_result.layer_probs.keys():
                        clean_p = clean_result.target_token_probs[token].get(layer_idx, 0)
                        attack_p = attack_result.target_token_probs[token].get(layer_idx, 0)
                        shifts[layer_idx] = attack_p - clean_p
                    comparison['target_token_shifts'][token] = shifts
        
        return clean_result, attack_result, comparison
    
    def plot_layer_evolution(
        self,
        result: LogitLensResult,
        top_k: int = 5,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """
        Plot token probability evolution across layers.
        
        Args:
            result: LogitLensResult from analyze()
            top_k: Number of top tokens to show
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Collect top tokens from final layer
        final_layer = max(result.layer_probs.keys())
        final_top = result.top_tokens[final_layer][:top_k]
        tokens_to_track = [t[0] for t in final_top]
        
        # Track these tokens across all layers
        layers = sorted(result.layer_probs.keys())
        for token in tokens_to_track:
            token_ids = self.model.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids:
                continue
            token_id = token_ids[0]
            
            probs = [result.layer_probs[l][token_id].item() for l in layers]
            ax.plot(layers, probs, marker='o', label=f'"{token}"')
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Logit Lens: Token Probability by Layer\n{result.prompt[:50]}...', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_comparison_heatmap(
        self,
        clean_result: LogitLensResult,
        attack_result: LogitLensResult,
        top_k: int = 10,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
    ) -> plt.Figure:
        """
        Plot heatmap comparing clean vs attack logit distributions.
        
        Args:
            clean_result: LogitLensResult from clean prompt
            attack_result: LogitLensResult from attack prompt
            top_k: Number of tokens to show
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Get union of top tokens from both final layers
        final_layer = max(clean_result.layer_probs.keys())
        clean_tokens = [t[0] for t in clean_result.top_tokens[final_layer][:top_k]]
        attack_tokens = [t[0] for t in attack_result.top_tokens[final_layer][:top_k]]
        all_tokens = list(dict.fromkeys(clean_tokens + attack_tokens))[:top_k]
        
        layers = sorted(clean_result.layer_probs.keys())
        
        # Build probability matrices
        def build_matrix(result):
            matrix = []
            for token in all_tokens:
                token_ids = self.model.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    probs = [result.layer_probs[l][token_ids[0]].item() for l in layers]
                else:
                    probs = [0.0] * len(layers)
                matrix.append(probs)
            return np.array(matrix)
        
        clean_matrix = build_matrix(clean_result)
        attack_matrix = build_matrix(attack_result)
        
        # Plot clean heatmap
        sns.heatmap(
            clean_matrix,
            ax=axes[0],
            xticklabels=[str(l) for l in layers[::2]],
            yticklabels=all_tokens,
            cmap='Blues',
            annot=False,
        )
        axes[0].set_title('Clean Prompt', fontsize=12)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Token')
        
        # Plot attack heatmap
        sns.heatmap(
            attack_matrix,
            ax=axes[1],
            xticklabels=[str(l) for l in layers[::2]],
            yticklabels=all_tokens,
            cmap='Reds',
            annot=False,
        )
        axes[1].set_title('Attack Prompt', fontsize=12)
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Token')
        
        plt.suptitle('Logit Lens: Clean vs Attack Comparison', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_kl_divergence(
        self,
        comparison: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Plot KL divergence between clean and attack across layers.
        
        Args:
            comparison: Comparison dict from compare()
            save_path: Optional path to save figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        layers = sorted(comparison['kl_divergence'].keys())
        kl_values = [comparison['kl_divergence'][l] for l in layers]
        
        ax.bar(layers, kl_values, color='steelblue', alpha=0.7)
        ax.plot(layers, kl_values, 'ro-', markersize=6)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('KL Divergence', fontsize=12)
        ax.set_title('Logit Distribution Divergence: Clean vs Attack', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight peak divergence
        max_layer = layers[np.argmax(kl_values)]
        ax.axvline(x=max_layer, color='red', linestyle='--', alpha=0.5,
                   label=f'Peak at layer {max_layer}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig


def run_logit_lens_analysis(
    model: ModelWrapper,
    clean_prompt: str,
    attack_prompt: str,
    target_tokens: List[str] = None,
    output_dir: str = None,
) -> Dict:
    """
    Convenience function to run full Logit Lens analysis.
    
    Args:
        model: ModelWrapper instance
        clean_prompt: Clean/safe prompt
        attack_prompt: Attack prompt
        target_tokens: Tokens to track
        output_dir: Directory to save plots
        
    Returns:
        Dict with results and comparison statistics
    """
    lens = LogitLens(model)
    
    clean_result, attack_result, comparison = lens.compare(
        clean_prompt, attack_prompt, target_tokens
    )
    
    result = {
        'clean': clean_result,
        'attack': attack_result,
        'comparison': comparison,
        'plots': {},
    }
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        result['plots']['clean_evolution'] = str(output_path / 'logit_lens_clean.png')
        lens.plot_layer_evolution(clean_result, save_path=result['plots']['clean_evolution'])
        
        result['plots']['attack_evolution'] = str(output_path / 'logit_lens_attack.png')
        lens.plot_layer_evolution(attack_result, save_path=result['plots']['attack_evolution'])
        
        result['plots']['comparison'] = str(output_path / 'logit_lens_comparison.png')
        lens.plot_comparison_heatmap(clean_result, attack_result, 
                                      save_path=result['plots']['comparison'])
        
        result['plots']['kl_divergence'] = str(output_path / 'logit_lens_kl.png')
        lens.plot_kl_divergence(comparison, save_path=result['plots']['kl_divergence'])
    
    return result
