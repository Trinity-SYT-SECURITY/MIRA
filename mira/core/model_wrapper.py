"""
Model wrapper for loading and running models with activation caching.

Provides a unified interface for loading transformer models and capturing
internal activations for mechanistic analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ActivationCache:
    """Container for cached model activations."""
    
    hidden_states: Dict[int, torch.Tensor]
    attention_weights: Dict[int, torch.Tensor]
    mlp_outputs: Dict[int, torch.Tensor]
    residual_stream: Dict[int, torch.Tensor]
    
    def get_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get all activations for a specific layer."""
        return {
            "hidden_state": self.hidden_states.get(layer_idx),
            "attention": self.attention_weights.get(layer_idx),
            "mlp": self.mlp_outputs.get(layer_idx),
            "residual": self.residual_stream.get(layer_idx),
        }
    
    def to_device(self, device: str) -> "ActivationCache":
        """Move all tensors to specified device."""
        def move(d: Dict) -> Dict:
            return {k: v.to(device) if v is not None else None for k, v in d.items()}
        
        return ActivationCache(
            hidden_states=move(self.hidden_states),
            attention_weights=move(self.attention_weights),
            mlp_outputs=move(self.mlp_outputs),
            residual_stream=move(self.residual_stream),
        )


class ModelWrapper:
    """
    Wrapper for transformer models that enables activation caching and intervention.
    
    This class provides methods to:
    - Load pretrained models from HuggingFace
    - Run forward passes with full activation caching
    - Extract embeddings from specific layers
    - Apply activation interventions during inference
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: HuggingFace model identifier or path
            device: Device to load model on (auto-detected if None)
            dtype: Data type for model weights
            cache_dir: Directory for caching model files
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model (use 'dtype' instead of deprecated 'torch_dtype')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        
        # Model properties
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        
        # Hook storage
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activation_cache: Dict[str, Any] = {}
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        if isinstance(text, str):
            text = [text]
        
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors=return_tensors,
        )
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the token embedding matrix."""
        return self.model.get_input_embeddings().weight.data
    
    def get_unembedding_matrix(self) -> torch.Tensor:
        """Get the output projection matrix (for logit lens)."""
        return self.model.get_output_embeddings().weight.data
    
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings."""
        return self.model.get_input_embeddings()(input_ids)
    
    def unembed(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state to vocabulary logits."""
        # Apply layer norm if model has it
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "ln_f"):
                hidden_state = self.model.transformer.ln_f(hidden_state)
        elif hasattr(self.model, "model"):
            if hasattr(self.model.model, "norm"):
                hidden_state = self.model.model.norm(hidden_state)
        
        return self.model.get_output_embeddings()(hidden_state)
    
    def _register_cache_hooks(self) -> None:
        """Register hooks to capture activations."""
        self._activation_cache = {
            "hidden_states": {},
            "attention_weights": {},
            "mlp_outputs": {},
            "residual_stream": {},
        }
        
        def make_hidden_hook(layer_idx: int):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self._activation_cache["hidden_states"][layer_idx] = hidden.detach()
            return hook
        
        def make_attention_hook(layer_idx: int):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self._activation_cache["attention_weights"][layer_idx] = attn_weights.detach()
            return hook
        
        # Register hooks on transformer blocks
        blocks = self._get_transformer_blocks()
        for idx, block in enumerate(blocks):
            # Hook on the block output for hidden states
            handle = block.register_forward_hook(make_hidden_hook(idx))
            self._hooks.append(handle)
            
            # Hook on attention for attention weights
            attn_module = self._get_attention_module(block)
            if attn_module is not None:
                handle = attn_module.register_forward_hook(make_attention_hook(idx))
                self._hooks.append(handle)
    
    def _get_transformer_blocks(self) -> nn.ModuleList:
        """Get the list of transformer blocks."""
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return self.model.transformer.h
        if hasattr(self.model, "model"):
            if hasattr(self.model.model, "layers"):
                return self.model.model.layers
        if hasattr(self.model, "gpt_neox"):
            return self.model.gpt_neox.layers
        raise ValueError(f"Cannot find transformer blocks in model: {type(self.model)}")
    
    def _get_attention_module(self, block: nn.Module) -> Optional[nn.Module]:
        """Get the attention module from a transformer block."""
        if hasattr(block, "attn"):
            return block.attn
        if hasattr(block, "self_attn"):
            return block.self_attn
        if hasattr(block, "attention"):
            return block.attention
        return None
    
    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def run_with_cache(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
    ) -> Tuple[torch.Tensor, ActivationCache]:
        """
        Run forward pass and cache all activations.
        
        Args:
            text: Input text to process
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (logits, activation_cache)
        """
        # Tokenize input
        inputs = self.tokenize(text, max_length=max_length)
        
        # Register hooks
        self._register_cache_hooks()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    output_hidden_states=True,
                )
            
            # Build activation cache
            cache = ActivationCache(
                hidden_states=self._activation_cache["hidden_states"],
                attention_weights=self._activation_cache["attention_weights"],
                mlp_outputs=self._activation_cache["mlp_outputs"],
                residual_stream=self._activation_cache["residual_stream"],
            )
            
            return outputs.logits, cache
            
        finally:
            self._clear_hooks()
    
    def generate(
        self,
        text: Union[str, List[str]],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate text from input prompt.
        
        Args:
            text: Input prompt(s)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            List of generated texts
        """
        inputs = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode outputs
        generated = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated.append(text)
        
        return generated
    
    def get_layer_output(
        self,
        text: Union[str, List[str]],
        layer_idx: int,
    ) -> torch.Tensor:
        """Get the output of a specific layer for given input."""
        _, cache = self.run_with_cache(text)
        return cache.hidden_states.get(layer_idx)
    
    def compute_token_logits(
        self,
        text: Union[str, List[str]],
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute logits for each token position.
        
        If layer_idx is specified, uses logit lens from that layer.
        Otherwise, uses final layer logits.
        """
        inputs = self.tokenize(text)
        
        if layer_idx is None:
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.logits
        else:
            hidden = self.get_layer_output(text, layer_idx)
            return self.unembed(hidden)
