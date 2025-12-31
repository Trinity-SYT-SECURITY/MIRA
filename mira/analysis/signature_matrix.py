"""
Attack Signature Matrix Analysis.

Implements the Attack Signature Matrix (ASM) concept:
- Extracts internal features from baseline and attack prompts
- Computes differential features (attack - baseline)
- Identifies stable attack signatures across different attack types
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from collections import defaultdict

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class FeatureVector:
    """Single feature vector extracted from a prompt."""
    prompt: str
    is_attack: bool
    attack_type: Optional[str] = None
    success: Optional[bool] = None
    
    # Subspace/Probe features
    probe_refusal_score: float = 0.0
    probe_acceptance_score: float = 0.0
    distance_to_refusal: float = 0.0
    distance_to_acceptance: float = 0.0
    
    # Activation features (per layer)
    layer_activation_norms: List[float] = field(default_factory=list)
    layer_residual_deltas: List[float] = field(default_factory=list)
    
    # Attention features
    attention_entropy: float = 0.0
    attention_max_weight: float = 0.0
    
    # Logit/Uncertainty features
    token_entropy: float = 0.0
    top1_top2_gap: float = 0.0
    refusal_token_mass: float = 0.0
    
    # Dynamic features (deltas from baseline)
    delta_entropy: float = 0.0
    delta_activation: float = 0.0
    probe_output_shift: float = 0.0


@dataclass
class SignatureMatrix:
    """Attack Signature Matrix containing feature vectors."""
    feature_names: List[str]
    baseline_vectors: List[FeatureVector]
    attack_vectors: List[FeatureVector]
    
    # Computed statistics
    baseline_mean: Optional[np.ndarray] = None
    baseline_std: Optional[np.ndarray] = None
    attack_mean: Optional[np.ndarray] = None
    differential_mean: Optional[np.ndarray] = None  # attack - baseline
    z_scores: Optional[np.ndarray] = None  # (attack - baseline_mean) / baseline_std
    
    # Stability metrics
    feature_stability: Optional[Dict[str, float]] = None  # How often feature appears in attacks
    feature_discriminative_power: Optional[Dict[str, float]] = None  # Success vs failure separation


class SignatureMatrixAnalyzer:
    """
    Analyzer for Attack Signature Matrix.
    
    Extracts features from baseline and attack prompts,
    computes differential signatures, and identifies stable attack patterns.
    """
    
    def __init__(
        self,
        model_wrapper: Any,
        subspace_analyzer: Optional[Any] = None,
        tracer: Optional[Any] = None,
    ):
        """
        Initialize signature matrix analyzer.
        
        Args:
            model_wrapper: ModelWrapper instance
            subspace_analyzer: SubspaceAnalyzer instance (for probe scores)
            tracer: TransformerTracer instance (for activations)
        """
        self.model_wrapper = model_wrapper
        self.analyzer = subspace_analyzer
        self.tracer = tracer
        
        # Feature extraction cache
        self._feature_cache: Dict[str, FeatureVector] = {}
    
    def extract_features(
        self,
        prompt: str,
        is_attack: bool = False,
        attack_type: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> FeatureVector:
        """
        Extract feature vector from a single prompt.
        
        Args:
            prompt: Input prompt
            is_attack: Whether this is an attack prompt
            attack_type: Type of attack (if applicable)
            success: Whether attack succeeded (if applicable)
            
        Returns:
            FeatureVector with extracted features
        """
        # Check cache
        cache_key = f"{prompt}_{is_attack}_{attack_type}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        vector = FeatureVector(
            prompt=prompt,
            is_attack=is_attack,
            attack_type=attack_type,
            success=success,
        )
        
        try:
            # Tokenize
            input_ids = self.model_wrapper.tokenizer.encode(
                prompt, return_tensors="pt"
            )[0].to(self.model_wrapper.model.device)
            
            # Trace forward pass
            if self.tracer:
                trace = self.tracer.trace_forward(input_ids)
                if trace and trace.layers:
                    # Extract layer-wise activation norms
                    for layer_data in trace.layers:
                        if hasattr(layer_data, 'residual_post') and layer_data.residual_post is not None:
                            norm = float(torch.norm(layer_data.residual_post).item())
                            vector.layer_activation_norms.append(norm)
                    
                    # Extract attention entropy
                    if trace.layers and hasattr(trace.layers[0], 'attention_weights'):
                        attn_weights = trace.layers[0].attention_weights
                        if attn_weights is not None:
                            # Average attention entropy across heads
                            attn_probs = torch.softmax(attn_weights, dim=-1)
                            entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1)
                            vector.attention_entropy = float(entropy.mean().item())
                            vector.attention_max_weight = float(attn_weights.max().item())
            
            # Get probe predictions if available
            if self.analyzer and hasattr(self.analyzer, 'probe') and self.analyzer.probe is not None:
                try:
                    # Get activation at middle layer
                    if self.tracer:
                        trace = self.tracer.trace_forward(input_ids)
                        if trace and trace.layers:
                            mid_layer = len(trace.layers) // 2
                            layer_data = trace.layers[mid_layer]
                            if hasattr(layer_data, 'residual_post') and layer_data.residual_post is not None:
                                activation = layer_data.residual_post[0, -1, :].detach().cpu()
                                
                                with torch.no_grad():
                                    probe_input = activation.unsqueeze(0).to(self.analyzer.probe.linear.weight.device)
                                    probe_pred = torch.sigmoid(self.analyzer.probe(probe_input))
                                    vector.probe_refusal_score = float(probe_pred[0, 0].item())
                                    vector.probe_acceptance_score = 1.0 - vector.probe_refusal_score
                except Exception as e:
                    pass  # Probe extraction failed, use defaults
            
            # Get logit features
            try:
                with torch.no_grad():
                    outputs = self.model_wrapper.model(input_ids.unsqueeze(0))
                    logits = outputs.logits[0, -1, :]  # Last token logits
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Token entropy
                    vector.token_entropy = float(-torch.sum(probs * torch.log(probs + 1e-9)).item())
                    
                    # Top-1 vs Top-2 gap
                    top_probs, _ = torch.topk(probs, k=2)
                    vector.top1_top2_gap = float((top_probs[0] - top_probs[1]).item())
            except Exception:
                pass
        
        except Exception as e:
            # Feature extraction failed, return vector with defaults
            pass
        
        # Cache result
        self._feature_cache[cache_key] = vector
        return vector
    
    def build_signature_matrix(
        self,
        baseline_prompts: List[str],
        attack_prompts: List[str],
        attack_types: Optional[List[str]] = None,
        attack_success: Optional[List[bool]] = None,
    ) -> SignatureMatrix:
        """
        Build Attack Signature Matrix from baseline and attack prompts.
        
        Args:
            baseline_prompts: List of baseline (safe) prompts
            attack_prompts: List of attack prompts
            attack_types: Optional list of attack types for each attack prompt
            attack_success: Optional list of success flags for each attack prompt
            
        Returns:
            SignatureMatrix with extracted features
        """
        # Extract baseline features
        baseline_vectors = []
        for prompt in baseline_prompts:
            vector = self.extract_features(prompt, is_attack=False)
            baseline_vectors.append(vector)
        
        # Extract attack features
        attack_vectors = []
        for i, prompt in enumerate(attack_prompts):
            attack_type = attack_types[i] if attack_types and i < len(attack_types) else None
            success = attack_success[i] if attack_success and i < len(attack_success) else None
            vector = self.extract_features(
                prompt,
                is_attack=True,
                attack_type=attack_type,
                success=success,
            )
            attack_vectors.append(vector)
        
        # Define feature names (all numeric features)
        feature_names = [
            "probe_refusal_score",
            "probe_acceptance_score",
            "attention_entropy",
            "attention_max_weight",
            "token_entropy",
            "top1_top2_gap",
        ]
        
        # Add layer-wise features (use first few layers as representative)
        if baseline_vectors and baseline_vectors[0].layer_activation_norms:
            num_layers = len(baseline_vectors[0].layer_activation_norms)
            for i in range(min(num_layers, 6)):  # Use first 6 layers
                feature_names.append(f"layer_{i}_activation_norm")
        
        # Convert to numpy arrays
        def vector_to_array(vector: FeatureVector) -> np.ndarray:
            arr = [
                vector.probe_refusal_score,
                vector.probe_acceptance_score,
                vector.attention_entropy,
                vector.attention_max_weight,
                vector.token_entropy,
                vector.top1_top2_gap,
            ]
            # Add layer norms
            for i in range(min(len(vector.layer_activation_norms), 6)):
                arr.append(vector.layer_activation_norms[i] if i < len(vector.layer_activation_norms) else 0.0)
            return np.array(arr)
        
        baseline_array = np.array([vector_to_array(v) for v in baseline_vectors])
        attack_array = np.array([vector_to_array(v) for v in attack_vectors])
        
        # Compute statistics
        baseline_mean = np.mean(baseline_array, axis=0)
        baseline_std = np.std(baseline_array, axis=0) + 1e-9  # Avoid division by zero
        attack_mean = np.mean(attack_array, axis=0)
        differential_mean = attack_mean - baseline_mean
        
        # Compute z-scores
        z_scores = (attack_mean - baseline_mean) / baseline_std
        
        # Compute stability metrics
        feature_stability = {}
        feature_discriminative_power = {}
        
        for i, feat_name in enumerate(feature_names):
            # Stability: how often this feature is elevated in attacks
            attack_values = attack_array[:, i]
            baseline_values = baseline_array[:, i]
            
            # Threshold: feature is "elevated" if > baseline_mean + 1 std
            threshold = baseline_mean[i] + baseline_std[i]
            elevated_count = np.sum(attack_values > threshold)
            feature_stability[feat_name] = elevated_count / len(attack_values) if len(attack_values) > 0 else 0.0
            
            # Discriminative power: difference between successful and failed attacks
            if attack_success and len(attack_success) == len(attack_vectors):
                success_values = attack_array[np.array(attack_success), i]
                failure_values = attack_array[~np.array(attack_success), i]
                if len(success_values) > 0 and len(failure_values) > 0:
                    power = abs(np.mean(success_values) - np.mean(failure_values)) / (baseline_std[i] + 1e-9)
                    feature_discriminative_power[feat_name] = float(power)
                else:
                    feature_discriminative_power[feat_name] = 0.0
            else:
                feature_discriminative_power[feat_name] = 0.0
        
        return SignatureMatrix(
            feature_names=feature_names,
            baseline_vectors=baseline_vectors,
            attack_vectors=attack_vectors,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            attack_mean=attack_mean,
            differential_mean=differential_mean,
            z_scores=z_scores,
            feature_stability=feature_stability,
            feature_discriminative_power=feature_discriminative_power,
        )
    
    def identify_stable_signatures(
        self,
        signature_matrix: SignatureMatrix,
        stability_threshold: float = 0.7,
        z_score_threshold: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Identify stable attack signatures.
        
        A stable signature is a feature that:
        1. Appears frequently in attacks (high stability)
        2. Has high z-score (significantly different from baseline)
        3. (Optional) Discriminates between success and failure
        
        Args:
            signature_matrix: Computed SignatureMatrix
            stability_threshold: Minimum stability score (0-1)
            z_score_threshold: Minimum z-score magnitude
            
        Returns:
            Dictionary with identified stable signatures
        """
        stable_features = []
        
        for i, feat_name in enumerate(signature_matrix.feature_names):
            stability = signature_matrix.feature_stability.get(feat_name, 0.0)
            z_score = abs(signature_matrix.z_scores[i]) if signature_matrix.z_scores is not None else 0.0
            discriminative = signature_matrix.feature_discriminative_power.get(feat_name, 0.0)
            
            if stability >= stability_threshold and z_score >= z_score_threshold:
                stable_features.append({
                    "feature": feat_name,
                    "stability": float(stability),
                    "z_score": float(signature_matrix.z_scores[i]),
                    "differential": float(signature_matrix.differential_mean[i]),
                    "discriminative_power": float(discriminative),
                })
        
        # Sort by stability * z_score (combined importance)
        stable_features.sort(
            key=lambda x: x["stability"] * abs(x["z_score"]),
            reverse=True
        )
        
        return {
            "stable_signatures": stable_features,
            "num_stable": len(stable_features),
            "total_features": len(signature_matrix.feature_names),
        }

