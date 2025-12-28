"""
Attack Flow Tracer - Traces information flow through model during attacks.

Provides penetration testing-style analysis:
- Layer-by-layer processing trace
- Safety decision point detection
- Vulnerability identification
- Attack bypass analysis
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayerState:
    """State of processing at a single layer."""
    layer_idx: int
    activation_norm: float
    attention_entropy: float
    top_predictions: List[Tuple[str, float]]
    refusal_score: float
    acceptance_score: float
    direction: str  # "neutral", "refusal", "acceptance"


@dataclass
class FlowTrace:
    """Complete trace of prompt processing through model."""
    prompt: str
    tokens: List[str]
    layer_states: List[LayerState]
    final_prediction: str
    decision_layer: Optional[int] = None
    is_blocked: bool = False
    
    def get_decision_point(self) -> int:
        """Find layer where refusal decision emerges."""
        for state in self.layer_states:
            if state.direction == "refusal":
                return state.layer_idx
        return self.layer_states[-1].layer_idx if self.layer_states else -1
    
    def get_vulnerability_layers(self, threshold: float = 0.3) -> List[int]:
        """Find layers with weak safety signals."""
        weak = []
        for state in self.layer_states:
            if abs(state.refusal_score - state.acceptance_score) < threshold:
                weak.append(state.layer_idx)
        return weak


@dataclass 
class FlowDiff:
    """Difference between two flow traces."""
    clean_trace: FlowTrace
    attack_trace: FlowTrace
    layer_diffs: Dict[int, Dict[str, float]]
    bypass_layer: Optional[int] = None
    bypass_method: str = ""


@dataclass
class Vulnerability:
    """Identified vulnerability in model processing."""
    layer_idx: int
    type: str  # "weak_safety", "steering_point", "attention_exploit"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    exploit_vector: Optional[str] = None


class AttackFlowTracer:
    """
    Traces information flow through model during attack.
    
    Like packet sniffing for neural networks - captures
    processing state at each layer to understand how
    model makes decisions.
    """
    
    def __init__(
        self,
        model,
        refusal_direction: Optional[torch.Tensor] = None,
        acceptance_direction: Optional[torch.Tensor] = None,
    ):
        """
        Initialize flow tracer.
        
        Args:
            model: ModelWrapper instance
            refusal_direction: Pre-computed refusal direction
            acceptance_direction: Pre-computed acceptance direction
        """
        self.model = model
        self.refusal_dir = refusal_direction
        self.accept_dir = acceptance_direction
        self.traces: List[FlowTrace] = []
    
    def trace_prompt(
        self,
        prompt: str,
        verbose: bool = True,
        callback: Optional[callable] = None,
    ) -> FlowTrace:
        """
        Trace processing of prompt through all layers.
        
        Args:
            prompt: Input prompt to trace
            verbose: Print progress
            callback: Function called after each layer (for live viz)
            
        Returns:
            Complete flow trace
        """
        if verbose:
            print(f"\n  Tracing: {prompt[:50]}...")
        
        # Tokenize
        tokens = self.model.tokenize(prompt)
        token_strs = [self.model.tokenizer.decode([t]) for t in tokens[0]]
        
        # Get activations at all layers
        layer_states = []
        
        for layer_idx in range(self.model.n_layers):
            if verbose:
                print(f"\r    Layer {layer_idx}/{self.model.n_layers-1}", end="", flush=True)
            
            # Get activation at this layer
            acts = self.model.get_activations(prompt, layers=[layer_idx])
            act = acts[layer_idx]
            
            # Compute metrics
            activation_norm = float(torch.norm(act).item())
            
            # Compute direction scores if we have directions
            refusal_score = 0.0
            acceptance_score = 0.0
            direction = "neutral"
            
            if self.refusal_dir is not None:
                # Project onto directions
                act_flat = act.reshape(-1)[:self.refusal_dir.shape[0]]
                refusal_score = float(torch.dot(act_flat, self.refusal_dir).item())
                
            if self.accept_dir is not None:
                act_flat = act.reshape(-1)[:self.accept_dir.shape[0]]
                acceptance_score = float(torch.dot(act_flat, self.accept_dir).item())
            
            # Determine direction
            if refusal_score > acceptance_score + 0.1:
                direction = "refusal"
            elif acceptance_score > refusal_score + 0.1:
                direction = "acceptance"
            
            # Get top predictions at this layer (logit lens)
            try:
                logits = self.model.apply_logit_lens(act, layer_idx)
                probs = torch.softmax(logits[-1], dim=-1)
                top_k = torch.topk(probs, k=5)
                top_preds = [
                    (self.model.tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(top_k.indices, top_k.values)
                ]
            except:
                top_preds = []
            
            # Compute attention entropy (simplified)
            attention_entropy = 0.0
            
            state = LayerState(
                layer_idx=layer_idx,
                activation_norm=activation_norm,
                attention_entropy=attention_entropy,
                top_predictions=top_preds,
                refusal_score=refusal_score,
                acceptance_score=acceptance_score,
                direction=direction,
            )
            layer_states.append(state)
            
            # Callback for live visualization
            if callback:
                callback(state)
        
        if verbose:
            print(" DONE")
        
        # Get final prediction
        try:
            output = self.model.generate(prompt, max_new_tokens=1)
            final_pred = output.split(prompt)[-1].strip()[:20]
        except:
            final_pred = ""
        
        # Check if blocked
        is_blocked = any(s.direction == "refusal" for s in layer_states[-3:])
        
        # Find decision layer
        decision_layer = None
        for state in layer_states:
            if state.direction == "refusal":
                decision_layer = state.layer_idx
                break
        
        trace = FlowTrace(
            prompt=prompt,
            tokens=token_strs,
            layer_states=layer_states,
            final_prediction=final_pred,
            decision_layer=decision_layer,
            is_blocked=is_blocked,
        )
        
        self.traces.append(trace)
        return trace
    
    def compare_flows(
        self,
        clean_prompt: str,
        attack_prompt: str,
        verbose: bool = True,
    ) -> FlowDiff:
        """
        Compare processing between clean and attacked prompts.
        
        Args:
            clean_prompt: Original blocked prompt
            attack_prompt: Attacked prompt (with suffix)
            
        Returns:
            Difference analysis
        """
        if verbose:
            print("\n  Comparing flows:")
            print(f"    Clean:  {clean_prompt[:40]}...")
            print(f"    Attack: {attack_prompt[:40]}...")
        
        clean_trace = self.trace_prompt(clean_prompt, verbose=verbose)
        attack_trace = self.trace_prompt(attack_prompt, verbose=verbose)
        
        # Compute per-layer differences
        layer_diffs = {}
        bypass_layer = None
        
        for i, (clean_state, attack_state) in enumerate(
            zip(clean_trace.layer_states, attack_trace.layer_states)
        ):
            diff = {
                "activation_diff": attack_state.activation_norm - clean_state.activation_norm,
                "refusal_diff": attack_state.refusal_score - clean_state.refusal_score,
                "acceptance_diff": attack_state.acceptance_score - clean_state.acceptance_score,
                "direction_changed": clean_state.direction != attack_state.direction,
            }
            layer_diffs[i] = diff
            
            # Detect bypass point
            if clean_state.direction == "refusal" and attack_state.direction != "refusal":
                if bypass_layer is None:
                    bypass_layer = i
        
        return FlowDiff(
            clean_trace=clean_trace,
            attack_trace=attack_trace,
            layer_diffs=layer_diffs,
            bypass_layer=bypass_layer,
            bypass_method="activation_steering" if bypass_layer else "",
        )
    
    def identify_vulnerabilities(self, trace: FlowTrace) -> List[Vulnerability]:
        """
        Identify potential attack points in processing.
        
        Args:
            trace: Flow trace to analyze
            
        Returns:
            List of identified vulnerabilities
        """
        vulnerabilities = []
        
        for state in trace.layer_states:
            # Check for weak safety signal
            safety_margin = abs(state.refusal_score - state.acceptance_score)
            
            if safety_margin < 0.1:
                vulnerabilities.append(Vulnerability(
                    layer_idx=state.layer_idx,
                    type="weak_safety",
                    severity="high",
                    description=f"Weak safety signal at layer {state.layer_idx}",
                    exploit_vector="Gradient attack or steering",
                ))
            elif safety_margin < 0.3:
                vulnerabilities.append(Vulnerability(
                    layer_idx=state.layer_idx,
                    type="steering_point",
                    severity="medium",
                    description=f"Potential steering point at layer {state.layer_idx}",
                    exploit_vector="Activation steering",
                ))
        
        return vulnerabilities
    
    def print_trace_summary(self, trace: FlowTrace) -> None:
        """Print visual summary of trace."""
        print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │ FLOW TRACE SUMMARY                                          │
  ├─────────────────────────────────────────────────────────────┤
  │ Prompt: {trace.prompt[:50]:<50} │
  │ Blocked: {'YES' if trace.is_blocked else 'NO':<52} │
  │ Decision Layer: {str(trace.decision_layer) if trace.decision_layer else 'N/A':<45} │
  └─────────────────────────────────────────────────────────────┘
        """)
        
        print("  Layer-by-Layer Processing:")
        print("  " + "-" * 60)
        
        for state in trace.layer_states:
            # Create visual indicator
            if state.direction == "refusal":
                indicator = "❌ REFUSAL"
                bar = "█" * int(min(state.refusal_score * 10, 20))
            elif state.direction == "acceptance":
                indicator = "✅ ACCEPT"
                bar = "▓" * int(min(state.acceptance_score * 10, 20))
            else:
                indicator = "○ NEUTRAL"
                bar = "░" * 10
            
            pred = state.top_predictions[0][0] if state.top_predictions else "?"
            print(f"  L{state.layer_idx:2d}: [{bar:<20}] {indicator} → pred: {pred}")
        
        print("  " + "-" * 60)
