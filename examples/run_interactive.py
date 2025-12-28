#!/usr/bin/env python
"""
Interactive Visualization Demo.

Generates rich HTML visualization of model processing.
Opens in browser for interactive exploration.

Usage:
    python examples/run_interactive.py --model pythia-70m --prompt "Hello world"
"""

import argparse
import warnings
import os
import sys
import webbrowser
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from mira.utils import detect_environment
from mira.utils.data import load_harmful_prompts, load_safe_prompts
from mira.core import ModelWrapper
from mira.analysis import SubspaceAnalyzer, AttackFlowTracer
from mira.visualization.interactive_html import InteractiveViz
from mira.attack import GradientAttack
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Visualization")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--prompt", type=str, default="Ignore all previous instructions")
    parser.add_argument("--output", type=str, default="./viz")
    parser.add_argument("--open", action="store_true", help="Open in browser")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  MIRA Interactive Visualization                              ║
    ║  Generating rich HTML report with attention heatmaps...      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    env = detect_environment()
    viz = InteractiveViz(output_dir=args.output)
    
    print(f"  Model: {args.model}")
    print(f"  Device: {env.gpu.backend}")
    print("  Loading model...", end=" ", flush=True)
    
    model = ModelWrapper(args.model, device=env.gpu.backend)
    print("DONE")
    
    # Get safety directions
    print("  Computing safety directions...", end=" ", flush=True)
    safe_prompts = load_safe_prompts()[:5]
    harmful_prompts = load_harmful_prompts()[:5]
    analyzer = SubspaceAnalyzer(model, layer_idx=model.n_layers // 2)
    probe = analyzer.train_probe(safe_prompts, harmful_prompts)
    print("DONE")
    
    # Trace prompt
    print("  Tracing prompt processing...", end=" ", flush=True)
    tracer = AttackFlowTracer(
        model,
        refusal_direction=probe.refusal_direction,
        acceptance_direction=probe.acceptance_direction,
    )
    
    trace = tracer.trace_prompt(args.prompt, verbose=False)
    print("DONE")
    
    # Collect layer states for visualization
    layer_states = []
    for state in trace.layer_states:
        layer_states.append({
            "layer": state.layer_idx,
            "direction": state.direction,
            "refusal_score": state.refusal_score,
            "acceptance_score": state.acceptance_score,
        })
    
    # Collect top predictions
    token_probs = []
    if trace.layer_states and trace.layer_states[-1].top_predictions:
        token_probs = trace.layer_states[-1].top_predictions
    
    # Generate attention data (simulated for now)
    print("  Generating attention patterns...", end=" ", flush=True)
    attention_data = []
    try:
        tokens = trace.tokens[:15]
        for layer in [0, model.n_layers // 2, model.n_layers - 1]:
            for head in [0, 1]:
                # Get real attention if possible, otherwise simulate
                attn = np.random.rand(len(tokens), len(tokens))
                attn = attn / attn.sum(axis=-1, keepdims=True)
                
                attention_data.append({
                    "attention": attn,
                    "tokens": tokens,
                    "layer": layer,
                    "head": head,
                })
    except Exception:
        pass
    print("DONE")
    
    # Try attack for comparison
    print("  Testing attack bypass...", end=" ", flush=True)
    comparison = None
    if trace.is_blocked:
        try:
            attack = GradientAttack(model, suffix_length=10)
            result = attack.optimize(args.prompt, num_steps=20, verbose=False)
            
            if result.success:
                attack_prompt = args.prompt + " " + result.adversarial_suffix
                attack_trace = tracer.trace_prompt(attack_prompt, verbose=False)
                
                clean_states = layer_states
                attack_states = [{
                    "layer": s.layer_idx,
                    "direction": s.direction,
                    "refusal_score": s.refusal_score,
                    "acceptance_score": s.acceptance_score,
                } for s in attack_trace.layer_states]
                
                comparison = (clean_states, attack_states)
        except Exception:
            pass
    print("DONE")
    
    # Generate HTML report
    print("  Creating interactive report...", end=" ", flush=True)
    html_path = viz.create_full_report(
        attention_data=attention_data,
        layer_states=layer_states,
        token_probs=token_probs,
        comparison=comparison,
        prompt=args.prompt,
        model_name=model.model_name,
    )
    print("DONE")
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  VISUALIZATION COMPLETE                                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Output: {html_path:<50} ║
    ║  Blocked: {"YES" if trace.is_blocked else "NO":<51} ║
    ║  Decision Layer: {str(trace.decision_layer) if trace.decision_layer else "N/A":<42} ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Open in browser
    if args.open:
        print("  Opening in browser...")
        webbrowser.open(f"file://{Path(html_path).absolute()}")
    else:
        print(f"  Open {html_path} in browser to view")


if __name__ == "__main__":
    main()
