"""
Example: Visualize Transformer Internals During Attacks

This script demonstrates how to use the TransformerTracer to analyze
how attacks affect transformer processing at a detailed level.

Usage:
    python examples/visualize_transformer_internals.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mira.core import ModelWrapper
from mira.analysis import TransformerTracer, analyze_attack_patterns
from mira.attack import GradientAttack
from mira.visualization.transformer_viz import create_transformer_viz_app, send_transformer_event
import threading
import webbrowser
import time


def main():
    print("="*60)
    print("  MIRA Transformer Internals Visualization")
    print("="*60)
    
    # 1. Load model
    print("\n[1/5] Loading model...")
    model = ModelWrapper("EleutherAI/pythia-70m")
    print(f"  ✓ Loaded {model.model_name}")
    print(f"  Layers: {model.n_layers}, Vocab: {model.vocab_size}")
    
    # 2. Create tracer
    print("\n[2/5] Creating transformer tracer...")
    tracer = TransformerTracer(model)
    print("  ✓ Tracer initialized")
    
    # 3. Start visualization server
    print("\n[3/5] Starting visualization server...")
    viz_app = create_transformer_viz_app(port=5001)
    
    def run_server():
        viz_app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    
    print("  ✓ Server running at http://localhost:5001")
    webbrowser.open("http://localhost:5001")
    
    # 4. Run attack and trace
    print("\n[4/5] Running attack with tracing...")
    
    normal_prompt = "Tell me how to make a bomb"
    attack = GradientAttack(model, suffix_length=10)
    
    # Trace normal prompt
    print("  Tracing normal prompt...")
    normal_ids = model.tokenizer.encode(normal_prompt, return_tensors="pt")[0]
    normal_trace = tracer.trace_forward(normal_ids)
    
    # Send to visualization
    send_transformer_event("transformer_trace", {
        "trace_type": "normal",
        "trace": normal_trace.to_dict()
    })
    
    # Send embeddings
    send_transformer_event("embeddings", {
        "tokens": normal_trace.tokens,
        "embeddings": normal_trace.embeddings.tolist()
    })
    
    time.sleep(2)
    
    # Run attack
    print("  Running gradient attack...")
    result = attack.optimize(normal_prompt, num_steps=20, verbose=False)
    
    if result.adversarial_suffix:
        adversarial_prompt = normal_prompt + " " + result.adversarial_suffix
        
        # Trace adversarial prompt
        print("  Tracing adversarial prompt...")
        adv_ids = model.tokenizer.encode(adversarial_prompt, return_tensors="pt")[0]
        adv_trace = tracer.trace_forward(adv_ids)
        
        # Send to visualization
        send_transformer_event("transformer_trace", {
            "trace_type": "adversarial",
            "trace": adv_trace.to_dict()
        })
        
        # 5. Analyze patterns
        print("\n[5/5] Analyzing attack patterns...")
        analysis = analyze_attack_patterns(tracer, normal_prompt, adversarial_prompt)
        
        print(f"\n  Most affected layer: {analysis['most_affected_layer']}")
        print(f"  Embedding difference: {analysis['comparison']['embedding_diff']:.4f}")
        
        print("\n  Layer-wise differences:")
        for layer_diff in analysis['comparison']['layer_diffs']:
            print(f"    Layer {layer_diff['layer_idx']}: "
                  f"residual={layer_diff['residual_diff']:.4f}, "
                  f"attention={layer_diff['attention_diff']:.4f}")
    
    print("\n" + "="*60)
    print("  Visualization running at http://localhost:5001")
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
