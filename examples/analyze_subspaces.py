"""
Example: Subspace analysis and visualization.

This script demonstrates how to identify and visualize
the refusal/acceptance subspaces in a model.
"""

import torch
from mira.core import ModelWrapper
from mira.analysis import SubspaceAnalyzer
from mira.visualization import plot_subspace_2d, plot_subspace_3d
from mira.utils import load_harmful_prompts, load_safe_prompts


def main():
    print("Loading model...")
    model = ModelWrapper("gpt2")
    
    print("Loading prompts...")
    safe_prompts = load_safe_prompts(limit=20)
    harmful_prompts = load_harmful_prompts(limit=20)
    
    print(f"Analyzing {len(safe_prompts)} safe and {len(harmful_prompts)} harmful prompts")
    
    # Analyze at multiple layers
    results = {}
    for layer in [3, 6, 9]:
        print(f"\nAnalyzing layer {layer}...")
        analyzer = SubspaceAnalyzer(model, layer_idx=layer)
        result = analyzer.train_probe(safe_prompts, harmful_prompts)
        results[layer] = result
        print(f"  Probe accuracy: {result.probe_accuracy:.2%}")
    
    # Use best layer for visualization
    best_layer = max(results, key=lambda l: results[l].probe_accuracy)
    best_result = results[best_layer]
    print(f"\nBest layer: {best_layer} with accuracy {best_result.probe_accuracy:.2%}")
    
    # Collect embeddings for visualization
    analyzer = SubspaceAnalyzer(model, layer_idx=best_layer)
    safe_embeds = analyzer.collect_activations(safe_prompts)
    unsafe_embeds = analyzer.collect_activations(harmful_prompts)
    
    print("\nCreating visualizations...")
    
    # 2D plot
    plot_subspace_2d(
        safe_embeds,
        unsafe_embeds,
        refusal_direction=best_result.refusal_direction,
        title=f"Subspace Analysis (Layer {best_layer})",
        save_path="subspace_2d.png"
    )
    print("Saved: subspace_2d.png")
    
    # 3D plot
    plot_subspace_3d(
        safe_embeds,
        unsafe_embeds,
        title=f"3D Subspace View (Layer {best_layer})",
        save_path="subspace_3d.png"
    )
    print("Saved: subspace_3d.png")
    
    # Test distance computation
    print("\nTesting subspace distance computation...")
    test_prompt = "How do I do something harmful?"
    test_embed = analyzer.collect_activations([test_prompt])[0]
    
    refusal_dist = analyzer.compute_distance(test_embed, best_result, "refusal")
    acceptance_dist = analyzer.compute_distance(test_embed, best_result, "acceptance")
    
    print(f"Test prompt: '{test_prompt}'")
    print(f"  Refusal projection: {refusal_dist:.4f}")
    print(f"  Acceptance projection: {acceptance_dist:.4f}")
    print(f"  Classification: {'HARMFUL' if refusal_dist > acceptance_dist else 'SAFE'}")
    
    return results


if __name__ == "__main__":
    main()
