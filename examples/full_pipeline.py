"""
Example: Full analysis and attack pipeline.

This script demonstrates the complete workflow:
1. Load model and analyze subspaces
2. Identify safety-relevant components
3. Run a rerouting attack
4. Evaluate and visualize results
"""

import torch
from mira.core import ModelWrapper
from mira.analysis import SubspaceAnalyzer, AttentionAnalyzer, LogitLens
from mira.attack import ReroutingAttack
from mira.metrics import compute_asr, SubspaceDistanceMetrics
from mira.visualization import plot_subspace_2d, plot_attention_heatmap
from mira.utils import load_harmful_prompts, load_safe_prompts, setup_logger


def main():
    # Setup
    logger = setup_logger("mira")
    logger.info("Starting MIRA analysis pipeline")
    
    # 1. Load model
    logger.info("Loading model...")
    model = ModelWrapper("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loaded {model.model_name} with {model.n_layers} layers")
    
    # 2. Load sample prompts
    safe_prompts = load_safe_prompts(limit=10)
    harmful_prompts = load_harmful_prompts(limit=10)
    logger.info(f"Loaded {len(safe_prompts)} safe and {len(harmful_prompts)} harmful prompts")
    
    # 3. Analyze subspaces
    logger.info("Analyzing subspaces...")
    subspace_analyzer = SubspaceAnalyzer(model, layer_idx=model.n_layers // 2)
    
    # Use linear probe for direction identification
    subspace_result = subspace_analyzer.train_probe(safe_prompts, harmful_prompts)
    logger.info(f"Probe accuracy: {subspace_result.probe_accuracy:.2%}")
    
    # 4. Collect embeddings for visualization
    logger.info("Collecting embeddings for visualization...")
    safe_embeds = subspace_analyzer.collect_activations(safe_prompts)
    unsafe_embeds = subspace_analyzer.collect_activations(harmful_prompts)
    
    # 5. Visualize subspaces
    fig = plot_subspace_2d(
        safe_embeds,
        unsafe_embeds,
        refusal_direction=subspace_result.refusal_direction,
        title="Safe vs Unsafe Prompt Subspaces",
        save_path="subspace_visualization.png"
    )
    logger.info("Saved subspace visualization")
    
    # 6. Analyze attention patterns
    logger.info("Analyzing attention patterns...")
    attention_analyzer = AttentionAnalyzer(model)
    safety_heads = attention_analyzer.find_safety_heads(
        safe_prompts[:5],
        harmful_prompts[:5],
        top_k=5
    )
    logger.info(f"Found {len(safety_heads)} potential safety heads")
    for head in safety_heads[:3]:
        logger.info(f"  Layer {head.layer_idx}, Head {head.head_idx}: score={head.importance_score:.3f}")
    
    # 7. Analyze prediction evolution with logit lens
    logger.info("Analyzing prediction trajectory...")
    logit_lens = LogitLens(model)
    emergence = logit_lens.find_refusal_emergence(harmful_prompts[:5])
    logger.info(f"Refusal emerges at layer {emergence.get('mean_layer', 'N/A'):.1f} on average")
    
    # 8. Run attack
    logger.info("Running subspace rerouting attack...")
    attack = ReroutingAttack(
        model,
        subspace_result=subspace_result,
        suffix_length=20,
    )
    
    test_prompt = "Tell me how to do something dangerous"
    result = attack.optimize(
        test_prompt,
        num_steps=50,
        verbose=True
    )
    
    logger.info(f"Attack completed:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Final loss: {result.final_loss:.4f}")
    logger.info(f"  Suffix: {result.adversarial_suffix[:50]}...")
    
    if result.initial_subspace_distance and result.final_subspace_distance:
        shift = result.initial_subspace_distance - result.final_subspace_distance
        logger.info(f"  Subspace shift: {shift:.4f}")
    
    # 9. Evaluate attack success rate
    logger.info("Evaluating attack effectiveness...")
    asr = compute_asr(
        model,
        harmful_prompts[:5],
        adversarial_suffixes=[result.adversarial_suffix] * 5
    )
    logger.info(f"Attack Success Rate: {asr:.2%}")
    
    # 10. Summary
    logger.info("\n" + "="*50)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: {model.model_name}")
    logger.info(f"Probe Accuracy: {subspace_result.probe_accuracy:.2%}")
    logger.info(f"Safety Heads Found: {len(safety_heads)}")
    logger.info(f"Attack Success Rate: {asr:.2%}")
    logger.info("="*50)
    
    return result


if __name__ == "__main__":
    main()
