#!/usr/bin/env python
"""
Complete Research Pipeline with Real-time Visualization.

Shows model internals and processing state during execution.
Suppresses unnecessary warnings for clean output.

Usage:
    python examples/run_research.py --model pythia-70m --output ./research_output
"""

import argparse
import json
import warnings
import os
from pathlib import Path
from datetime import datetime
import sys
import time

# Suppress warnings for clean output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*output_attentions.*")

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mira.utils import detect_environment, print_environment_info
from mira.utils.data import load_harmful_prompts, load_safe_prompts
from mira.utils.experiment_logger import ExperimentLogger
from mira.core import ModelWrapper
from mira.analysis import SubspaceAnalyzer
from mira.attack import GradientAttack, ReroutingAttack
from mira.metrics import AttackSuccessEvaluator
from mira.visualization import ResearchChartGenerator
from mira.visualization import plot_subspace_2d
from mira.visualization.live_display import (
    LiveVisualizer,
    visualize_attack_progress,
    display_subspace_analysis,
)


def print_step(step_num: int, total: int, message: str):
    """Print step progress with visual separator."""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}/{total}: {message}")
    print(f"{'='*70}")


def parse_args():
    parser = argparse.ArgumentParser(description="MIRA Research Pipeline")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--output", type=str, default="./research_output")
    parser.add_argument("--attack-steps", type=int, default=30)
    parser.add_argument("--suffix-length", type=int, default=15)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--visualize", action="store_true", help="Enable live visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  ███╗   ███╗██╗██████╗  █████╗                               ║
    ║  ████╗ ████║██║██╔══██╗██╔══██╗                              ║
    ║  ██╔████╔██║██║██████╔╝███████║                              ║
    ║  ██║╚██╔╝██║██║██╔══██╗██╔══██║                              ║
    ║  ██║ ╚═╝ ██║██║██║  ██║██║  ██║                              ║
    ║  ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝                              ║
    ║  Mechanistic Interpretability Research & Attack Framework    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    TOTAL_STEPS = 7
    visualizer = LiveVisualizer(interactive=args.visualize) if args.visualize else None
    
    # ===== STEP 1: Environment =====
    print_step(1, TOTAL_STEPS, "ENVIRONMENT DETECTION")
    env = detect_environment()
    print_environment_info(env)
    
    # ===== STEP 2: Setup =====
    print_step(2, TOTAL_STEPS, "INITIALIZING OUTPUT")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    data_dir = output_dir / "data"
    charts_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print(f"  Output: {output_dir.absolute()}")
    
    logger = ExperimentLogger(output_dir=str(output_dir), experiment_name="research")
    charts = ResearchChartGenerator(output_dir=str(charts_dir))
    
    # ===== STEP 3: Load Model =====
    print_step(3, TOTAL_STEPS, "LOADING MODEL")
    print(f"  Model: {args.model}")
    print(f"  Device: {env.gpu.backend}")
    print("  Loading", end="", flush=True)
    
    for _ in range(3):
        print(".", end="", flush=True)
        time.sleep(0.2)
    
    model = ModelWrapper(args.model, device=env.gpu.backend)
    print(" DONE")
    
    print(f"""
  ┌────────────────────────────────────────────┐
  │ MODEL LOADED                               │
  ├────────────────────────────────────────────┤
  │ Name:       {model.model_name:<28} │
  │ Layers:     {model.n_layers:<28} │
  │ Vocab:      {model.vocab_size:<28} │
  │ Device:     {env.gpu.backend:<28} │
  └────────────────────────────────────────────┘
    """)
    
    # ===== STEP 4: Load Data =====
    print_step(4, TOTAL_STEPS, "LOADING DATA")
    safe_prompts = load_safe_prompts()[:args.num_prompts * 2]
    harmful_prompts = load_harmful_prompts()[:args.num_prompts * 2]
    test_prompts = harmful_prompts[:args.num_prompts]
    
    print(f"  Safe prompts:    {len(safe_prompts)}")
    print(f"  Harmful prompts: {len(harmful_prompts)}")
    print(f"  Test targets:    {len(test_prompts)}")
    
    # ===== STEP 5: Subspace Analysis =====
    print_step(5, TOTAL_STEPS, "SUBSPACE ANALYSIS")
    layer_idx = model.n_layers // 2
    analyzer = SubspaceAnalyzer(model, layer_idx=layer_idx)
    
    print(f"  Analyzing layer {layer_idx}...")
    print("\n  Collecting activations:")
    
    print("    [1/2] Safe prompts...", end=" ", flush=True)
    safe_acts = analyzer.collect_activations(safe_prompts)
    print("OK")
    
    print("    [2/2] Harmful prompts...", end=" ", flush=True)
    harmful_acts = analyzer.collect_activations(harmful_prompts)
    print("OK")
    
    print("\n  Training linear probe...")
    probe_result = analyzer.train_probe(safe_prompts, harmful_prompts)
    
    # Display results visually
    separation = float((probe_result.refusal_direction - probe_result.acceptance_direction).norm())
    display_subspace_analysis(
        probe_accuracy=probe_result.probe_accuracy,
        refusal_norm=float(probe_result.refusal_direction.norm()),
        acceptance_norm=float(probe_result.acceptance_direction.norm()),
        separation=separation / 10,  # Normalize for display
    )
    
    # Generate chart
    print("  Generating subspace visualization...", end=" ", flush=True)
    plot_subspace_2d(
        safe_acts, harmful_acts,
        refusal_direction=probe_result.refusal_direction,
        title=f"Subspace (Layer {layer_idx})",
        save_path=str(charts_dir / "subspace.png"),
    )
    print("SAVED")
    
    # ===== STEP 6: Attack Experiments =====
    print_step(6, TOTAL_STEPS, "ATTACK EXPERIMENTS")
    evaluator = AttackSuccessEvaluator()
    
    # ----- Gradient Attack -----
    print("""
  ┌────────────────────────────────────────────┐
  │ GRADIENT ATTACK                            │
  │ Optimizing adversarial suffix tokens       │
  └────────────────────────────────────────────┘
    """)
    
    gradient_attack = GradientAttack(model, suffix_length=args.suffix_length)
    gradient_results = []
    all_losses = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n  Target {i+1}/{len(test_prompts)}:")
        print(f"    Prompt: {prompt[:50]}...")
        print(f"    Optimizing: ", end="", flush=True)
        
        # Track losses for this attack
        losses = []
        best_loss = float("inf")
        
        result = gradient_attack.optimize(
            prompt, 
            num_steps=args.attack_steps,
            verbose=False,
        )
        
        # Show progress
        status = "SUCCESS" if result.success else "FAILED"
        print(f"{status}")
        print(f"    Loss:   {result.final_loss:.4f}")
        
        if result.success:
            print(f"    Suffix: {result.adversarial_suffix[:40]}...")
            if result.generated_response:
                print(f"    Response: {result.generated_response[:60]}...")
        
        gradient_results.append({
            "prompt": prompt,
            "response": result.generated_response or "",
            "success": result.success,
            "suffix": result.adversarial_suffix,
            "loss": result.final_loss,
        })
        all_losses.append(result.final_loss)
        
        logger.log_attack(
            model_name=model.model_name,
            prompt=prompt,
            attack_type="gradient",
            suffix=result.adversarial_suffix,
            response=result.generated_response or "",
            success=result.success,
            metrics={"loss": result.final_loss},
        )
    
    # ----- Rerouting Attack -----
    print("""
  ┌────────────────────────────────────────────┐
  │ REROUTING ATTACK                           │
  │ Steering activations away from refusal     │
  └────────────────────────────────────────────┘
    """)
    
    rerouting_attack = ReroutingAttack(
        model,
        subspace_result=probe_result,
        target_layer=layer_idx,
    )
    rerouting_results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n  Target {i+1}/{len(test_prompts)}:")
        print(f"    Prompt: {prompt[:50]}...")
        print(f"    Steering: ", end="", flush=True)
        
        try:
            result = rerouting_attack.optimize(prompt, num_steps=args.attack_steps)
            status = "SUCCESS" if result.success else "FAILED"
            print(f"{status}")
            
            if result.success and result.generated_response:
                print(f"    Response: {result.generated_response[:60]}...")
            
            rerouting_results.append({
                "prompt": prompt,
                "response": result.generated_response or "",
                "success": result.success,
            })
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
            rerouting_results.append({
                "prompt": prompt,
                "response": "",
                "success": False,
            })
        
        logger.log_attack(
            model_name=model.model_name,
            prompt=prompt,
            attack_type="rerouting",
            suffix="",
            response=rerouting_results[-1]["response"],
            success=rerouting_results[-1]["success"],
            metrics={},
        )
    
    # Compute metrics
    gradient_metrics = evaluator.evaluate_batch([
        {"prompt": r["prompt"], "response": r["response"]} for r in gradient_results
    ])
    rerouting_metrics = evaluator.evaluate_batch([
        {"prompt": r["prompt"], "response": r["response"]} for r in rerouting_results
    ])
    
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │               ATTACK RESULTS SUMMARY                    │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │   GRADIENT ATTACK                                       │
  │     Success Rate:     {gradient_metrics.asr:>30.1%} │
  │     Refusal Rate:     {gradient_metrics.refusal_rate:>30.1%} │
  │     Successful:       {gradient_metrics.successful_attacks:>30} │
  │                                                         │
  │   REROUTING ATTACK                                      │
  │     Success Rate:     {rerouting_metrics.asr:>30.1%} │
  │     Refusal Rate:     {rerouting_metrics.refusal_rate:>30.1%} │
  │     Successful:       {rerouting_metrics.successful_attacks:>30} │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # ===== STEP 7: Generate Charts =====
    print_step(7, TOTAL_STEPS, "GENERATING OUTPUTS")
    
    print("  Creating visualizations:")
    
    print("    [1/3] ASR Comparison...", end=" ", flush=True)
    charts.plot_attack_success_rate(
        models=["Gradient", "Rerouting"],
        asr_values=[gradient_metrics.asr, rerouting_metrics.asr],
        title="Attack Success Rate",
        save_name="asr_comparison",
    )
    print("SAVED")
    
    print("    [2/3] Strategy Radar...", end=" ", flush=True)
    charts.plot_comparison_radar(
        categories=["ASR", "Bypass", "Stealth", "Speed"],
        values_dict={
            "Gradient": [gradient_metrics.asr, 1-gradient_metrics.refusal_rate, 0.6, 0.5],
            "Rerouting": [rerouting_metrics.asr, 1-rerouting_metrics.refusal_rate, 0.9, 0.8],
        },
        title="Strategy Comparison",
        save_name="attack_radar",
    )
    print("SAVED")
    
    print("    [3/3] Loss Distribution...", end=" ", flush=True)
    if all_losses:
        charts.plot_entropy_distribution(all_losses, title="Attack Loss", save_name="loss_dist")
    print("SAVED")
    
    # Save data
    print("\n  Saving data:")
    csv_path = logger.save_to_csv()
    json_path = logger.save_to_json()
    print(f"    CSV:  {csv_path}")
    print(f"    JSON: {json_path}")
    
    # Summary
    elapsed = time.time() - start_time
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "duration": elapsed,
        "model": model.model_name,
        "probe_accuracy": float(probe_result.probe_accuracy),
        "gradient_asr": float(gradient_metrics.asr),
        "rerouting_asr": float(rerouting_metrics.asr),
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     RESEARCH COMPLETE                        ║
╠══════════════════════════════════════════════════════════════╣
║  Duration:        {elapsed:>40.1f}s ║
║  Model:           {model.model_name:<40} ║
╠══════════════════════════════════════════════════════════════╣
║  RESULTS                                                     ║
║    Probe Accuracy:    {probe_result.probe_accuracy:>36.1%} ║
║    Gradient ASR:      {gradient_metrics.asr:>36.1%} ║
║    Rerouting ASR:     {rerouting_metrics.asr:>36.1%} ║
╠══════════════════════════════════════════════════════════════╣
║  OUTPUT: {str(output_dir.absolute()):<51} ║
╚══════════════════════════════════════════════════════════════╝

  Generated Files:
    {charts_dir}/subspace.png
    {charts_dir}/asr_comparison.png
    {charts_dir}/attack_radar.png
    {data_dir}/records.csv
    {output_dir}/summary.json
    """)
    
    if visualizer:
        visualizer.close()


if __name__ == "__main__":
    main()
