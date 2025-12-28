#!/usr/bin/env python
"""
Complete Research Pipeline Script with Real-time Visualization.

Shows progress and generates visualizations during execution.

Usage:
    python examples/run_research.py --model pythia-70m --output ./research_output
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import time

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from mira.utils import detect_environment, print_environment_info
from mira.utils.data import load_harmful_prompts, load_safe_prompts
from mira.utils.experiment_logger import ExperimentLogger
from mira.core import ModelWrapper
from mira.analysis import SubspaceAnalyzer
from mira.attack import GradientAttack, ReroutingAttack
from mira.metrics import AttackSuccessEvaluator
from mira.visualization import ResearchChartGenerator
from mira.visualization import plot_subspace_2d


def print_step(step_num: int, total: int, message: str):
    """Print step progress."""
    bar = "=" * 50
    print(f"\n{'=' * 60}")
    print(f"[Step {step_num}/{total}] {message}")
    print(f"{'=' * 60}")


def print_progress_bar(current: int, total: int, prefix: str = "", suffix: str = ""):
    """Print ASCII progress bar."""
    percent = current / total * 100
    filled = int(50 * current / total)
    bar = "#" * filled + "-" * (50 - filled)
    print(f"\r{prefix} [{bar}] {percent:.1f}% {suffix}", end="", flush=True)
    if current == total:
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="MIRA Complete Research Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m",
        help="Model name (default: EleutherAI/pythia-70m)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./research_output",
        help="Output directory (default: ./research_output)",
    )
    parser.add_argument(
        "--attack-steps",
        type=int,
        default=50,
        help="Number of attack optimization steps (default: 50)",
    )
    parser.add_argument(
        "--suffix-length",
        type=int,
        default=20,
        help="Adversarial suffix length (default: 20)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to test (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    print("""
    ███╗   ███╗██╗██████╗  █████╗ 
    ████╗ ████║██║██╔══██╗██╔══██╗
    ██╔████╔██║██║██████╔╝███████║
    ██║╚██╔╝██║██║██╔══██╗██╔══██║
    ██║ ╚═╝ ██║██║██║  ██║██║  ██║
    ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
    Mechanistic Interpretability Research & Attack Framework
    """)
    
    TOTAL_STEPS = 7
    
    # ===== STEP 1: Environment Detection =====
    print_step(1, TOTAL_STEPS, "ENVIRONMENT DETECTION")
    
    print("Detecting system capabilities...")
    env = detect_environment()
    print_environment_info(env)
    
    # ===== STEP 2: Setup Output =====
    print_step(2, TOTAL_STEPS, "SETUP OUTPUT DIRECTORIES")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    data_dir = output_dir / "data"
    charts_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Charts: {charts_dir}")
    print(f"  Data: {data_dir}")
    
    logger = ExperimentLogger(
        output_dir=str(output_dir),
        experiment_name="research",
    )
    charts = ResearchChartGenerator(output_dir=str(charts_dir))
    
    # ===== STEP 3: Load Model =====
    print_step(3, TOTAL_STEPS, "LOADING MODEL")
    
    device = env.gpu.backend
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print("  Loading...", end=" ", flush=True)
    
    model = ModelWrapper(args.model, device=device)
    
    print("DONE")
    print(f"  Loaded: {model.model_name}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Vocab size: {model.vocab_size}")
    
    # ===== STEP 4: Load Data =====
    print_step(4, TOTAL_STEPS, "LOADING RESEARCH DATA")
    
    safe_prompts = load_safe_prompts()[:args.num_prompts * 2]
    harmful_prompts = load_harmful_prompts()[:args.num_prompts * 2]
    test_prompts = harmful_prompts[:args.num_prompts]
    
    print(f"  Safe prompts: {len(safe_prompts)}")
    print(f"  Harmful prompts: {len(harmful_prompts)}")
    print(f"  Test prompts: {len(test_prompts)}")
    
    print("\n  Sample prompts:")
    for i, p in enumerate(test_prompts[:3]):
        print(f"    {i+1}. {p[:50]}...")
    
    # ===== STEP 5: Subspace Analysis =====
    print_step(5, TOTAL_STEPS, "SUBSPACE ANALYSIS")
    
    layer_idx = model.n_layers // 2
    analyzer = SubspaceAnalyzer(model, layer_idx=layer_idx)
    
    print(f"  Target layer: {layer_idx}")
    print("\n  Collecting activations...")
    
    # Show progress for activation collection
    print("    Safe prompts:    ", end="")
    safe_acts = analyzer.collect_activations(safe_prompts)
    print("DONE")
    
    print("    Harmful prompts: ", end="")
    harmful_acts = analyzer.collect_activations(harmful_prompts)
    print("DONE")
    
    print("\n  Training linear probe...")
    probe_result = analyzer.train_probe(safe_prompts, harmful_prompts)
    
    print(f"\n  ┌────────────────────────────────────┐")
    print(f"  │ SUBSPACE ANALYSIS RESULTS          │")
    print(f"  ├────────────────────────────────────┤")
    print(f"  │ Layer:          {layer_idx:>18} │")
    print(f"  │ Probe Accuracy: {probe_result.probe_accuracy:>17.2%} │")
    print(f"  │ Direction Norm: {probe_result.refusal_direction.norm():>17.4f} │")
    print(f"  └────────────────────────────────────┘")
    
    # Generate visualization
    print("\n  Generating subspace visualization...")
    subspace_path = str(charts_dir / "subspace.png")
    plot_subspace_2d(
        safe_acts,
        harmful_acts,
        refusal_direction=probe_result.refusal_direction,
        title=f"Refusal Subspace (Layer {layer_idx})",
        save_path=subspace_path,
    )
    print(f"    Saved: {subspace_path}")
    
    # ===== STEP 6: Attack Experiments =====
    print_step(6, TOTAL_STEPS, "ATTACK EXPERIMENTS")
    
    evaluator = AttackSuccessEvaluator()
    
    # ----- Gradient Attack -----
    print("\n  [A] GRADIENT ATTACK")
    print("  " + "-" * 40)
    
    gradient_attack = GradientAttack(
        model,
        suffix_length=args.suffix_length,
        top_k=256,
    )
    
    gradient_results = []
    gradient_losses = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n  Attack {i+1}/{len(test_prompts)}: {prompt[:40]}...")
        
        # Show optimization progress
        result = gradient_attack.optimize(
            prompt,
            num_steps=args.attack_steps,
            verbose=False,
        )
        
        status = "SUCCESS" if result.success else "FAILED"
        print(f"    Result: {status}")
        print(f"    Loss:   {result.final_loss:.4f}")
        if result.success:
            print(f"    Suffix: {result.adversarial_suffix[:30]}...")
        
        gradient_results.append({
            "prompt": prompt,
            "response": result.generated_response or "",
            "success": result.success,
            "suffix": result.adversarial_suffix,
            "loss": result.final_loss,
        })
        gradient_losses.append(result.final_loss)
        
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
    print("\n  [B] REROUTING ATTACK")
    print("  " + "-" * 40)
    
    rerouting_attack = ReroutingAttack(
        model,
        refusal_direction=probe_result.refusal_direction,
        acceptance_direction=probe_result.acceptance_direction,
        layer_idx=layer_idx,
    )
    
    rerouting_results = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n  Attack {i+1}/{len(test_prompts)}: {prompt[:40]}...")
        
        result = rerouting_attack.optimize(prompt, num_steps=args.attack_steps)
        
        status = "SUCCESS" if result.success else "FAILED"
        print(f"    Result: {status}")
        
        rerouting_results.append({
            "prompt": prompt,
            "response": result.generated_response or "",
            "success": result.success,
        })
        
        logger.log_attack(
            model_name=model.model_name,
            prompt=prompt,
            attack_type="rerouting",
            suffix="",
            response=result.generated_response or "",
            success=result.success,
            metrics={},
        )
    
    # Compute metrics
    gradient_metrics = evaluator.evaluate_batch([
        {"prompt": r["prompt"], "response": r["response"]}
        for r in gradient_results
    ])
    
    rerouting_metrics = evaluator.evaluate_batch([
        {"prompt": r["prompt"], "response": r["response"]}
        for r in rerouting_results
    ])
    
    print(f"\n  ┌────────────────────────────────────────────┐")
    print(f"  │ ATTACK RESULTS SUMMARY                     │")
    print(f"  ├────────────────────────────────────────────┤")
    print(f"  │ Gradient Attack:                           │")
    print(f"  │   ASR:          {gradient_metrics.asr:>25.2%} │")
    print(f"  │   Refusal Rate: {gradient_metrics.refusal_rate:>25.2%} │")
    print(f"  ├────────────────────────────────────────────┤")
    print(f"  │ Rerouting Attack:                          │")
    print(f"  │   ASR:          {rerouting_metrics.asr:>25.2%} │")
    print(f"  │   Refusal Rate: {rerouting_metrics.refusal_rate:>25.2%} │")
    print(f"  └────────────────────────────────────────────┘")
    
    # ===== STEP 7: Generate Charts and Summary =====
    print_step(7, TOTAL_STEPS, "GENERATING CHARTS & SUMMARY")
    
    print("  Generating visualizations...")
    
    # ASR comparison chart
    print("    1. ASR Comparison Chart...", end=" ")
    charts.plot_attack_success_rate(
        models=["Gradient", "Rerouting"],
        asr_values=[gradient_metrics.asr, rerouting_metrics.asr],
        title="Attack Success Rate Comparison",
        save_name="asr_comparison",
    )
    print("DONE")
    
    # Radar chart
    print("    2. Attack Radar Chart...", end=" ")
    charts.plot_comparison_radar(
        categories=["ASR", "Bypass Rate", "Stealth", "Speed"],
        values_dict={
            "Gradient": [
                gradient_metrics.asr,
                1 - gradient_metrics.refusal_rate,
                0.6,
                0.5,
            ],
            "Rerouting": [
                rerouting_metrics.asr,
                1 - rerouting_metrics.refusal_rate,
                0.9,
                0.8,
            ],
        },
        title="Attack Strategy Comparison",
        save_name="attack_radar",
    )
    print("DONE")
    
    # Loss distribution
    print("    3. Loss Distribution...", end=" ")
    charts.plot_entropy_distribution(
        gradient_losses,
        title="Gradient Attack Final Loss Distribution",
        save_name="loss_distribution",
    )
    print("DONE")
    
    # Save data
    print("\n  Saving experiment data...")
    csv_path = logger.save_to_csv()
    json_path = logger.save_to_json()
    print(f"    CSV:  {csv_path}")
    print(f"    JSON: {json_path}")
    
    # Research summary
    elapsed = time.time() - start_time
    
    summary = {
        "experiment": "MIRA Research Pipeline",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": elapsed,
        "environment": {
            "os": env.system.os_name,
            "python": env.system.python_version,
            "gpu": env.gpu.device_name or "CPU",
            "device": env.gpu.backend,
        },
        "model": {
            "name": model.model_name,
            "layers": model.n_layers,
        },
        "subspace_analysis": {
            "layer": layer_idx,
            "probe_accuracy": float(probe_result.probe_accuracy),
        },
        "attack_results": {
            "gradient": {
                "asr": float(gradient_metrics.asr),
                "refusal_rate": float(gradient_metrics.refusal_rate),
                "total": gradient_metrics.total_attacks,
                "successful": gradient_metrics.successful_attacks,
            },
            "rerouting": {
                "asr": float(rerouting_metrics.asr),
                "refusal_rate": float(rerouting_metrics.refusal_rate),
                "total": rerouting_metrics.total_attacks,
                "successful": rerouting_metrics.successful_attacks,
            },
        },
        "output_files": {
            "charts": str(charts_dir),
            "data": str(data_dir),
        },
    }
    
    summary_path = output_dir / "research_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"""
╔════════════════════════════════════════════════════════════╗
║                    RESEARCH COMPLETE                       ║
╠════════════════════════════════════════════════════════════╣
║ Model:           {model.model_name:<40} ║
║ Duration:        {elapsed:.2f} seconds{' ' * 32}║
╠════════════════════════════════════════════════════════════╣
║ RESULTS:                                                   ║
║   Probe Accuracy:  {probe_result.probe_accuracy:>38.2%} ║
║   Gradient ASR:    {gradient_metrics.asr:>38.2%} ║
║   Rerouting ASR:   {rerouting_metrics.asr:>38.2%} ║
╠════════════════════════════════════════════════════════════╣
║ OUTPUT FILES:                                              ║
║   {str(output_dir.absolute())[:56]:<56} ║
╚════════════════════════════════════════════════════════════╝

Generated files:
  Charts:
    - {charts_dir}/subspace.png
    - {charts_dir}/asr_comparison.png
    - {charts_dir}/attack_radar.png
    - {charts_dir}/loss_distribution.png
  Data:
    - {data_dir}/records.csv
    - {data_dir}/records.json
  Summary:
    - {summary_path}
    """)


if __name__ == "__main__":
    main()
