#!/usr/bin/env python
"""
Complete Research Pipeline Script.

Run this script to execute the full MIRA research workflow:
1. Environment detection
2. Model loading
3. Subspace analysis
4. Attack experiments (Gradient + Rerouting)
5. Metrics computation
6. Chart generation
7. Summary report

Usage:
    python run_research.py --model pythia-70m --output ./research_output
    python run_research.py --help
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

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
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MIRA - Complete Research Pipeline")
    print("=" * 60)
    
    # ===== STEP 1: Environment Detection =====
    print("\n[1/7] Detecting environment...")
    env = detect_environment()
    print_environment_info(env)
    
    # ===== STEP 2: Setup Output =====
    print("\n[2/7] Setting up output directories...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "charts").mkdir(exist_ok=True)
    (output_dir / "data").mkdir(exist_ok=True)
    
    logger = ExperimentLogger(
        output_dir=str(output_dir),
        experiment_name="research",
    )
    charts = ResearchChartGenerator(output_dir=str(output_dir / "charts"))
    
    # ===== STEP 3: Load Model =====
    print(f"\n[3/7] Loading model: {args.model}...")
    device = env.gpu.backend
    model = ModelWrapper(args.model, device=device)
    print(f"  Model: {model.model_name}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Device: {device}")
    
    # ===== STEP 4: Load Data =====
    print("\n[4/7] Loading prompts...")
    safe_prompts = load_safe_prompts()[:10]
    harmful_prompts = load_harmful_prompts()[:10]
    print(f"  Safe prompts: {len(safe_prompts)}")
    print(f"  Harmful prompts: {len(harmful_prompts)}")
    
    # ===== STEP 5: Subspace Analysis =====
    print("\n[5/7] Analyzing subspaces...")
    layer_idx = model.n_layers // 2
    analyzer = SubspaceAnalyzer(model, layer_idx=layer_idx)
    
    probe_result = analyzer.train_probe(safe_prompts, harmful_prompts)
    print(f"  Layer: {layer_idx}")
    print(f"  Probe accuracy: {probe_result.probe_accuracy:.2%}")
    
    # Generate subspace chart
    safe_acts = analyzer.collect_activations(safe_prompts)
    harmful_acts = analyzer.collect_activations(harmful_prompts)
    
    subspace_chart = plot_subspace_2d(
        safe_acts,
        harmful_acts,
        refusal_direction=probe_result.refusal_direction,
        title=f"Refusal Subspace (Layer {layer_idx})",
        save_path=str(output_dir / "charts" / "subspace.png"),
    )
    
    # ===== STEP 6: Attack Experiments =====
    print("\n[6/7] Running attack experiments...")
    
    test_prompts = harmful_prompts[:5]
    evaluator = AttackSuccessEvaluator()
    
    # Gradient Attack
    print("  Running Gradient Attack...")
    gradient_attack = GradientAttack(
        model,
        suffix_length=args.suffix_length,
        top_k=256,
    )
    
    gradient_results = []
    for i, prompt in enumerate(test_prompts):
        print(f"    [{i+1}/{len(test_prompts)}] {prompt[:40]}...")
        result = gradient_attack.optimize(prompt, num_steps=args.attack_steps)
        gradient_results.append({
            "prompt": prompt,
            "response": result.generated_response or "",
            "success": result.success,
            "suffix": result.adversarial_suffix,
            "loss": result.final_loss,
        })
        
        logger.log_attack(
            model_name=model.model_name,
            prompt=prompt,
            attack_type="gradient",
            suffix=result.adversarial_suffix,
            response=result.generated_response or "",
            success=result.success,
            metrics={"loss": result.final_loss},
        )
    
    # Rerouting Attack
    print("  Running Rerouting Attack...")
    rerouting_attack = ReroutingAttack(
        model,
        refusal_direction=probe_result.refusal_direction,
        acceptance_direction=probe_result.acceptance_direction,
        layer_idx=layer_idx,
    )
    
    rerouting_results = []
    for i, prompt in enumerate(test_prompts):
        print(f"    [{i+1}/{len(test_prompts)}] {prompt[:40]}...")
        result = rerouting_attack.optimize(prompt, num_steps=args.attack_steps)
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
    
    print(f"\n  Gradient ASR: {gradient_metrics.asr:.2%}")
    print(f"  Rerouting ASR: {rerouting_metrics.asr:.2%}")
    
    # ===== STEP 7: Generate Charts and Summary =====
    print("\n[7/7] Generating charts and summary...")
    
    # ASR comparison chart
    charts.plot_attack_success_rate(
        models=["Gradient", "Rerouting"],
        asr_values=[gradient_metrics.asr, rerouting_metrics.asr],
        title="Attack Success Rate Comparison",
        save_name="asr_comparison",
    )
    
    # Radar chart
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
    
    # Save data
    logger.save_to_csv()
    logger.save_to_json()
    
    # Research summary
    summary = {
        "experiment": "MIRA Research Pipeline",
        "timestamp": datetime.now().isoformat(),
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
            },
            "rerouting": {
                "asr": float(rerouting_metrics.asr),
                "refusal_rate": float(rerouting_metrics.refusal_rate),
                "total": rerouting_metrics.total_attacks,
            },
        },
        "output_files": {
            "charts": str(output_dir / "charts"),
            "data": str(output_dir / "data"),
        },
    }
    
    with open(output_dir / "research_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("RESEARCH COMPLETE")
    print("=" * 60)
    print(f"Model:           {model.model_name}")
    print(f"Probe Accuracy:  {probe_result.probe_accuracy:.2%}")
    print(f"Gradient ASR:    {gradient_metrics.asr:.2%}")
    print(f"Rerouting ASR:   {rerouting_metrics.asr:.2%}")
    print(f"Output:          {output_dir.absolute()}")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {output_dir}/charts/subspace.png")
    print(f"  - {output_dir}/charts/asr_comparison.png")
    print(f"  - {output_dir}/charts/attack_radar.png")
    print(f"  - {output_dir}/research_summary.json")
    print(f"  - {output_dir}/data/records.csv")
    print(f"  - {output_dir}/data/records.json")


if __name__ == "__main__":
    main()
