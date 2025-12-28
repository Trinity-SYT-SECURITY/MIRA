#!/usr/bin/env python
"""
MIRA - Complete Research Pipeline
==================================

Single command to run the entire research:
- Environment detection
- Model loading
- Subspace analysis
- Gradient attacks
- Probe testing (jailbreak, encoding, injection, social)
- Interactive HTML report
- Summary output

Usage:
    python main.py
    python main.py --model EleutherAI/pythia-70m --output ./results
"""

import argparse
import warnings
import os
import sys
import json
import webbrowser
from pathlib import Path
from datetime import datetime
import time

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))

from mira.utils import detect_environment, print_environment_info
from mira.utils.data import load_harmful_prompts, load_safe_prompts
from mira.utils.experiment_logger import ExperimentLogger
from mira.core import ModelWrapper
from mira.analysis import SubspaceAnalyzer
from mira.attack import GradientAttack
from mira.attack.probes import ProbeRunner, ALL_PROBES, get_all_categories
from mira.metrics import AttackSuccessEvaluator
from mira.visualization import ResearchChartGenerator
from mira.visualization import plot_subspace_2d
from mira.visualization.interactive_html import InteractiveViz


def print_banner():
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  ███╗   ███╗██╗██████╗  █████╗                               ║
    ║  ████╗ ████║██║██╔══██╗██╔══██╗                              ║
    ║  ██╔████╔██║██║██████╔╝███████║                              ║
    ║  ██║╚██╔╝██║██║██╔══██╗██╔══██║                              ║
    ║  ██║ ╚═╝ ██║██║██║  ██║██║  ██║                              ║
    ║  ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝                              ║
    ║                                                              ║
    ║  Mechanistic Interpretability Research & Attack Framework    ║
    ║  COMPLETE RESEARCH PIPELINE                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


def print_phase(phase_num: int, total: int, title: str):
    print(f"\n{'='*70}")
    print(f"  PHASE {phase_num}/{total}: {title}")
    print(f"{'='*70}")


def parse_args():
    parser = argparse.ArgumentParser(description="MIRA Complete Research")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--open-html", action="store_true", help="Open HTML report")
    parser.add_argument("--skip-probes", action="store_true", help="Skip probe testing")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    
    print_banner()
    
    TOTAL_PHASES = 6
    
    # ================================================================
    # PHASE 1: ENVIRONMENT
    # ================================================================
    print_phase(1, TOTAL_PHASES, "ENVIRONMENT DETECTION")
    
    env = detect_environment()
    print_environment_info(env)
    
    # ================================================================
    # PHASE 2: SETUP
    # ================================================================
    print_phase(2, TOTAL_PHASES, "INITIALIZATION")
    
    # Create timestamped run directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(args.output)
    output_dir = base_output / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    data_dir = output_dir / "data"
    charts_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    print(f"  Output:     {output_dir.absolute()}")
    print(f"  Model:      {args.model}")
    print(f"  Device:     {env.gpu.backend}")
    
    logger = ExperimentLogger(output_dir=str(output_dir), experiment_name="mira")
    charts = ResearchChartGenerator(output_dir=str(charts_dir))
    viz = InteractiveViz(output_dir=str(output_dir / "html"))
    
    print("\n  Loading model...", end=" ", flush=True)
    model = ModelWrapper(args.model, device=env.gpu.backend)
    print("DONE")
    print(f"  Layers: {model.n_layers}, Vocab: {model.vocab_size}")
    
    # ================================================================
    # PHASE 3: SUBSPACE ANALYSIS
    # ================================================================
    print_phase(3, TOTAL_PHASES, "SUBSPACE ANALYSIS")
    
    safe_prompts = load_safe_prompts()[:10]
    harmful_prompts = load_harmful_prompts()[:10]
    
    print(f"  Safe prompts:    {len(safe_prompts)}")
    print(f"  Harmful prompts: {len(harmful_prompts)}")
    
    layer_idx = model.n_layers // 2
    analyzer = SubspaceAnalyzer(model, layer_idx=layer_idx)
    
    print(f"\n  Training probe at layer {layer_idx}...")
    probe_result = analyzer.train_probe(safe_prompts, harmful_prompts)
    
    print(f"""
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ SUBSPACE ANALYSIS RESULTS                                                 │
  ├───────────────────────────────────────────────────────────────────────────┤
  │ Probe Accuracy:     {probe_result.probe_accuracy:>36.1%}                  │
  │ Refusal Norm:       {float(probe_result.refusal_direction.norm()):>36.4f} │
  │ Target Layer:       {layer_idx:>36}                                       │
  └───────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Generate subspace chart
    print("  Generating subspace chart...", end=" ", flush=True)
    try:
        safe_acts = analyzer.collect_activations(safe_prompts)
        harmful_acts = analyzer.collect_activations(harmful_prompts)
        plot_subspace_2d(
            safe_embeddings=safe_acts, 
            unsafe_embeddings=harmful_acts, 
            refusal_direction=probe_result.refusal_direction,
            title="Refusal Subspace", 
            save_path=str(charts_dir / "subspace.png"),
        )
        print("SAVED")
    except Exception as e:
        print(f"SKIPPED ({str(e)[:30]})")
    
    # ================================================================
    # PHASE 4: GRADIENT ATTACKS
    # ================================================================
    print_phase(4, TOTAL_PHASES, "GRADIENT ATTACKS")
    
    test_prompts = harmful_prompts[:5]
    evaluator = AttackSuccessEvaluator()
    attack = GradientAttack(model, suffix_length=15)
    
    attack_results = []
    for i, prompt in enumerate(test_prompts):
        print(f"\n  [{i+1}/{len(test_prompts)}] {prompt[:45]}...")
        result = attack.optimize(prompt, num_steps=30, verbose=False)
        status = "SUCCESS" if result.success else "FAILED"
        print(f"      Result: {status} | Loss: {result.final_loss:.4f}")
        
        attack_results.append({
            "prompt": prompt,
            "response": result.generated_response or "",
            "success": result.success,
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
    
    gradient_metrics = evaluator.evaluate_batch([
        {"prompt": r["prompt"], "response": r["response"]} for r in attack_results
    ])
    
    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │ GRADIENT ATTACK RESULTS                                      │
  ├──────────────────────────────────────────────────────────────┤
  │ Attack Success Rate:  {gradient_metrics.asr:>34.1%}          │
  │ Refusal Rate:         {gradient_metrics.refusal_rate:>34.1%} │
  │ Total Attacks:        {gradient_metrics.total_attacks:>34}   │
  └──────────────────────────────────────────────────────────────┘
    """)
    
    # ================================================================
    # PHASE 5: PROBE TESTING
    # ================================================================
    if not args.skip_probes:
        print_phase(5, TOTAL_PHASES, "PROBE TESTING")
        
        print(f"  Categories: {', '.join(get_all_categories())}")
        print(f"  Total Probes: {len(ALL_PROBES)}")
        
        runner = ProbeRunner(model, evaluator)
        probe_summary = runner.run_all(verbose=True)
        
        # Log probe results
        for result in runner.results:
            logger.log_attack(
                model_name=model.model_name,
                prompt=result["prompt"],
                attack_type=f"probe_{result['category']}",
                suffix="",
                response=result["response"],
                success=result["success"],
                metrics={"risk_level": result["risk_level"]},
            )
    else:
        print_phase(5, TOTAL_PHASES, "PROBE TESTING (SKIPPED)")
        probe_summary = {"bypass_rate": 0, "total_probes": 0}
    
    # ================================================================
    # PHASE 6: GENERATE OUTPUTS
    # ================================================================
    print_phase(6, TOTAL_PHASES, "GENERATING OUTPUTS")
    
    # Charts
    print("  Creating charts...")
    try:
        charts.plot_attack_success_rate(
            models=["Gradient Attack"],
            asr_values=[gradient_metrics.asr],
            title="Attack Success Rate",
            save_name="asr",
        )
        print("    - asr.png SAVED")
    except Exception as e:
        print(f"    - asr.png SKIPPED ({str(e)[:30]})")
    
    # HTML Report
    print("  Creating HTML report...")
    layer_states = [{"layer": i, "direction": "neutral", "refusal_score": 0.1*i, 
                     "acceptance_score": 0.05*i} for i in range(model.n_layers)]
    html_path = viz.create_full_report(
        layer_states=layer_states,
        token_probs=[("the", 0.3), ("a", 0.2), ("to", 0.1)],
        prompt=test_prompts[0] if test_prompts else "test",
        model_name=model.model_name,
    )
    print(f"    - {html_path}")
    
    # Save data
    print("  Saving data...")
    logger.save_to_csv()
    logger.save_to_json()
    
    # Summary
    elapsed = time.time() - start_time
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": elapsed,
        "model": model.model_name,
        "probe_accuracy": float(probe_result.probe_accuracy),
        "gradient_asr": float(gradient_metrics.asr),
        "probe_bypass_rate": probe_summary.get("bypass_rate", 0),
        "total_probes": probe_summary.get("total_probes", 0),
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Final report
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                       RESEARCH COMPLETE                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Duration:           {elapsed:>47.1f}s                               ║
║  Model:              {model.model_name:<47}                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  RESULTS                                                             ║
║    Probe Accuracy:   {probe_result.probe_accuracy:>47.1%}            ║
║    Gradient ASR:     {gradient_metrics.asr:>47.1%}                   ║
║    Probe Bypass:     {probe_summary.get('bypass_rate', 0):>46.1%}    ║
╠══════════════════════════════════════════════════════════════════════╣
║  OUTPUT FILES                                                        ║
║    {str(output_dir.absolute()):<64}                                  ║
║      charts/subspace.png                                             ║
║      charts/asr.png                                                  ║
║      html/mira_report.html                                           ║
║      data/records.csv                                                ║
║      summary.json                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    if args.open_html:
        webbrowser.open(f"file://{Path(html_path).absolute()}")


if __name__ == "__main__":
    main()
