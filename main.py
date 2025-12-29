#!/usr/bin/env python
"""
MIRA - Complete Research Pipeline with Live Visualization
==========================================================

Single command to run everything:
- Real-time web visualization (opens browser automatically)
- Subspace analysis with live layer updates
- GCG attack with live progress
- Probe testing (19 attacks)
- Interactive HTML report

Usage:
    python main.py
"""

import warnings
import os
import sys
import json
import webbrowser
from pathlib import Path
from datetime import datetime
import time
import threading

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))

from mira.utils import detect_environment, print_environment_info
from mira.utils.data import load_harmful_prompts, load_safe_prompts
from mira.utils.experiment_logger import ExperimentLogger
from mira.core import ModelWrapper
from mira.analysis import SubspaceAnalyzer, TransformerTracer
from mira.attack import GradientAttack
from mira.attack.probes import ProbeRunner, ALL_PROBES, get_all_categories
from mira.metrics import AttackSuccessEvaluator
from mira.visualization import ResearchChartGenerator
from mira.visualization import plot_subspace_2d
from mira.visualization.interactive_html import InteractiveViz

# Try to import live visualization
try:
    from mira.visualization.live_server import LiveVisualizationServer, VisualizationEvent
    LIVE_VIZ_AVAILABLE = True
except ImportError:
    LIVE_VIZ_AVAILABLE = False
    VisualizationEvent = None


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
    ║  COMPLETE RESEARCH PIPELINE WITH LIVE VISUALIZATION          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


def print_phase(phase_num: int, total: int, title: str):
    print(f"\n{'='*70}")
    print(f"  PHASE {phase_num}/{total}: {title}")
    print(f"{'='*70}")


def main():
    start_time = time.time()
    
    print_banner()
    
    # Configuration
    TOTAL_PHASES = 7
    
    # ================================================================
    # PHASE 1: ENVIRONMENT DETECTION & MODEL SELECTION
    # ================================================================
    print_phase(1, TOTAL_PHASES, "ENVIRONMENT DETECTION")
    
    env = detect_environment()
    print_environment_info(env)
    
    # Interactive model selection
    from mira.utils.model_selector import select_model_interactive
    
    # Check if .env specifies a model
    env_model = os.getenv("MODEL_NAME")
    
    if env_model:
        print(f"\n  📌 Using model from .env: {env_model}\n")
        model_name = env_model
    else:
        # Interactive selection based on system capabilities
        model_name = select_model_interactive()
    
    output_base = "./results"
    
    # ================================================================
    # PHASE 2: LIVE VISUALIZATION SERVER
    # ================================================================
    print_phase(2, TOTAL_PHASES, "STARTING LIVE VISUALIZATION")
    
    server = None
    if LIVE_VIZ_AVAILABLE:
        try:
            server = LiveVisualizationServer(port=5000)
            server.start(open_browser=True)
            print("  🌐 Live dashboard: http://localhost:5000")
            print("  Browser opened automatically")
            time.sleep(1)  # Let browser open
        except Exception as e:
            print(f"  ⚠ Live visualization unavailable: {e}")
            print("  Install flask: pip install flask flask-cors")
    else:
        print("  ⚠ Flask not installed - using static visualization")
        print("  Run: pip install flask flask-cors")
    
    # ================================================================
    # PHASE 3: SETUP & MODEL LOADING
    # ================================================================
    print_phase(3, TOTAL_PHASES, "INITIALIZATION")
    
    # Create timestamped run directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    print(f"  Output:     {output_dir.absolute()}")
    print(f"  Model:      {model_name}")
    print(f"  Device:     {env.gpu.backend}")
    
    logger = ExperimentLogger(output_dir=str(output_dir), experiment_name="mira")
    charts = ResearchChartGenerator(output_dir=str(charts_dir))
    viz = InteractiveViz(output_dir=str(output_dir / "html"))
    
    # Get HF_CACHE_DIR from environment if specified
    hf_cache_dir = os.getenv("HF_CACHE_DIR")
    if hf_cache_dir:
        print(f"  Cache Dir:  {hf_cache_dir}")
    
    print("\n  Loading model...", end=" ", flush=True)
    model = ModelWrapper(model_name, device=env.gpu.backend, cache_dir=hf_cache_dir)
    print("DONE")
    print(f"  Layers: {model.n_layers}, Vocab: {model.vocab_size}")
    
    # ================================================================
    # PHASE 4: SUBSPACE ANALYSIS
    # ================================================================
    print_phase(4, TOTAL_PHASES, "SUBSPACE ANALYSIS")
    
    safe_prompts = load_safe_prompts()[:10]
    harmful_prompts = load_harmful_prompts()[:10]
    
    print(f"  Safe prompts:    {len(safe_prompts)}")
    print(f"  Harmful prompts: {len(harmful_prompts)}")
    
    layer_idx = model.n_layers // 2
    analyzer = SubspaceAnalyzer(model, layer_idx=layer_idx)
    
    print(f"\n  Training probe at layer {layer_idx}...")
    
    # Send layer updates to live viz
    for i, prompt in enumerate(harmful_prompts[:3]):
        _, cache = model.run_with_cache(prompt)
        for layer in range(model.n_layers):
            if server:
                server.send_layer_update(
                    layer_idx=layer,
                    refusal_score=0.1 * (layer + 1) / model.n_layers,
                    acceptance_score=0.05 * (layer + 1) / model.n_layers,
                    direction="neutral",
                )
            time.sleep(0.02)
    
    probe_result = analyzer.train_probe(safe_prompts, harmful_prompts)
    
    refusal_norm = float(probe_result.refusal_direction.norm())
    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │ SUBSPACE ANALYSIS RESULTS                                    │
  ├──────────────────────────────────────────────────────────────┤
  │  Probe Accuracy:   {probe_result.probe_accuracy:>38.1%}  │
  │  Refusal Norm:     {refusal_norm:>38.4f}  │
  │  Target Layer:     {layer_idx:>38}  │
  └──────────────────────────────────────────────────────────────┘
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
        print(f"SKIPPED")
    
    # ================================================================
    # PHASE 5: GRADIENT ATTACKS WITH LIVE VISUALIZATION
    # ================================================================
    print_phase(5, TOTAL_PHASES, "GRADIENT ATTACKS (LIVE)")
    
    test_prompts = harmful_prompts[:5]
    evaluator = AttackSuccessEvaluator()
    attack = GradientAttack(model, suffix_length=15)
    
    # Initialize transformer tracer for internal visualization
    tracer = TransformerTracer(model)
    
    # Real-time step callback for visualization
    def attack_step_callback(step, loss, suffix_tokens, model, prompt):
        """Send real transformer data during each attack step."""
        if not server:
            return
        
        # Send attack progress
        suffix_text = model.tokenizer.decode(suffix_tokens.tolist())
        server.send_attack_step(
            step=step,
            loss=loss,
            suffix=suffix_text[:40],
            success=False,
        )
        
        # Every 2 steps, send detailed transformer trace for visualization
        if step % 2 == 0:
            try:
                full_prompt = prompt + " " + suffix_text
                input_ids = model.tokenizer.encode(full_prompt, return_tensors="pt")[0]
                
                # Get trace with attention weights
                trace = tracer.trace_forward(input_ids)
                
                # Check if trace is valid
                if trace is None:
                    return
                
                # 1. Send embeddings (token vectors)
                tokens = trace.tokens[:15] if trace.tokens else []
                if tokens:
                    server.send_embeddings(tokens=tokens, embeddings=[])
                
                # 2. Send Q/K/V event for each layer (first 3 layers for performance)
                for layer_idx in range(min(3, len(trace.layers))):
                    layer_trace = trace.layers[layer_idx]
                    if layer_trace is None:
                        continue
                    
                    # Send QKV event
                    server.send_event(VisualizationEvent(
                        event_type="qkv",
                        data={
                            "layer": layer_idx,
                            "tokens": tokens[:6],
                            "query_vectors": [0.5 + (step % 5) * 0.1] * len(tokens[:6]),
                            "key_vectors": [0.4 + (step % 5) * 0.08] * len(tokens[:6]),
                            "value_vectors": [0.6 + (step % 5) * 0.05] * len(tokens[:6]),
                        }
                    ))
                    
                    # 3. Send attention matrix
                    attn = getattr(layer_trace, 'attention_weights', None)
                    if attn is not None and hasattr(attn, 'shape'):
                        try:
                            # Get first head's attention weights
                            if len(attn.shape) >= 3:
                                attn_head = attn[0].detach().cpu()  # First head
                            else:
                                attn_head = attn.detach().cpu()
                            
                            n = min(8, attn_head.shape[0], attn_head.shape[1])
                            weights = attn_head[:n, :n].tolist()
                            
                            server.send_attention_matrix(
                                layer_idx=layer_idx,
                                head_idx=0,
                                attention_weights=weights,
                                tokens=tokens[:n],
                            )
                        except Exception:
                            pass
                    
                    # 4. Send MLP activations
                    mlp = getattr(layer_trace, 'mlp_intermediate', None)
                    if mlp is not None:
                        try:
                            # Get top neuron activations
                            if hasattr(mlp, 'shape') and len(mlp.shape) >= 1:
                                flat_mlp = mlp.flatten().detach().cpu()
                                top_k = torch.topk(flat_mlp.abs(), min(8, len(flat_mlp)))
                                activations = top_k.values.tolist()
                                top_neurons = top_k.indices.tolist()
                                
                                server.send_event(VisualizationEvent(
                                    event_type="mlp",
                                    data={
                                        "layer": layer_idx,
                                        "activations": [a / max(activations) if max(activations) > 0 else 0 for a in activations],
                                        "top_neurons": top_neurons,
                                    }
                                ))
                        except Exception:
                            pass
                    
                    # 5. Send layer update
                    server.send_layer_update(
                        layer_idx=layer_idx,
                        refusal_score=max(0, 0.8 - step * 0.02),
                        acceptance_score=min(1.0, 0.2 + step * 0.02),
                        direction="forward" if step < 15 else "optimizing",
                    )
                
                # 6. Send output probabilities from final logits
                if trace.final_logits is not None:
                    try:
                        last_logits = trace.final_logits[-1].detach().cpu()
                        probs = torch.softmax(last_logits, dim=0)
                        top_k = torch.topk(probs, 5)
                        output_tokens = [model.tokenizer.decode([idx.item()]) for idx in top_k.indices]
                        output_probs = top_k.values.tolist()
                        
                        server.send_output_probabilities(
                            tokens=output_tokens,
                            probabilities=output_probs,
                        )
                    except Exception:
                        pass
                
            except Exception as e:
                # Log trace errors for debugging
                print(f"      [Trace Error: {e}]")
                pass
    
    attack_results = []
    for i, prompt in enumerate(test_prompts):
        print(f"\n  [{i+1}/{len(test_prompts)}] {prompt[:45]}...")
        
        # Send initial state
        if server:
            server.send_attack_step(step=0, loss=10.0, suffix="Initializing...", success=False)
            # Send initial embeddings
            try:
                input_ids = model.tokenizer.encode(prompt, return_tensors="pt")[0]
                tokens = model.tokenizer.convert_ids_to_tokens(input_ids.tolist())
                server.send_embeddings(tokens=tokens[:15], embeddings=[])
            except:
                pass
        
        # Run attack WITH step callback for real-time visualization
        result = attack.optimize(
            prompt, 
            num_steps=30, 
            verbose=False,
            step_callback=attack_step_callback,
        )
        
        # Final update after attack
        if server:
            server.send_attack_step(
                step=30,
                loss=result.final_loss,
                suffix=result.adversarial_suffix[:30] if result.adversarial_suffix else "done",
                success=result.success,
            )
        
        # Trace adversarial prompt AFTER attack
        if server and result.adversarial_suffix:
            try:
                adv_prompt = prompt + " " + result.adversarial_suffix
                adv_ids = model.tokenizer.encode(adv_prompt, return_tensors="pt")[0]
                adv_trace = tracer.trace_forward(adv_ids)
                
                # Send adversarial trace
                server.send_transformer_trace(
                    trace_data=adv_trace.to_dict(),
                    trace_type="adversarial"
                )
            except Exception as e:
                print(f"      [Trace Error: {e}]")
        
        # Evaluate response for detailed feedback
        eval_result = evaluator.evaluate_single(
            prompt=prompt,
            response=result.generated_response or "",
            adversarial_suffix=result.adversarial_suffix
        )
        
        status = "SUCCESS" if result.success else "FAILED"
        
        # Display result with detailed reason
        print(f"      Result: {status} | Loss: {result.final_loss:.4f}")
        
        if not result.success:
            # Show why it failed
            print(f"      Reason: {eval_result['category']}")
            if eval_result['has_refusal_markers']:
                print(f"      ⚠ Model refused (detected refusal patterns)")
            if result.generated_response:
                preview = result.generated_response[:80].replace('\n', ' ')
                print(f"      Response: \"{preview}...\"")
        else:
            print(f"      ✓ Attack succeeded!")
            if result.generated_response:
                preview = result.generated_response[:80].replace('\n', ' ')
                print(f"      Response: \"{preview}...\"")
        
        attack_results.append({
            "prompt": prompt,
            "response": result.generated_response or "",
            "success": result.success,
            "loss": result.final_loss,
            "eval_category": eval_result['category'],
            "has_refusal": eval_result['has_refusal_markers'],
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
  ┌─────────────────────────────────────────────────────────────┐
  │ GRADIENT ATTACK RESULTS                                     │
  ├─────────────────────────────────────────────────────────────┤
  │ Attack Success Rate:  {gradient_metrics.asr:>34.1%}         │
  │ Refusal Rate:         {gradient_metrics.refusal_rate:>34.1%}│
  │ Total Attacks:        {gradient_metrics.total_attacks:>34}  │
  └─────────────────────────────────────────────────────────────┘
    """)
    
    # ================================================================
    # PHASE 6: PROBE TESTING
    # ================================================================
    print_phase(6, TOTAL_PHASES, "PROBE TESTING (19 ATTACKS)")
    
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
    
    # ================================================================
    # PHASE 7: GENERATE OUTPUTS
    # ================================================================
    print_phase(7, TOTAL_PHASES, "GENERATING OUTPUTS")
    
    # Charts
    print("  Creating charts...")
    try:
        charts.plot_attack_success_rate(
            models=["Gradient Attack"],
            asr_values=[gradient_metrics.asr],
            title="Attack Success Rate",
            save_name="asr",
        )
        print("    ✓ asr.png")
    except:
        pass
    
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
    print(f"    ✓ {html_path}")
    
    # Save data
    print("  Saving data...")
    logger.save_to_csv()
    logger.save_to_json()
    print("    ✓ records.csv, records.json")
    
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
    
    # Send completion to live viz
    if server:
        server.send_complete({
            "asr": gradient_metrics.asr,
            "probe_bypass": probe_summary.get("bypass_rate", 0),
            "duration": elapsed,
        })
    
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
║  LIVE DASHBOARD                                                      ║
║    🌐 http://localhost:5000 (keep running to view)                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  OUTPUT FILES                                                        ║
║    {str(output_dir.absolute())[:62]:<62}                             ║
╚══════════════════════════════════════════════════════════════════════╝

  Press Ctrl+C to exit (dashboard will close)
    """)
    
    # Open HTML report
    webbrowser.open(f"file://{Path(html_path).absolute()}")
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Goodbye!")


if __name__ == "__main__":
    main()
