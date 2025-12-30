"""
Mechanistic Analysis Mode

Deep analysis of model internals using Logit Lens, Uncertainty Analysis, and Activation Hooks.
"""


def run_mechanistic_analysis():
    """Run mechanistic analysis tools mode."""
    from mira.utils.model_manager import get_model_manager
    
    print("\n" + "="*70)
    print("  MECHANISTIC ANALYSIS TOOLS")
    print("="*70 + "\n")
    
    # Select model
    manager = get_model_manager()
    downloaded = manager.list_downloaded_models()
    
    if downloaded:
        print("  Downloaded models:")
        for i, m in enumerate(downloaded):
            print(f"    [{i+1}] {m}")
        print()
    
    model_name = input("  Model name (default: gpt2): ").strip()
    if not model_name:
        model_name = "gpt2"
    
    # Select analysis type
    print("""
  Analysis Types:
    [1] Logit Lens - Track prediction evolution across layers
    [2] Uncertainty Analysis - Entropy, confidence, risk detection
    [3] Activation Hooks - Capture internal activations
    """)
    
    analysis_type = input("  Select analysis (1-3, default: 1): ").strip()
    if not analysis_type:
        analysis_type = "1"
    
    # Get prompt
    prompt = input("  Input prompt (default: 'Hello, how are you?'): ").strip()
    if not prompt:
        prompt = "Hello, how are you?"
    
    print(f"\n  Loading model: {model_name}...")
    
    # Load model using model manager
    model, tokenizer = manager.load_model(model_name)
    device = next(model.parameters()).device
    
    print(f"  Model loaded on {device}\n")
    
    if analysis_type == "1":
        # Logit Lens
        from mira.analysis.logit_lens import run_logit_lens_analysis, LogitLensVisualizer, LogitProjector
        
        print("  Running Logit Lens analysis...")
        trajectory = run_logit_lens_analysis(model, tokenizer, prompt)
        
        visualizer = LogitLensVisualizer(LogitProjector(model, tokenizer))
        print("\n" + visualizer.format_trajectory_table(trajectory))
        
    elif analysis_type == "2":
        # Uncertainty Analysis
        from mira.analysis.uncertainty import analyze_generation_uncertainty
        
        print("  Analyzing generation uncertainty...")
        result = analyze_generation_uncertainty(model, tokenizer, prompt, max_tokens=50)
        
        print(f"\n  Risk Level: {result['risk']['risk_level']}")
        print(f"  Risk Score: {result['risk']['risk_score']:.2f}")
        print(f"  Mean Entropy: {result['metrics']['mean_entropy']:.2f}")
        print(f"  Min Confidence: {result['metrics']['min_confidence']:.4f}")
        print(f"  Entropy Spikes: {len(result['risk']['entropy_spikes'])}")
        
    elif analysis_type == "3":
        # Activation Hooks
        from mira.core.hooks import ActivationHookManager
        
        print("  Capturing activations...")
        hook_manager = ActivationHookManager(model)
        hook_manager.register_all_layers(["residual"])
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        logits, cache = hook_manager.run_with_cache(input_ids)
        
        print(f"\n  Captured {len(cache.residual)} layer activations:")
        for layer_idx, act in cache.residual.items():
            print(f"    Layer {layer_idx}: shape {list(act.shape)}")
    
    print("\n" + "="*70)
