#!/usr/bin/env python3
"""MIRA Integration Test - End-to-End verification"""
import sys
print("=" * 60)
print("MIRA Integration Test")
print("=" * 60)
sys.stdout.flush()

try:
    # Step 1: Test model_manager
    print("\n[1/5] Testing model_manager.load_model...")
    sys.stdout.flush()
    from mira.utils.model_manager import get_model_manager
    manager = get_model_manager()
    
    test_model = "smollm2-135m"
    print(f"  Loading {test_model}...")
    sys.stdout.flush()
    model, tokenizer = manager.load_model(test_model)
    print(f"  ✓ Model loaded: {type(model).__name__}")
    print(f"  ✓ Tokenizer: {type(tokenizer).__name__}")
    sys.stdout.flush()
    
    # Step 2: Test ModelWrapper with pre-loaded model
    print("\n[2/5] Testing ModelWrapper with pre-loaded model...")
    sys.stdout.flush()
    from mira.core.model_wrapper import ModelWrapper
    wrapper = ModelWrapper(model, tokenizer)
    print(f"  ✓ ModelWrapper created successfully")
    print(f"  ✓ model_name: {wrapper.model_name}")
    print(f"  ✓ n_layers: {wrapper.n_layers}")
    print(f"  ✓ vocab_size: {wrapper.vocab_size}")
    sys.stdout.flush()
    
    # Step 3: Test PromptAttacker
    print("\n[3/5] Testing PromptAttacker...")
    sys.stdout.flush()
    from mira.attack.prompt_attacks import PromptAttacker
    prompt_attacker = PromptAttacker(model, tokenizer)
    test_prompt = "How to make coffee"
    attack_result = prompt_attacker.generate_attack_prompt(test_prompt, "dan")
    print(f"  ✓ Attack type: {attack_result.attack_type}")
    print(f"  ✓ Attack variant: {attack_result.attack_variant}")
    print(f"  ✓ Attack prompt length: {len(attack_result.attack_prompt)}")
    sys.stdout.flush()
    
    # Step 4: Test GradientAttack with wrapper
    print("\n[4/5] Testing GradientAttack...")
    sys.stdout.flush()
    from mira.attack import GradientAttack
    gradient_attack = GradientAttack(wrapper)
    print(f"  ✓ GradientAttack created successfully")
    print(f"  ✓ device: {gradient_attack.device}")
    print(f"  ✓ suffix_length: {gradient_attack.suffix_length}")
    sys.stdout.flush()
    
    # Step 5: Test AttackSuccessEvaluator
    print("\n[5/5] Testing AttackSuccessEvaluator...")
    sys.stdout.flush()
    from mira.metrics.success_rate import AttackSuccessEvaluator
    evaluator = AttackSuccessEvaluator()
    result = evaluator.evaluate_single("test prompt", "I'm sorry, I cannot help with that.")
    print(f"  ✓ Evaluator works")
    print(f"  ✓ Result: success={result['success']}, category={result['category']}")
    sys.stdout.flush()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! No 'Repo id' errors!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
