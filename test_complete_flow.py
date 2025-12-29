"""
Complete Flow Test Suite for MIRA Visualization

Tests all components before running main.py to catch errors early.
Run: python test_complete_flow.py
"""

import sys
import os
import traceback

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("  MIRA COMPLETE FLOW TEST SUITE")
print("=" * 70)
print()

all_tests_passed = True
test_results = []


def run_test(name: str, test_func):
    """Run a test and record result."""
    global all_tests_passed
    print(f"\n{'─' * 50}")
    print(f"TEST: {name}")
    print("─" * 50)
    
    try:
        result = test_func()
        if result:
            print(f"  ✅ PASSED: {name}")
            test_results.append((name, True, None))
            return True
        else:
            print(f"  ❌ FAILED: {name}")
            test_results.append((name, False, "Test returned False"))
            all_tests_passed = False
            return False
    except Exception as e:
        print(f"  ❌ ERROR: {name}")
        print(f"     {type(e).__name__}: {e}")
        traceback.print_exc()
        test_results.append((name, False, str(e)))
        all_tests_passed = False
        return False


# ============================================================
# TEST 1: Core Imports
# ============================================================
def test_core_imports():
    """Test core MIRA module imports."""
    from mira.core import ModelWrapper
    print("  ✓ mira.core.ModelWrapper")
    
    from mira.utils.environment import detect_environment
    print("  ✓ mira.utils.environment.detect_environment")
    
    return True

run_test("Core Imports", test_core_imports)


# ============================================================
# TEST 2: Analysis Imports
# ============================================================
def test_analysis_imports():
    """Test analysis module imports."""
    from mira.analysis.subspace_probe import SubspaceProbe
    print("  ✓ mira.analysis.subspace_probe.SubspaceProbe")
    
    from mira.analysis.evaluator import AttackSuccessEvaluator
    print("  ✓ mira.analysis.evaluator.AttackSuccessEvaluator")
    
    from mira.analysis.transformer_tracer import TransformerTracer
    print("  ✓ mira.analysis.transformer_tracer.TransformerTracer")
    
    return True

run_test("Analysis Imports", test_analysis_imports)


# ============================================================
# TEST 3: Attack Imports
# ============================================================
def test_attack_imports():
    """Test attack module imports."""
    from mira.attack.gradient import GradientAttack
    print("  ✓ mira.attack.gradient.GradientAttack")
    
    from mira.attack.base import BaseAttack
    print("  ✓ mira.attack.base.BaseAttack")
    
    return True

run_test("Attack Imports", test_attack_imports)


# ============================================================
# TEST 4: Visualization Imports
# ============================================================
def test_visualization_imports():
    """Test visualization module imports."""
    from mira.visualization.live_server import LiveVisualizationServer
    print("  ✓ mira.visualization.live_server.LiveVisualizationServer")
    
    from mira.visualization.transformer_flow import get_transformer_flow_html
    print("  ✓ mira.visualization.transformer_flow.get_transformer_flow_html")
    
    from mira.visualization.simple_dashboard import get_simple_dashboard
    print("  ✓ mira.visualization.simple_dashboard.get_simple_dashboard")
    
    return True

run_test("Visualization Imports", test_visualization_imports)


# ============================================================
# TEST 5: Transformer Flow Dashboard
# ============================================================
def test_transformer_flow_dashboard():
    """Test transformer flow dashboard HTML."""
    from mira.visualization.transformer_flow import get_transformer_flow_html
    
    html = get_transformer_flow_html()
    
    # Check HTML structure
    assert len(html) > 5000, f"HTML too short: {len(html)} chars"
    print(f"  ✓ HTML length: {len(html)} chars")
    
    assert "<html" in html, "Missing <html> tag"
    print("  ✓ HTML structure valid")
    
    assert "EventSource" in html, "Missing SSE EventSource"
    print("  ✓ SSE EventSource present")
    
    assert "flow-container" in html, "Missing flow container"
    print("  ✓ Flow container present")
    
    assert "stage-embedding" in html, "Missing embedding stage"
    assert "stage-qkv" in html, "Missing QKV stage"
    assert "stage-attention" in html, "Missing attention stage"
    assert "stage-mlp" in html, "Missing MLP stage"
    print("  ✓ All processing stages present")
    
    assert "attention-matrix" in html, "Missing attention matrix"
    print("  ✓ Attention matrix present")
    
    assert "handleAttackStep" in html, "Missing attack step handler"
    assert "handleEmbeddings" in html, "Missing embeddings handler"
    assert "handleAttention" in html, "Missing attention handler"
    print("  ✓ Event handlers present")
    
    return True

run_test("Transformer Flow Dashboard", test_transformer_flow_dashboard)


# ============================================================
# TEST 6: Live Server Methods
# ============================================================
def test_live_server_methods():
    """Test all LiveVisualizationServer methods exist with correct signatures."""
    from mira.visualization.live_server import LiveVisualizationServer
    import inspect
    
    # Only test methods that are actually used
    required_methods = {
        'send_event': ['event'],
        'send_layer_update': ['layer_idx', 'refusal_score', 'acceptance_score', 'direction'],
        'send_attack_step': ['step', 'loss', 'suffix', 'success'],
        'send_attention_update': ['layer_idx', 'head_idx', 'attention_weights', 'tokens'],
        'send_embeddings': ['tokens', 'embeddings'],
        'send_transformer_trace': ['trace_data', 'trace_type'],
        'send_attention_matrix': ['layer_idx', 'head_idx', 'attention_weights', 'tokens'],
        'send_residual_update': ['layer_idx', 'residual_norm', 'delta_norm'],  # Actual params
        'send_complete': ['summary'],  # Actual param
    }
    
    for method_name, required_params in required_methods.items():
        method = getattr(LiveVisualizationServer, method_name, None)
        assert method is not None, f"Method {method_name} not found"
        
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        
        for param in required_params:
            assert param in params, f"Method {method_name} missing param: {param}"
        
        print(f"  ✓ {method_name}({', '.join(required_params)})")
    
    return True

run_test("Live Server Methods", test_live_server_methods)


# ============================================================
# TEST 7: Server Method Calls
# ============================================================
def test_server_calls():
    """Test that server method calls work without errors."""
    from mira.visualization.live_server import LiveVisualizationServer
    
    # Test send_layer_update
    LiveVisualizationServer.send_layer_update(
        layer_idx=0,
        refusal_score=0.5,
        acceptance_score=0.5,
        direction="forward"
    )
    print("  ✓ send_layer_update OK")
    
    # Test send_attack_step
    LiveVisualizationServer.send_attack_step(
        step=1,
        loss=5.0,
        suffix="test suffix",
        success=False
    )
    print("  ✓ send_attack_step OK")
    
    # Test send_embeddings
    LiveVisualizationServer.send_embeddings(
        tokens=["hello", "world"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]]
    )
    print("  ✓ send_embeddings OK")
    
    # Test send_attention_matrix
    LiveVisualizationServer.send_attention_matrix(
        layer_idx=0,
        head_idx=0,
        attention_weights=[[0.5, 0.5], [0.5, 0.5]],
        tokens=["a", "b"]
    )
    print("  ✓ send_attention_matrix OK")
    
    # Test send_transformer_trace
    LiveVisualizationServer.send_transformer_trace(
        trace_data={"tokens": ["test"], "layers": []},
        trace_type="normal"
    )
    print("  ✓ send_transformer_trace OK")
    
    # Test send_residual_update (with correct params)
    LiveVisualizationServer.send_residual_update(
        layer_idx=0,
        residual_norm=1.0,
        delta_norm=0.1
    )
    print("  ✓ send_residual_update OK")
    
    # Test send_complete (with dict param)
    LiveVisualizationServer.send_complete(
        summary={"asr": 0.8, "status": "done"}
    )
    print("  ✓ send_complete OK")
    
    return True

run_test("Server Method Calls", test_server_calls)


# ============================================================
# TEST 8: Model Selector
# ============================================================
def test_model_selector():
    """Test model selector utilities."""
    from mira.utils.model_selector import RECOMMENDED_MODELS
    
    assert len(RECOMMENDED_MODELS) > 0, "No models defined"
    print(f"  ✓ {len(RECOMMENDED_MODELS)} models available")
    
    # Check model structure
    first_model = RECOMMENDED_MODELS[0]
    assert "name" in first_model, "Model missing 'name'"
    assert "model_id" in first_model, "Model missing 'model_id'"
    print(f"  ✓ First model: {first_model['name']}")
    
    return True

run_test("Model Selector", test_model_selector)


# ============================================================
# TEST 9: Main.py Imports Check
# ============================================================
def test_main_py_imports():
    """Test all imports that main.py uses."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
sys.path.insert(0, '.')
from mira.core import ModelWrapper
from mira.utils.environment import detect_environment
from mira.analysis.subspace_probe import SubspaceProbe
from mira.analysis.evaluator import AttackSuccessEvaluator
from mira.attack.gradient import GradientAttack
from mira.visualization.live_server import LiveVisualizationServer
from mira.utils.model_selector import interactive_model_selection
print('ALL_IMPORTS_OK')
"""],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if "ALL_IMPORTS_OK" in result.stdout:
        print("  ✓ All main.py imports verified in subprocess")
        return True
    else:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        return False

run_test("Main.py Imports", test_main_py_imports)


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  TEST SUMMARY")
print("=" * 70)

passed = sum(1 for _, p, _ in test_results if p)
failed = sum(1 for _, p, _ in test_results if not p)

for name, passed_flag, error in test_results:
    status = "✅ PASS" if passed_flag else "❌ FAIL"
    print(f"  {status}: {name}")
    if error and not passed_flag:
        print(f"         Error: {error[:60]}...")

print()
print(f"  Total: {len(test_results)} tests")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print()

if all_tests_passed:
    print("  ✅ ALL TESTS PASSED - Safe to run main.py!")
else:
    print("  ❌ SOME TESTS FAILED - Fix errors before running main.py")

print("=" * 70)
print()

sys.exit(0 if all_tests_passed else 1)
