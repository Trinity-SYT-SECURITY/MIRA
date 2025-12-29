"""
Complete Flow Test Suite for MIRA Visualization

Tests ALL components before running main.py to catch errors early.
Run: python tests/test_complete_flow.py
"""

import sys
import os
import traceback

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        test_results.append((name, False, str(e)))
        all_tests_passed = False
        return False


# ============================================================
# TEST 1: Core Module Imports
# ============================================================
def test_core_imports():
    """Test core MIRA module imports."""
    from mira.core import ModelWrapper
    print("  ✓ mira.core.ModelWrapper")
    
    from mira.core.hook_manager import HookManager
    print("  ✓ mira.core.hook_manager.HookManager")
    
    return True

run_test("Core Imports", test_core_imports)


# ============================================================
# TEST 2: Visualization Module Imports
# ============================================================
def test_visualization_imports():
    """Test visualization module imports."""
    from mira.visualization.live_server import LiveVisualizationServer
    print("  ✓ mira.visualization.live_server.LiveVisualizationServer")
    
    from mira.visualization.attack_flow import get_attack_flow_html
    print("  ✓ mira.visualization.attack_flow.get_attack_flow_html")
    
    from mira.visualization.transformer_flow import get_transformer_flow_html
    print("  ✓ mira.visualization.transformer_flow.get_transformer_flow_html")
    
    from mira.visualization.simple_dashboard import get_simple_dashboard
    print("  ✓ mira.visualization.simple_dashboard.get_simple_dashboard")
    
    return True

run_test("Visualization Imports", test_visualization_imports)


# ============================================================
# TEST 3: Attack Flow Dashboard HTML
# ============================================================
def test_attack_flow_dashboard():
    """Test attack flow dashboard HTML."""
    from mira.visualization.attack_flow import get_attack_flow_html
    
    html = get_attack_flow_html()
    
    assert len(html) > 5000, f"HTML too short: {len(html)} chars"
    print(f"  ✓ HTML length: {len(html)} chars")
    
    # Check essential elements
    checks = [
        ("<html", "HTML tag"),
        ("EventSource", "SSE connection"),
        ("flow-container", "Flow container"),
        ("stage-input", "Input stage"),
        ("stage-embedding", "Embedding stage"),
        ("stage-attention", "Attention stage"),
        ("stage-mlp", "MLP stage"),
        ("stage-output", "Output stage"),
        ("handleAttackStep", "Attack step handler"),
        ("handleEmbeddings", "Embeddings handler"),
        ("animateStages", "Stage animation"),
    ]
    
    for pattern, desc in checks:
        assert pattern in html, f"Missing: {desc}"
        print(f"  ✓ {desc} present")
    
    return True

run_test("Attack Flow Dashboard", test_attack_flow_dashboard)


# ============================================================
# TEST 4: Live Server Methods Signatures
# ============================================================
def test_live_server_methods():
    """Test all LiveVisualizationServer methods exist."""
    from mira.visualization.live_server import LiveVisualizationServer
    import inspect
    
    # Methods used in main.py
    required_methods = [
        'send_layer_update',
        'send_attack_step', 
        'send_embeddings',
        'send_transformer_trace',
        'send_attention_matrix',
        'send_residual_update',
        'send_complete',
    ]
    
    for method_name in required_methods:
        method = getattr(LiveVisualizationServer, method_name, None)
        assert method is not None, f"Method {method_name} not found"
        print(f"  ✓ {method_name} exists")
    
    return True

run_test("Live Server Methods", test_live_server_methods)


# ============================================================
# TEST 5: Server Method Execution
# ============================================================
def test_server_method_calls():
    """Test that server method calls work without errors."""
    from mira.visualization.live_server import LiveVisualizationServer, event_queue
    
    # Clear queue first
    while not event_queue.empty():
        try:
            event_queue.get_nowait()
        except:
            break
    
    # Test send_layer_update
    LiveVisualizationServer.send_layer_update(
        layer_idx=0,
        refusal_score=0.5,
        acceptance_score=0.5,
        direction="forward"
    )
    print("  ✓ send_layer_update call OK")
    
    # Test send_attack_step
    LiveVisualizationServer.send_attack_step(
        step=1,
        loss=5.0,
        suffix="test suffix",
        success=False
    )
    print("  ✓ send_attack_step call OK")
    
    # Test send_embeddings
    LiveVisualizationServer.send_embeddings(
        tokens=["hello", "world"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]]
    )
    print("  ✓ send_embeddings call OK")
    
    # Test send_attention_matrix
    LiveVisualizationServer.send_attention_matrix(
        layer_idx=0,
        head_idx=0,
        attention_weights=[[0.5, 0.5], [0.5, 0.5]],
        tokens=["a", "b"]
    )
    print("  ✓ send_attention_matrix call OK")
    
    # Test send_transformer_trace
    LiveVisualizationServer.send_transformer_trace(
        trace_data={"tokens": ["test"], "layers": []},
        trace_type="normal"
    )
    print("  ✓ send_transformer_trace call OK")
    
    # Test send_residual_update
    LiveVisualizationServer.send_residual_update(
        layer_idx=0,
        residual_norm=1.0,
        delta_norm=0.1
    )
    print("  ✓ send_residual_update call OK")
    
    # Test send_complete
    LiveVisualizationServer.send_complete(
        summary={"asr": 0.8, "status": "done"}
    )
    print("  ✓ send_complete call OK")
    
    # Verify events were queued
    count = 0
    while not event_queue.empty():
        try:
            event_queue.get_nowait()
            count += 1
        except:
            break
    
    assert count >= 7, f"Expected 7+ events, got {count}"
    print(f"  ✓ {count} events queued correctly")
    
    return True

run_test("Server Method Calls", test_server_method_calls)


# ============================================================
# TEST 6: Attack Module
# ============================================================
def test_attack_module():
    """Test attack module imports."""
    from mira.attack.gradient import GradientAttack
    print("  ✓ mira.attack.gradient.GradientAttack")
    
    from mira.attack.base import BaseAttack
    print("  ✓ mira.attack.base.BaseAttack")
    
    return True

run_test("Attack Module", test_attack_module)


# ============================================================
# TEST 7: Analysis Module
# ============================================================
def test_analysis_module():
    """Test analysis module imports."""
    try:
        from mira.analysis.transformer_tracer import TransformerTracer
        print("  ✓ mira.analysis.transformer_tracer.TransformerTracer")
    except ImportError as e:
        print(f"  ⚠ TransformerTracer: {e}")
    
    try:
        from mira.analysis.evaluator import AttackSuccessEvaluator
        print("  ✓ mira.analysis.evaluator.AttackSuccessEvaluator")
    except ImportError as e:
        print(f"  ⚠ AttackSuccessEvaluator: {e}")
    
    return True

run_test("Analysis Module", test_analysis_module)


# ============================================================
# TEST 8: Utils Module
# ============================================================
def test_utils_module():
    """Test utils module imports."""
    from mira.utils.environment import detect_environment
    print("  ✓ mira.utils.environment.detect_environment")
    
    from mira.utils.model_selector import select_model_interactive
    print("  ✓ mira.utils.model_selector.select_model_interactive")
    
    return True

run_test("Utils Module", test_utils_module)


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
    print()
    print("  Next step:")
    print("    python main.py")
    print()
    print("  Tip: Select Pythia 70M (option 1) for fastest response")
else:
    print("  ❌ SOME TESTS FAILED - Fix errors before running main.py")

print("=" * 70)
print()

sys.exit(0 if all_tests_passed else 1)
