"""
Test script to verify all visualization components work correctly.
Run this BEFORE running main.py to catch errors early.
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    try:
        from mira.visualization.live_server import LiveVisualizationServer
        from mira.visualization.simple_dashboard import get_simple_dashboard
        print("  ✓ Visualization imports OK")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_dashboard():
    """Test dashboard HTML is valid."""
    print("Testing dashboard...")
    try:
        from mira.visualization.simple_dashboard import get_simple_dashboard
        html = get_simple_dashboard()
        assert len(html) > 1000, "Dashboard HTML too short"
        assert "<html" in html, "Missing HTML tag"
        assert "EventSource" in html, "Missing SSE code"
        print("  ✓ Dashboard HTML OK")
        return True
    except Exception as e:
        print(f"  ✗ Dashboard error: {e}")
        return False


def test_server_methods():
    """Test all LiveVisualizationServer methods have correct signatures."""
    print("Testing server methods...")
    try:
        from mira.visualization.live_server import LiveVisualizationServer
        import inspect
        
        # Test each method signature
        methods_to_test = [
            ('send_layer_update', ['layer_idx', 'refusal_score', 'acceptance_score', 'direction']),
            ('send_attack_step', ['step', 'loss', 'suffix', 'success']),
            ('send_attention_update', ['layer_idx', 'head_idx', 'attention_weights', 'tokens']),
            ('send_embeddings', ['tokens', 'embeddings']),
            ('send_transformer_trace', ['trace_data', 'trace_type']),
            ('send_attention_matrix', ['layer_idx', 'head_idx', 'attention_weights', 'tokens']),
        ]
        
        for method_name, expected_params in methods_to_test:
            method = getattr(LiveVisualizationServer, method_name, None)
            if method is None:
                print(f"  ✗ Method {method_name} not found")
                return False
            
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            
            for param in expected_params:
                if param not in params:
                    print(f"  ✗ Method {method_name} missing param: {param}")
                    print(f"    Actual params: {params}")
                    return False
            
            print(f"  ✓ {method_name} signature OK")
        
        return True
    except Exception as e:
        print(f"  ✗ Server method error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server_calls():
    """Test that server method calls work without errors."""
    print("Testing server calls...")
    try:
        from mira.visualization.live_server import LiveVisualizationServer
        
        # These should not raise errors (even without a running server)
        LiveVisualizationServer.send_layer_update(
            layer_idx=0,
            refusal_score=0.5,
            acceptance_score=0.5,
            direction="forward"
        )
        print("  ✓ send_layer_update OK")
        
        LiveVisualizationServer.send_attack_step(
            step=1,
            loss=5.0,
            suffix="test",
            success=False
        )
        print("  ✓ send_attack_step OK")
        
        LiveVisualizationServer.send_embeddings(
            tokens=["hello", "world"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]]
        )
        print("  ✓ send_embeddings OK")
        
        LiveVisualizationServer.send_attention_matrix(
            layer_idx=0,
            head_idx=0,
            attention_weights=[[0.5, 0.5], [0.5, 0.5]],
            tokens=["a", "b"]
        )
        print("  ✓ send_attention_matrix OK")
        
        return True
    except Exception as e:
        print(f"  ✗ Server call error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_imports():
    """Test main.py imports work."""
    print("Testing main.py imports...")
    try:
        # Import what main.py imports
        from mira.core import ModelWrapper
        from mira.utils.environment import detect_environment  # Use function
        from mira.analysis import SubspaceProbe, AttackSuccessEvaluator
        from mira.attack import GradientAttack
        from mira.visualization.live_server import LiveVisualizationServer
        from mira.utils.model_selector import interactive_model_selection
        print("  ✓ Main imports OK")
        return True
    except Exception as e:
        print(f"  ✗ Main import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("  MIRA VISUALIZATION TEST SUITE")
    print("="*60 + "\n")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Dashboard", test_dashboard()))
    results.append(("Server Methods", test_server_methods()))
    results.append(("Server Calls", test_server_calls()))
    results.append(("Main Imports", test_main_imports()))
    
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("  ✅ ALL TESTS PASSED - Safe to run main.py")
    else:
        print("  ❌ SOME TESTS FAILED - Fix errors before running main.py")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
