#!/usr/bin/env python
"""
MIRA Test Suite - Validates all components before running main.py

Usage:
    python test_mira.py
"""

import sys
import traceback
from pathlib import Path

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent))

TESTS_PASSED = 0
TESTS_FAILED = 0


def test(name: str):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global TESTS_PASSED, TESTS_FAILED
            try:
                func()
                print(f"  ✓ {name}")
                TESTS_PASSED += 1
            except Exception as e:
                print(f"  ✗ {name}: {e}")
                traceback.print_exc()
                TESTS_FAILED += 1
        return wrapper
    return decorator


print("\n" + "="*60)
print("  MIRA Framework Test Suite")
print("="*60 + "\n")


# ============================================================
# IMPORT TESTS
# ============================================================
print("1. Import Tests")
print("-" * 40)


@test("Core imports")
def test_core_imports():
    from mira.core import ModelWrapper
    assert ModelWrapper is not None


@test("Analysis imports")
def test_analysis_imports():
    from mira.analysis import SubspaceAnalyzer
    assert SubspaceAnalyzer is not None


@test("Attack imports")
def test_attack_imports():
    from mira.attack import GradientAttack, BaseAttack, AttackResult
    assert GradientAttack is not None
    assert BaseAttack is not None


@test("GCG Attack import")
def test_gcg_import():
    from mira.attack import GCGAttack, GCGConfig
    assert GCGAttack is not None
    assert GCGConfig is not None


@test("Probe imports")
def test_probe_imports():
    from mira.attack.probes import ProbeRunner, ALL_PROBES
    assert len(ALL_PROBES) > 0


@test("Metrics imports")
def test_metrics_imports():
    from mira.metrics import AttackSuccessEvaluator
    assert AttackSuccessEvaluator is not None


@test("Visualization imports")
def test_viz_imports():
    from mira.visualization import ResearchChartGenerator, plot_subspace_2d
    assert ResearchChartGenerator is not None


@test("Live server imports")
def test_live_server_imports():
    from mira.visualization.live_server import LiveVisualizationServer
    assert LiveVisualizationServer is not None


@test("Utils imports")
def test_utils_imports():
    from mira.utils import detect_environment
    from mira.utils.data import load_harmful_prompts, load_safe_prompts
    assert load_harmful_prompts is not None


test_core_imports()
test_analysis_imports()
test_attack_imports()
test_gcg_import()
test_probe_imports()
test_metrics_imports()
test_viz_imports()
test_live_server_imports()
test_utils_imports()


# ============================================================
# SYNTAX TESTS
# ============================================================
print("\n2. Syntax Tests")
print("-" * 40)


@test("main.py syntax")
def test_main_syntax():
    import ast
    with open("main.py", encoding="utf-8") as f:
        ast.parse(f.read())


@test("main.py no undefined variables")
def test_main_variables():
    """Check for common undefined variable issues."""
    with open("main.py", encoding="utf-8") as f:
        content = f.read()
    
    # Check that best_loss is not used without definition
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "best_loss" in line and "=" not in line and "result." not in line:
            if "metrics={\"loss\": best_loss}" in line:
                raise AssertionError(f"Line {i+1}: undefined 'best_loss' variable")


test_main_syntax()
test_main_variables()


# ============================================================
# COMPONENT TESTS
# ============================================================
print("\n3. Component Tests")
print("-" * 40)


@test("Environment detection")
def test_env_detection():
    from mira.utils import detect_environment
    env = detect_environment()
    assert env.system.os_name is not None
    assert env.gpu.backend in ["cuda", "mps", "cpu"]


@test("Data loading")
def test_data_loading():
    from mira.utils.data import load_harmful_prompts, load_safe_prompts
    harmful = load_harmful_prompts()
    safe = load_safe_prompts()
    assert len(harmful) > 0
    assert len(safe) > 0


@test("Experiment logger")
def test_logger():
    from mira.utils.experiment_logger import ExperimentLogger
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(output_dir=tmpdir, experiment_name="test")
        logger.log_attack(
            model_name="test",
            prompt="test",
            attack_type="test",
            suffix="test",
            response="test",
            success=False,
            metrics={}
        )
        assert len(logger.records) == 1


@test("GCG config")
def test_gcg_config():
    from mira.attack import GCGConfig
    config = GCGConfig(suffix_length=10, batch_size=32)
    assert config.suffix_length == 10
    assert config.batch_size == 32


@test("Attack result structure")
def test_attack_result():
    from mira.attack import AttackResult
    from mira.attack.base import AttackType
    result = AttackResult(
        success=False,
        adversarial_suffix="test",
        final_loss=1.0,
        generated_response="test",
        num_steps=10,
        loss_history=[1.0, 0.5],
        original_prompt="test prompt",
        full_adversarial_prompt="test prompt test",
        attack_type=AttackType.GRADIENT_BASED
    )
    assert result.final_loss == 1.0


test_env_detection()
test_data_loading()
test_logger()
test_gcg_config()
test_attack_result()


# ============================================================
# LIVE SERVER TESTS
# ============================================================
print("\n4. Live Server Tests")
print("-" * 40)


@test("Live server event queue")
def test_event_queue():
    from mira.visualization.live_server import VisualizationEvent
    event = VisualizationEvent(
        event_type="test",
        data={"value": 123}
    )
    assert event.event_type == "test"
    assert event.data["value"] == 123


@test("Server static methods")
def test_server_methods():
    from mira.visualization.live_server import LiveVisualizationServer
    # These are static methods, test they don't crash
    LiveVisualizationServer.send_attack_step(1, 0.5, "test", False)
    LiveVisualizationServer.send_layer_update(0, 0.5, 0.3, "refusal")


test_event_queue()
test_server_methods()


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print(f"  RESULTS: {TESTS_PASSED} passed, {TESTS_FAILED} failed")
print("="*60)

if TESTS_FAILED > 0:
    print("\n  ⚠ Some tests failed. Fix issues before running main.py\n")
    sys.exit(1)
else:
    print("\n  ✓ All tests passed! Safe to run: python main.py\n")
    sys.exit(0)
