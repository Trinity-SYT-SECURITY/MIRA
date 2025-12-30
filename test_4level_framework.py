"""
Test script for 4-level multi-model comparison framework.

Tests all components to ensure they work correctly.
"""

import sys
sys.path.insert(0, '/Users/yiting.shen/Library/CloudStorage/OneDrive-ONEDEGREEHONGKONGLIMITED/MIRA')

from mira.analysis.phase_comparison import PhaseComparisonAnalyzer
from mira.analysis.cross_model_metrics import CrossModelAnalyzer
from mira.analysis.transferability import TransferabilityAnalyzer
from mira.visualization.comprehensive_viz import ComprehensiveVisualizer
import numpy as np


def create_mock_model_results(n_models=3):
    """Create mock model results for testing."""
    results = []
    
    for i in range(n_models):
        model_name = f"test_model_{i}"
        
        # Create mock data
        result = {
            "success": True,
            "model_name": model_name,
            "asr": 0.3 + i * 0.2,
            "probe_accuracy": 0.7 - i * 0.1,
            "probe_bypass_rate": 0.2 + i * 0.15,
            "mean_entropy": 2.5 + i * 0.5,
            "attack_details": [
                {"prompt": f"prompt_{j}", "attack_type": "dan", "success": j % 2 == 0}
                for j in range(10)
            ] + [
                {"prompt": f"prompt_{j}", "attack_type": "gradient", "success": j % 3 == 0}
                for j in range(5)
            ],
            "layer_activations": {
                "clean": [0.1 + j * 0.05 for j in range(12)],
                "attack": [0.15 + j * 0.06 for j in range(12)],
            },
            "internal_metrics": {
                "refusal_direction": np.random.randn(768).tolist(),
                "refusal_norm": 10.0 + i,
                "entropy_by_attack": {
                    "successful": [-0.3, -0.4, -0.35],
                    "failed": [-0.05, -0.02, -0.03],
                },
                "layer_divergence_point": 4 + i,
            },
            "attention_data": {
                "baseline_clean": np.random.rand(8, 8).tolist(),
                "baseline_attack": np.random.rand(8, 8).tolist(),
            },
        }
        
        results.append(result)
    
    return results


def test_phase_comparison():
    """Test Phase Comparison Analyzer."""
    print("Testing Phase Comparison Analyzer...")
    
    results = create_mock_model_results(3)
    analyzer = PhaseComparisonAnalyzer()
    
    # Test analysis
    phase_comparison = analyzer.analyze_phase_sensitivity(results)
    assert len(phase_comparison) > 0, "Phase comparison should return results"
    
    # Test HTML generation
    html = analyzer.generate_phase_heatmap_html(phase_comparison)
    assert len(html) > 0, "Should generate HTML"
    assert "heatmap" in html.lower(), "Should contain heatmap"
    
    print("  ✓ Phase Comparison Analyzer works")
    return phase_comparison


def test_cross_model_metrics():
    """Test Cross-Model Metrics Analyzer."""
    print("Testing Cross-Model Metrics Analyzer...")
    
    results = create_mock_model_results(3)
    analyzer = CrossModelAnalyzer()
    
    # Test similarity matrix
    similarity_matrix, model_names = analyzer.compute_refusal_direction_similarity(results)
    assert similarity_matrix.shape[0] == len(model_names), "Matrix size should match model count"
    
    # Test entropy analysis
    entropy_analysis = analyzer.analyze_entropy_patterns(results)
    assert "summary" in entropy_analysis, "Should have summary"
    
    # Test layer analysis
    layer_analysis = analyzer.identify_common_failure_layers(results)
    assert "summary" in layer_analysis, "Should have summary"
    
    # Test HTML generation
    html = analyzer.generate_similarity_matrix_html(similarity_matrix, model_names)
    assert len(html) > 0, "Should generate HTML"
    
    html = analyzer.generate_entropy_comparison_html(entropy_analysis)
    assert len(html) > 0, "Should generate HTML"
    
    print("  ✓ Cross-Model Metrics Analyzer works")
    return {
        "similarity_matrix": similarity_matrix,
        "model_names": model_names,
        "entropy_analysis": entropy_analysis,
        "layer_analysis": layer_analysis,
    }


def test_transferability():
    """Test Transferability Analyzer."""
    print("Testing Transferability Analyzer...")
    
    results = create_mock_model_results(3)
    analyzer = TransferabilityAnalyzer()
    
    # Test transfer analysis
    transfer_data = analyzer.compute_cross_model_transfer(results)
    assert "transfer_matrix" in transfer_data, "Should have transfer matrix"
    assert "summary" in transfer_data, "Should have summary"
    
    # Test systematic vs random
    systematic_comparison = analyzer.compare_systematic_vs_random(results)
    assert "systematic_mean_asr" in systematic_comparison, "Should have systematic ASR"
    
    # Test HTML generation
    html = analyzer.generate_transfer_matrix_html(transfer_data)
    assert len(html) > 0, "Should generate HTML"
    
    html = analyzer.generate_systematic_vs_random_html(systematic_comparison)
    assert len(html) > 0, "Should generate HTML"
    
    print("  ✓ Transferability Analyzer works")
    return {
        "transfer_data": transfer_data,
        "systematic_comparison": systematic_comparison,
    }


def test_comprehensive_visualizer():
    """Test Comprehensive Visualizer."""
    print("Testing Comprehensive Visualizer...")
    
    results = create_mock_model_results(3)
    visualizer = ComprehensiveVisualizer()
    
    # Test layer heatmap
    html = visualizer.generate_layer_activation_heatmap(results)
    assert len(html) > 0, "Should generate HTML"
    assert "heatmap" in html.lower(), "Should contain heatmap"
    
    # Test attention difference maps
    html = visualizer.generate_attention_difference_maps(results)
    assert len(html) > 0, "Should generate HTML"
    
    print("  ✓ Comprehensive Visualizer works")


def test_integration():
    """Test full integration."""
    print("\nTesting Full Integration...")
    
    results = create_mock_model_results(3)
    
    # Run all analyzers
    phase_analyzer = PhaseComparisonAnalyzer()
    cross_analyzer = CrossModelAnalyzer()
    transfer_analyzer = TransferabilityAnalyzer()
    visualizer = ComprehensiveVisualizer()
    
    phase_comparison = phase_analyzer.analyze_phase_sensitivity(results)
    
    similarity_matrix, model_names = cross_analyzer.compute_refusal_direction_similarity(results)
    entropy_analysis = cross_analyzer.analyze_entropy_patterns(results)
    layer_analysis = cross_analyzer.identify_common_failure_layers(results)
    
    cross_model_analysis = {
        "similarity_matrix": similarity_matrix,
        "model_names": model_names,
        "entropy_analysis": entropy_analysis,
        "layer_analysis": layer_analysis,
    }
    
    transfer_data = transfer_analyzer.compute_cross_model_transfer(results)
    systematic_comparison = transfer_analyzer.compare_systematic_vs_random(results)
    
    transferability_analysis = {
        "transfer_data": transfer_data,
        "systematic_comparison": systematic_comparison,
    }
    
    # Generate all visualizations
    phase_html = phase_analyzer.generate_phase_heatmap_html(phase_comparison)
    similarity_html = cross_analyzer.generate_similarity_matrix_html(similarity_matrix, model_names)
    entropy_html = cross_analyzer.generate_entropy_comparison_html(entropy_analysis)
    transfer_html = transfer_analyzer.generate_transfer_matrix_html(transfer_data)
    systematic_html = transfer_analyzer.generate_systematic_vs_random_html(systematic_comparison)
    layer_heatmap_html = visualizer.generate_layer_activation_heatmap(results)
    attention_html = visualizer.generate_attention_difference_maps(results)
    
    # Verify all HTML generated
    all_html = [
        phase_html, similarity_html, entropy_html,
        transfer_html, systematic_html, layer_heatmap_html, attention_html
    ]
    
    for i, html in enumerate(all_html):
        assert len(html) > 0, f"HTML {i} should not be empty"
    
    print("  ✓ Full integration works")
    print(f"  ✓ Generated {len(all_html)} visualizations")


def main():
    """Run all tests."""
    print("=" * 60)
    print("4-Level Multi-Model Comparison Framework Test")
    print("=" * 60)
    
    try:
        test_phase_comparison()
        test_cross_model_metrics()
        test_transferability()
        test_comprehensive_visualizer()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
