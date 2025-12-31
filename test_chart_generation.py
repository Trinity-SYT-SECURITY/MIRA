#!/usr/bin/env python3
"""
Test script for chart generation.

Tests:
1. Subspace chart generation (subspace.png)
2. ASR chart generation (asr.png)
3. Directory creation
4. File existence verification
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_subspace_chart():
    """Test subspace chart generation."""
    print("=" * 70)
    print("TEST 1: Subspace Chart Generation")
    print("=" * 70)
    
    try:
        from mira.visualization import plot_subspace_2d
        
        # Create test data
        print("  Creating test embeddings...")
        safe_embeddings = torch.randn(10, 128)  # 10 safe prompts, 128-dim embeddings
        unsafe_embeddings = torch.randn(10, 128)  # 10 unsafe prompts
        refusal_direction = torch.randn(128)  # Refusal direction vector
        
        # Create test output directory
        test_output_dir = project_root / "test_output" / "charts"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = test_output_dir / "subspace.png"
        print(f"  Saving to: {save_path}")
        
        # Generate chart
        print("  Generating chart...")
        fig = plot_subspace_2d(
            safe_embeddings=safe_embeddings,
            unsafe_embeddings=unsafe_embeddings,
            refusal_direction=refusal_direction,
            title="Test Subspace Visualization",
            save_path=str(save_path),
        )
        
        # Verify file exists
        if save_path.exists():
            file_size = save_path.stat().st_size
            print(f"  ✓ Chart saved successfully!")
            print(f"    File: {save_path}")
            print(f"    Size: {file_size:,} bytes")
            return True
        else:
            print(f"  ✗ Chart file not found: {save_path}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_asr_chart():
    """Test ASR chart generation."""
    print("\n" + "=" * 70)
    print("TEST 2: ASR Chart Generation")
    print("=" * 70)
    
    try:
        from mira.visualization.research_charts import ResearchChartGenerator
        
        # Create test output directory
        test_output_dir = project_root / "test_output" / "charts"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Output directory: {test_output_dir}")
        
        # Initialize chart generator
        chart_gen = ResearchChartGenerator(output_dir=str(test_output_dir))
        print("  Chart generator initialized")
        
        # Generate ASR chart
        print("  Generating ASR chart...")
        chart_path = chart_gen.plot_attack_success_rate(
            models=["Gradient Attack", "Prompt Attack", "SSR Attack"],
            asr_values=[0.65, 0.45, 0.72],
            title="Test Attack Success Rate",
            save_name="asr",
        )
        
        # Verify file exists
        asr_file = test_output_dir / "asr.png"
        if asr_file.exists():
            file_size = asr_file.stat().st_size
            print(f"  ✓ Chart saved successfully!")
            print(f"    File: {asr_file}")
            print(f"    Size: {file_size:,} bytes")
            print(f"    Returned path: {chart_path}")
            return True
        else:
            print(f"  ✗ Chart file not found: {asr_file}")
            if chart_path:
                print(f"    Returned path was: {chart_path}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_charts():
    """Test multiple chart types."""
    print("\n" + "=" * 70)
    print("TEST 3: Multiple Chart Types")
    print("=" * 70)
    
    try:
        from mira.visualization.research_charts import ResearchChartGenerator
        
        # Create test output directory
        test_output_dir = project_root / "test_output" / "charts"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        chart_gen = ResearchChartGenerator(output_dir=str(test_output_dir))
        
        charts_to_test = [
            {
                "name": "asr_by_type",
                "method": "plot_asr_by_attack_type",
                "args": {
                    "attack_types": ["Jailbreak", "Encoding", "Social"],
                    "asr_values": [0.6, 0.4, 0.5],
                    "model_name": "Test Model",
                    "title": "ASR by Attack Type",
                    "save_name": "asr_by_type",
                }
            },
            {
                "name": "loss_curve",
                "method": "plot_loss_curve",
                "args": {
                    "loss_history": [1.0, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2],
                    "title": "Test Loss Curve",
                    "save_name": "loss_curve",
                }
            },
        ]
        
        results = {}
        for chart_info in charts_to_test:
            method_name = chart_info["method"]
            chart_name = chart_info["name"]
            args = chart_info["args"]
            
            print(f"  Testing {chart_name}...")
            try:
                method = getattr(chart_gen, method_name)
                chart_path = method(**args)
                
                expected_file = test_output_dir / f"{args['save_name']}.png"
                if expected_file.exists():
                    file_size = expected_file.stat().st_size
                    print(f"    ✓ {chart_name} saved ({file_size:,} bytes)")
                    results[chart_name] = True
                else:
                    print(f"    ✗ {chart_name} file not found")
                    results[chart_name] = False
            except Exception as e:
                print(f"    ✗ {chart_name} failed: {e}")
                results[chart_name] = False
        
        return all(results.values())
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_creation():
    """Test that directories are created correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: Directory Creation")
    print("=" * 70)
    
    try:
        from mira.visualization.research_charts import ResearchChartGenerator
        
        # Test with nested directory path
        nested_dir = project_root / "test_output" / "nested" / "charts" / "deep"
        print(f"  Testing nested directory: {nested_dir}")
        
        chart_gen = ResearchChartGenerator(output_dir=str(nested_dir))
        
        # Generate a simple chart
        chart_path = chart_gen.plot_attack_success_rate(
            models=["Test"],
            asr_values=[0.5],
            title="Directory Test",
            save_name="dir_test",
        )
        
        # Check if directory was created
        if nested_dir.exists() and nested_dir.is_dir():
            print(f"  ✓ Directory created: {nested_dir}")
            
            # Check if file exists
            test_file = nested_dir / "dir_test.png"
            if test_file.exists():
                print(f"  ✓ File created in nested directory")
                return True
            else:
                print(f"  ✗ File not found in nested directory")
                return False
        else:
            print(f"  ✗ Directory not created")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CHART GENERATION TEST SUITE")
    print("=" * 70)
    print()
    
    results = {}
    
    # Run tests
    results["subspace"] = test_subspace_chart()
    results["asr"] = test_asr_chart()
    results["multiple"] = test_multiple_charts()
    results["directory"] = test_directory_creation()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:15} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n  Total: {total} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    
    if passed == total:
        print("\n  ✓ All tests passed!")
        return 0
    else:
        print("\n  ✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


