"""
Comprehensive test suite for MIRA framework.

Tests all major components:
- Configuration loading
- Environment detection
- Analysis modules
- Attack modules
- Metrics
- Visualization
"""

import pytest
import sys
import os
from pathlib import Path

# Add framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfiguration:
    """Tests for configuration loading."""
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        from mira.config import load_config
        
        config = load_config()
        
        assert config is not None
        assert "model" in config
        assert "evaluation" in config
    
    def test_config_has_required_sections(self):
        """Test config contains all required sections."""
        from mira.config import load_config
        
        config = load_config()
        
        required_sections = ["model", "analysis", "attack", "evaluation", "visualization"]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"
    
    def test_mira_config_dataclass(self):
        """Test MiraConfig dataclass creation."""
        from mira.config import MiraConfig
        
        config = MiraConfig.load()
        
        assert config.model is not None
        assert config.analysis is not None
        assert config.attack is not None
        assert config.evaluation is not None
    
    def test_model_config_device_detection(self):
        """Test device detection in model config."""
        from mira.config import ModelConfig
        
        config = ModelConfig()
        device = config.get_device()
        
        assert device in ["cpu", "cuda", "mps"]


class TestEnvironmentDetection:
    """Tests for environment detection."""
    
    def test_detect_system(self):
        """Test system detection."""
        from mira.utils.environment import detect_system
        
        system = detect_system()
        
        assert system.os_name in ["Windows", "Linux", "Darwin"]
        assert system.python_version is not None
    
    def test_detect_gpu(self):
        """Test GPU detection."""
        from mira.utils.environment import detect_gpu
        
        gpu = detect_gpu()
        
        assert gpu.backend in ["cpu", "cuda", "mps"]
        assert isinstance(gpu.available, bool)
    
    def test_full_environment_detection(self):
        """Test complete environment detection."""
        from mira.utils.environment import detect_environment
        
        env = detect_environment()
        
        assert env.system is not None
        assert env.gpu is not None
        assert env.recommended_model is not None
    
    def test_device_string(self):
        """Test getting device string."""
        from mira.utils.environment import get_device_string
        
        device = get_device_string()
        
        assert device in ["cpu", "cuda", "mps"]


class TestExperimentLogger:
    """Tests for experiment logging."""
    
    def test_logger_creation(self, tmp_path):
        """Test logger initialization."""
        from mira.utils.experiment_logger import ExperimentLogger
        
        logger = ExperimentLogger(
            output_dir=str(tmp_path),
            experiment_name="test_exp",
        )
        
        assert logger.experiment_name == "test_exp"
        assert logger.data_dir.exists()
        assert logger.charts_dir.exists()
    
    def test_log_attack(self, tmp_path):
        """Test logging an attack."""
        from mira.utils.experiment_logger import ExperimentLogger
        
        logger = ExperimentLogger(output_dir=str(tmp_path))
        
        logger.log_attack(
            model_name="test-model",
            prompt="test prompt",
            attack_type="gradient",
            suffix="test suffix",
            response="test response",
            success=True,
            metrics={"loss": 0.5},
        )
        
        assert len(logger.records) == 1
        assert logger.records[0].success == True
    
    def test_save_to_csv(self, tmp_path):
        """Test saving to CSV."""
        from mira.utils.experiment_logger import ExperimentLogger
        
        logger = ExperimentLogger(output_dir=str(tmp_path))
        logger.log_attack(
            model_name="test",
            prompt="test",
            attack_type="test",
            suffix="",
            response="",
            success=True,
            metrics={},
        )
        
        csv_path = logger.save_to_csv()
        
        assert Path(csv_path).exists()
    
    def test_get_summary(self, tmp_path):
        """Test getting experiment summary."""
        from mira.utils.experiment_logger import ExperimentLogger
        
        logger = ExperimentLogger(output_dir=str(tmp_path))
        logger.log_attack(
            model_name="test",
            prompt="test",
            attack_type="test",
            suffix="",
            response="",
            success=True,
            metrics={"asr": 1.0},
        )
        
        summary = logger.get_summary()
        
        assert summary["total_attacks"] == 1
        assert summary["successful_attacks"] == 1


class TestMetrics:
    """Tests for metric computations."""
    
    def test_attack_success_evaluator_refusal(self):
        """Test refusal detection."""
        from mira.metrics.success_rate import AttackSuccessEvaluator
        
        evaluator = AttackSuccessEvaluator()
        
        result = evaluator.evaluate_single(
            "test",
            "I'm sorry, but I cannot help with that request."
        )
        
        assert result["success"] == False
        assert result["has_refusal_markers"] == True
    
    def test_attack_success_evaluator_acceptance(self):
        """Test acceptance detection."""
        from mira.metrics.success_rate import AttackSuccessEvaluator
        
        evaluator = AttackSuccessEvaluator()
        
        result = evaluator.evaluate_single(
            "test",
            "Sure, here is the information you requested."
        )
        
        assert result["success"] == True
        assert result["has_acceptance_markers"] == True
    
    def test_batch_evaluation(self):
        """Test batch evaluation."""
        from mira.metrics.success_rate import AttackSuccessEvaluator
        
        evaluator = AttackSuccessEvaluator()
        
        results = [
            {"prompt": "p1", "response": "Sure, here you go."},
            {"prompt": "p2", "response": "I cannot help with that."},
        ]
        
        metrics = evaluator.evaluate_batch(results)
        
        assert metrics.total_attacks == 2
        assert metrics.successful_attacks == 1
        assert metrics.asr == 0.5
    
    def test_probability_metrics(self):
        """Test probability metrics."""
        import torch
        from mira.metrics.probability import ProbabilityMetrics
        
        metrics = ProbabilityMetrics(vocab_size=100)
        
        # Uniform logits
        logits = torch.zeros(100)
        result = metrics.compute_entropy(logits)
        
        assert result.normalized_entropy > 0.9
    
    def test_subspace_distance_metrics(self):
        """Test subspace distance computation."""
        import torch
        from mira.metrics.distance import SubspaceDistanceMetrics
        
        refusal = torch.tensor([1.0, 0.0, 0.0])
        acceptance = torch.tensor([-1.0, 0.0, 0.0])
        
        metrics = SubspaceDistanceMetrics(refusal, acceptance)
        
        point = torch.tensor([2.0, 0.0, 0.0])
        proj = metrics.direction_projection(point, "refusal")
        
        assert proj > 0


class TestVisualization:
    """Tests for visualization modules."""
    
    def test_research_chart_generator_creation(self, tmp_path):
        """Test chart generator initialization."""
        from mira.visualization.research_charts import ResearchChartGenerator
        
        gen = ResearchChartGenerator(output_dir=str(tmp_path))
        
        assert gen.output_dir.exists()
    
    def test_plot_loss_curve(self, tmp_path):
        """Test loss curve generation."""
        from mira.visualization.research_charts import ResearchChartGenerator
        
        gen = ResearchChartGenerator(output_dir=str(tmp_path))
        
        loss_history = [1.0, 0.8, 0.6, 0.4, 0.3]
        path = gen.plot_loss_curve(loss_history, save_name="test_loss")
        
        if path:  # Only if matplotlib available
            assert Path(path).exists()
    
    def test_plot_asr(self, tmp_path):
        """Test ASR bar chart generation."""
        from mira.visualization.research_charts import ResearchChartGenerator
        
        gen = ResearchChartGenerator(output_dir=str(tmp_path))
        
        path = gen.plot_attack_success_rate(
            models=["Model A", "Model B"],
            asr_values=[0.3, 0.7],
            save_name="test_asr",
        )
        
        if path:  # Only if matplotlib available
            assert Path(path).exists()


class TestDataLoading:
    """Tests for data loading utilities."""
    
    def test_load_harmful_prompts(self):
        """Test loading harmful prompts."""
        from mira.utils.data import load_harmful_prompts
        
        prompts = load_harmful_prompts()
        
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)
    
    def test_load_safe_prompts(self):
        """Test loading safe prompts."""
        from mira.utils.data import load_safe_prompts
        
        prompts = load_safe_prompts()
        
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)
    
    def test_load_by_category(self):
        """Test loading prompts by category."""
        from mira.utils.data import load_harmful_prompts, load_safe_prompts
        
        harmful = load_harmful_prompts(category="instruction_override")
        safe = load_safe_prompts(category="greeting")
        
        assert len(harmful) > 0
        assert len(safe) > 0


class TestExperimentRunner:
    """Tests for integrated experiment runner."""
    
    def test_runner_creation(self, tmp_path):
        """Test runner initialization."""
        from mira.runner import ExperimentRunner
        
        runner = ExperimentRunner(
            output_dir=str(tmp_path),
            auto_detect_env=True,
        )
        
        assert runner.env is not None
        assert runner.logger is not None
    
    def test_environment_recommendations(self, tmp_path):
        """Test environment-based recommendations."""
        from mira.runner import ExperimentRunner
        
        runner = ExperimentRunner(output_dir=str(tmp_path))
        
        # Should have recommended model based on hardware
        assert runner.config.model.name is not None


class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_core(self):
        """Test core module imports."""
        from mira.core import ModelWrapper, HookManager
        assert ModelWrapper is not None
        assert HookManager is not None
    
    def test_import_analysis(self):
        """Test analysis module imports."""
        from mira.analysis import (
            SubspaceAnalyzer,
            ActivationAnalyzer,
            AttentionAnalyzer,
            LogitLens,
        )
        assert SubspaceAnalyzer is not None
    
    def test_import_attack(self):
        """Test attack module imports."""
        from mira.attack import (
            BaseAttack,
            ReroutingAttack,
            GradientAttack,
            ProxyAttack,
        )
        assert BaseAttack is not None
    
    def test_import_metrics(self):
        """Test metrics module imports."""
        from mira.metrics import (
            compute_asr,
            AttackSuccessEvaluator,
            SubspaceDistanceMetrics,
            ProbabilityMetrics,
        )
        assert compute_asr is not None
    
    def test_import_visualization(self):
        """Test visualization module imports."""
        from mira.visualization import (
            plot_subspace_2d,
            plot_attention_heatmap,
            ResearchChartGenerator,
        )
        assert plot_subspace_2d is not None
    
    def test_import_utils(self):
        """Test utils module imports."""
        from mira.utils import (
            detect_environment,
            ExperimentLogger,
            setup_logger,
        )
        assert detect_environment is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
