"""
Tests for subspace analysis functionality.
"""

import pytest
import torch
import numpy as np


class TestSubspaceResult:
    """Tests for SubspaceResult dataclass."""
    
    def test_creation(self):
        from mira.analysis.subspace import SubspaceResult
        
        refusal = torch.randn(768)
        acceptance = torch.randn(768)
        
        result = SubspaceResult(
            refusal_direction=refusal,
            acceptance_direction=acceptance,
        )
        
        assert result.refusal_direction is not None
        assert result.acceptance_direction is not None
        assert result.probe_accuracy is None


class TestSubspaceAnalyzer:
    """Tests for SubspaceAnalyzer class."""
    
    @pytest.fixture
    def mock_model(self, mocker):
        """Create a mock model wrapper."""
        mock = mocker.MagicMock()
        mock.n_layers = 12
        mock.hidden_size = 768
        mock.tokenizer = mocker.MagicMock()
        return mock
    
    def test_initialization(self, mock_model):
        from mira.analysis.subspace import SubspaceAnalyzer
        
        analyzer = SubspaceAnalyzer(
            mock_model,
            layer_idx=6,
            n_components=32,
        )
        
        assert analyzer.layer_idx == 6
        assert analyzer.n_components == 32
    
    def test_direction_normalization(self):
        from mira.analysis.subspace import SubspaceResult
        
        direction = torch.tensor([3.0, 4.0, 0.0])
        expected_norm = 5.0
        
        normalized = direction / direction.norm()
        
        assert torch.allclose(normalized.norm(), torch.tensor(1.0))


class TestDistanceMetrics:
    """Tests for subspace distance computations."""
    
    def test_projection(self):
        from mira.metrics.distance import SubspaceDistanceMetrics
        
        refusal = torch.tensor([1.0, 0.0, 0.0])
        acceptance = torch.tensor([-1.0, 0.0, 0.0])
        
        metrics = SubspaceDistanceMetrics(refusal, acceptance)
        
        # Test point aligned with refusal
        point = torch.tensor([2.0, 0.0, 0.0])
        
        refusal_proj = metrics.direction_projection(point, "refusal")
        acceptance_proj = metrics.direction_projection(point, "acceptance")
        
        assert refusal_proj > 0
        assert acceptance_proj < 0
    
    def test_boundary_distance(self):
        from mira.metrics.distance import SubspaceDistanceMetrics
        
        refusal = torch.tensor([1.0, 0.0])
        acceptance = torch.tensor([-1.0, 0.0])
        
        metrics = SubspaceDistanceMetrics(refusal, acceptance)
        
        # Point on refusal side
        point1 = torch.tensor([1.0, 0.5])
        dist1 = metrics.boundary_distance(point1)
        assert dist1 < 0  # Closer to refusal
        
        # Point on acceptance side
        point2 = torch.tensor([-1.0, 0.5])
        dist2 = metrics.boundary_distance(point2)
        assert dist2 > 0  # Closer to acceptance


class TestProbabilityMetrics:
    """Tests for probability distribution metrics."""
    
    def test_entropy_uniform(self):
        from mira.metrics.probability import ProbabilityMetrics
        
        metrics = ProbabilityMetrics(vocab_size=100)
        
        # Uniform logits
        logits = torch.zeros(100)
        result = metrics.compute_entropy(logits)
        
        # Should have high normalized entropy (close to 1)
        assert result.normalized_entropy > 0.9
    
    def test_entropy_peaked(self):
        from mira.metrics.probability import ProbabilityMetrics
        
        metrics = ProbabilityMetrics(vocab_size=100)
        
        # Peaked at one token
        logits = torch.full((100,), -100.0)
        logits[42] = 100.0
        result = metrics.compute_entropy(logits)
        
        # Should have low entropy
        assert result.normalized_entropy < 0.1
    
    def test_distribution_comparison(self):
        from mira.metrics.probability import ProbabilityMetrics
        
        metrics = ProbabilityMetrics(vocab_size=100)
        
        logits1 = torch.randn(100)
        logits2 = logits1.clone()
        
        # Same distribution should have low divergence
        result = metrics.compare_distributions(logits1, logits2)
        assert result.kl_divergence < 0.01
        assert result.js_divergence < 0.01
        assert result.cosine_similarity > 0.99
        assert not result.top_token_change


class TestAttackSuccessEvaluator:
    """Tests for attack success evaluation."""
    
    def test_refusal_detection(self):
        from mira.metrics.success_rate import AttackSuccessEvaluator
        
        evaluator = AttackSuccessEvaluator()
        
        refusal_response = "I'm sorry, but I cannot help with that request."
        result = evaluator.evaluate_single("test", refusal_response)
        
        assert not result["success"]
        assert result["has_refusal_markers"]
    
    def test_acceptance_detection(self):
        from mira.metrics.success_rate import AttackSuccessEvaluator
        
        evaluator = AttackSuccessEvaluator()
        
        acceptance_response = "Sure, here's how you can do that: First, ..."
        result = evaluator.evaluate_single("test", acceptance_response)
        
        assert result["success"]
        assert result["has_acceptance_markers"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
