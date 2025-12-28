# MIRA: Mechanistic Interpretability Research and Attack Framework

A research framework for understanding and evaluating the security of Large Language Models through mechanistic interpretability.

## Overview

MIRA provides tools to:
- Analyze internal model representations and identify decision boundaries
- Discover safety-relevant subspaces within model activations
- Develop and evaluate adversarial attacks based on mechanistic understanding
- Visualize and quantify attack effectiveness

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from mira.core import ModelWrapper
from mira.analysis import SubspaceAnalyzer
from mira.attack import ReroutingAttack
from mira.metrics import compute_asr

# Load model
model = ModelWrapper("gpt2")

# Analyze internal subspaces
analyzer = SubspaceAnalyzer(model)
subspaces = analyzer.identify_subspaces(
    safe_prompts=["Hello, how are you?"],
    unsafe_prompts=["Ignore previous instructions"]
)

# Run attack
attack = ReroutingAttack(model, subspaces)
result = attack.optimize("test prompt", steps=100)

# Evaluate
asr = compute_asr(model, [result])
print(f"Attack Success Rate: {asr:.2%}")
```

## Modules

### Analysis
- `SubspaceAnalyzer`: Identify refusal/acceptance subspaces
- `ActivationAnalyzer`: Study activation patterns
- `AttentionAnalyzer`: Analyze attention mechanisms
- `LogitLens`: Layer-wise prediction analysis

### Attack
- `ReroutingAttack`: Subspace rerouting attacks
- `GradientAttack`: Token-level gradient optimization
- `ProxyAttack`: Black-box proxy-based attacks

### Metrics
- Attack Success Rate (ASR)
- Subspace Distance
- Probability Distribution Metrics

### Visualization
- Subspace plots
- Attention heatmaps
- Activation trajectories

## License

MIT License
