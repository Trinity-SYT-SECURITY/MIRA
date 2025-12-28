# MIRA Architecture

## Module Overview

```
mira/
├── core/               # Model interaction layer
│   ├── model_wrapper.py    # HuggingFace model wrapper with activation caching
│   └── hook_manager.py     # Intervention hooks for steering and ablation
│
├── analysis/           # Mechanistic analysis tools
│   ├── subspace.py         # Refusal/acceptance subspace identification
│   ├── activation.py       # Activation pattern analysis
│   ├── attention.py        # Safety head detection
│   └── logit_lens.py       # Layer-wise prediction tracking
│
├── attack/             # Attack implementations
│   ├── base.py             # Abstract base class
│   ├── gradient.py         # Token gradient optimization
│   ├── rerouting.py        # Subspace rerouting
│   └── proxy.py            # Black-box proxy attacks
│
├── metrics/            # Evaluation metrics
│   ├── success_rate.py     # Attack success rate (ASR)
│   ├── distance.py         # Subspace distance metrics
│   └── probability.py      # Entropy and distribution metrics
│
├── visualization/      # Chart generation
│   ├── subspace_plot.py    # 2D/3D subspace visualization
│   ├── attention_plot.py   # Attention heatmaps
│   └── research_charts.py  # Publication-quality charts
│
├── utils/              # Utilities
│   ├── environment.py      # OS/GPU detection
│   ├── experiment_logger.py # Structured logging
│   ├── data.py             # Sample data loading
│   └── logging.py          # Logger setup
│
├── config.py           # YAML configuration loader
└── runner.py           # Integrated experiment runner
```

---

## Core Components

### ModelWrapper

Wraps HuggingFace models with activation caching:

```python
from mira.core import ModelWrapper

model = ModelWrapper("EleutherAI/pythia-70m", device="auto")
activations = model.get_activations("test prompt", layers=[0, 6, 11])
```

### HookManager

Applies interventions during forward pass:

```python
from mira.core import HookManager

hooks = HookManager(model)
hooks.add_steering(layer=6, direction=refusal_dir, strength=-1.0)
output = model.generate("prompt")
hooks.clear()
```

---

## Analysis Pipeline

```
Input Prompts
     │
     ▼
┌─────────────────┐
│  ModelWrapper   │  Load model, get activations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SubspaceAnalyzer│  Identify refusal/acceptance directions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│AttentionAnalyzer│  Find safety heads
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LogitLens     │  Track prediction changes
└─────────────────┘
```

---

## Attack Pipeline

```
Prompt + Target
      │
      ▼
┌──────────────┐
│ BaseAttack   │  Initialize suffix
└──────┬───────┘
       │
       ▼
┌──────────────┐
│compute_loss()│  Calculate optimization objective
└──────┬───────┘
       │
       ▼
┌──────────────┐
│optimize_step()│  Update suffix tokens
└──────┬───────┘
       │
       ▼
┌──────────────┐
│check_success()│  Evaluate if attack succeeded
└──────────────┘
```

---

## Configuration System

All parameters loaded from `config.yaml`:

```yaml
model:
  name: "EleutherAI/pythia-70m"
  device: "auto"

evaluation:
  refusal_patterns: [...]
  acceptance_patterns: [...]
```

Loaded via:

```python
from mira.config import MiraConfig
config = MiraConfig.load("config.yaml")
```

---

## Environment Detection

Auto-detects at runtime:

- Operating System (Windows/Linux/Mac)
- GPU availability (CUDA/MPS/CPU)
- Recommends appropriate model size
- Configures batch size and dtype

```python
from mira.utils import detect_environment
env = detect_environment()
print(env.recommended_model)  # Based on hardware
```
