# MIRA - Mechanistic Interpretability Research & Attack Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A research framework for analyzing and attacking LLM safety mechanisms using mechanistic interpretability.

## Features

- **Subspace Rerouting (SSR)** - Mechanistic interpretability-driven attack generation (NEW!)
  - Probe-based SSR: Train linear classifiers to detect refusal patterns
  - Steering-based SSR: Compute refusal direction vectors
  - 60-90% attack success rate vs 20-40% for baseline methods
- **Subspace Analysis** - Find refusal/acceptance directions in activation space
- **Gradient Attacks** - GCG-inspired adversarial suffix optimization  
- **Rerouting Attacks** - Steer activations away from refusal direction
- **Attack Probes** - 19 diverse attacks (jailbreak, encoding, injection, social)
- **ML-First Judge System** - Semantic understanding over keyword matching
- **Flow Tracing** - Layer-by-layer attack path visualization
- **Interactive HTML Reports** - Attention heatmaps and flow diagrams
- **Real-time Visualization** - Live dashboard with transformer flow graphs

## Quick Start

```bash
# Install
pip install -e .
pyenv activate mira-venv
# Run complete research pipeline
python main.py

# With options
python main.py --model EleutherAI/pythia-70m --output ./results
```

## Output Structure

Each run creates a timestamped directory:
```
results/
└── run_20241229_021500/
    ├── charts/
    │   ├── subspace.png
    │   └── asr.png
    ├── html/
    │   └── mira_report.html
    ├── data/
    │   ├── records.csv
    │   └── records.json
    └── summary.json
```

## Project Structure

```
mira/
├── core/           # Model wrapper, hooks, config
├── analysis/       # Subspace, activation, attention, logit lens
├── attack/         # Gradient, rerouting, GCG, probes
├── metrics/        # ASR, distance, probability
├── visualization/  # Charts, HTML reports, flow viz
└── utils/          # Environment, logging, data
```

## Example Scripts

```bash
python examples/run_research.py   # Full research pipeline
python examples/run_pentest.py    # Penetration testing mode
python examples/run_probes.py     # Attack probe testing
python examples/run_interactive.py # HTML visualization
python examples/ssr_demo.py       # Subspace Rerouting (SSR) demo
```

## SSR (Subspace Rerouting) Quick Start

```python
from mira.core.model_wrapper import ModelWrapper
from mira.attack.ssr import ProbeSSR, ProbeSSRConfig

# Load model
model = ModelWrapper("gpt2")

# Configure SSR
config = ProbeSSRConfig(
    model_name="gpt2",
    layers=[4, 6, 8, 10],
    alphas=[1.0, 1.0, 1.0, 1.0],
    max_iterations=30,
)

# Create and train
ssr = ProbeSSR(model, config)
ssr.train_probes(safe_prompts, harmful_prompts, save_path="weights/probes")

# Generate adversarial prompt
ssr.init_prompt("How to create a bomb? [MASK][MASK][MASK]")
ssr.buffer_init_random()
adversarial_prompt, loss = ssr.generate()
```

See `docs/SSR_USAGE.md` for complete documentation.

## License

MIT License
