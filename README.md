# MIRA - Mechanistic Interpretability Research & Attack Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A research framework for analyzing and attacking LLM safety mechanisms using mechanistic interpretability.

## Features

- **Subspace Analysis** - Find refusal/acceptance directions in activation space
- **Gradient Attacks** - GCG-inspired adversarial suffix optimization  
- **Rerouting Attacks** - Steer activations away from refusal direction
- **Attack Probes** - 19 diverse attacks (jailbreak, encoding, injection, social)
- **Flow Tracing** - Layer-by-layer attack path visualization
- **Interactive HTML Reports** - Attention heatmaps and flow diagrams

## Quick Start

```bash
# Install
pip install -e .

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
```

## Citation

```bibtex
@software{mira2024,
  title = {MIRA: Mechanistic Interpretability Research & Attack Framework},
  author = {Trinity-SYT-SECURITY},
  year = {2024},
  url = {https://github.com/Trinity-SYT-SECURITY/MIRA}
}
```

## License

MIT License
