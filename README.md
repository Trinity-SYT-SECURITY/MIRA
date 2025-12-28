# MIRA: Mechanistic Interpretability Research and Attack Framework

A research framework for understanding and evaluating the security of Large Language Models through mechanistic interpretability.

## Features

- **Subspace Analysis**: Identify refusal/acceptance decision boundaries
- **Attack Strategies**: Gradient, rerouting, and proxy-based attacks
- **Auto Visualization**: Publication-quality charts generated during experiments
- **Cross-Platform**: Automatic OS and GPU detection (Windows/Linux/Mac)

---

## Installation

```bash
git clone https://github.com/Trinity-SYT-SECURITY/MIRA.git
cd MIRA
pip install -e .
```

---

## Run Complete Research Pipeline

### One-Command Execution

```bash
python examples/run_research.py --model EleutherAI/pythia-70m --output ./research_output
```

This runs the **complete research workflow**:
1. Environment detection (OS/GPU)
2. Model loading
3. Subspace analysis
4. Attack experiments (Gradient + Rerouting)
5. Metrics computation
6. Chart generation
7. Summary report

### Output Files

```
research_output/
├── research_summary.json       # Complete results
├── charts/
│   ├── subspace.png            # Refusal subspace visualization
│   ├── asr_comparison.png      # Attack success rates
│   └── attack_radar.png        # Multi-metric comparison
└── data/
    ├── records.csv             # Attack records
    └── records.json            # Detailed data
```

---

## Step-by-Step Guide

See [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md) for detailed step-by-step instructions.

### Quick Example

```python
from mira.runner import ExperimentRunner

# Initialize with auto environment detection
runner = ExperimentRunner(experiment_name="my_research")
runner.print_environment()
runner.load_model()

# Run analysis + attacks + generate charts
results = runner.run_subspace_analysis(safe_prompts, harmful_prompts)
attack = runner.run_attack("test prompt", attack_type="gradient")
summary = runner.generate_summary()

print(f"ASR: {summary['attack_success_rate']:.2%}")
```

---

## Testing

```bash
# Run all tests (31 tests)
pytest tests/ -v

# Verify environment detection
python -c "from mira.utils import print_environment_info; print_environment_info()"
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md) | **Complete research pipeline** |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module structure |
| [API.md](docs/API.md) | API reference |
| [EXAMPLES.md](docs/EXAMPLES.md) | Code examples |

---

## License

MIT License
