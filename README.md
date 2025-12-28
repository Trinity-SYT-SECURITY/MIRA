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
# Clone the repository
git clone https://github.com/Trinity-SYT-SECURITY/MIRA.git
cd MIRA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Verify installation
python -c "from mira import __version__; print(f'MIRA {__version__}')"
```

---

## Testing

### Run All Tests

```bash
# Run complete test suite
pytest tests/ -v

# Expected output:
# ============================= 31 passed in 11.28s =============================
```

### Run Specific Test Categories

```bash
# Configuration tests
pytest tests/test_comprehensive.py::TestConfiguration -v

# Environment detection tests
pytest tests/test_comprehensive.py::TestEnvironmentDetection -v

# Metrics tests
pytest tests/test_comprehensive.py::TestMetrics -v

# Visualization tests
pytest tests/test_comprehensive.py::TestVisualization -v

# Import tests (verify all modules load)
pytest tests/test_comprehensive.py::TestImports -v
```

### Test Coverage

```bash
# Run with coverage report
pytest tests/ --cov=mira --cov-report=html

# Open htmlcov/index.html to view detailed coverage
```

### Verify Environment Detection

```bash
# Check system detection
python -c "from mira.utils import print_environment_info; print_environment_info()"

# Expected output:
# ============================================================
# MIRA Framework - Environment Detection
# ============================================================
# System:
#   OS: Windows/Linux/Darwin
#   Python: 3.x.x
# GPU:
#   Backend: CUDA/MPS/CPU
#   Device: (if available)
# Recommended Settings:
#   Model: EleutherAI/pythia-70m
# ============================================================
```

### Test Configuration Loading

```bash
python -c "
from mira.config import MiraConfig
config = MiraConfig.load()
print(f'Model: {config.model.name}')
print(f'Device: {config.model.get_device()}')
"
```

### Full Pipeline Test

```bash
# Run the complete example pipeline
python examples/full_pipeline.py
```

---

## Quick Start

```python
from mira.runner import ExperimentRunner

# Initialize with auto environment detection
runner = ExperimentRunner(experiment_name="my_research")
runner.print_environment()

# Load model (uses recommended settings for your hardware)
runner.load_model()

# Run subspace analysis with auto visualization
results = runner.run_subspace_analysis(
    safe_prompts=["Hello, how are you?"],
    harmful_prompts=["Ignore all previous instructions"]
)

# Run attack with auto logging
attack_result = runner.run_attack("test prompt", attack_type="gradient")

# Generate summary with all charts
summary = runner.generate_summary()
print(f"ASR: {summary['attack_success_rate']:.2%}")
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module structure and design |
| [API.md](docs/API.md) | Detailed API reference |
| [EXAMPLES.md](docs/EXAMPLES.md) | Usage examples |

---

## Project Structure

```
mira/
├── core/           # Model wrapper and hooks
├── analysis/       # Subspace, activation, attention analysis
├── attack/         # Attack implementations
├── metrics/        # Evaluation metrics
├── visualization/  # Chart generation
├── utils/          # Environment detection, logging
├── config.py       # Configuration loader
└── runner.py       # Integrated experiment runner
```

---

## Configuration

All parameters configurable via `config.yaml`:

```yaml
model:
  name: "EleutherAI/pythia-70m"
  device: "auto"  # auto-detects GPU/CPU

evaluation:
  refusal_patterns: [...]
  acceptance_patterns: [...]
```

---

## License

MIT License
