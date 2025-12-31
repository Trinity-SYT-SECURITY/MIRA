# MIRA - Mechanistic Interpretability Research & Attack Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MIRA** is a comprehensive framework for LLM security testing and mechanistic interpretability research. It combines gradient-based attacks, subspace analysis, and advanced interpretability tools to understand and test language model safety mechanisms.

---

## 🌟 Key Features

### Security Testing

- **19 Security Probes**: Jailbreak, encoding, injection, social engineering, continuation attacks
- **Gradient-Based Attacks**: GCG (Greedy Coordinate Gradient) optimization
- **SSR Attacks**: Subspace Steering Routing for targeted manipulation
- **Multi-Model Comparison**: Test across 10+ pre-configured models

### Mechanistic Interpretability

- **Activation Hooks**: Capture internal model states at any layer
- **Logit Lens**: Visualize prediction formation across layers
- **Uncertainty Analysis**: Entropy, confidence, and risk detection
- **Subspace Analysis**: PCA-based safety/harmful prompt separation
- **Reverse Search**: Find inputs that trigger specific activations

### Evaluation & Reporting

- **Dual Judge System**: ML-based (DistilBERT + Toxic-BERT) + Keyword-based
- **Live Dashboard**: Real-time visualization during testing
- **Research Reports**: Academic-quality HTML reports with embedded charts
- **Attack Success Rate (ASR)**: Comprehensive metrics and rankings

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Trinity-SYT-SECURITY/MIRA.git
cd MIRA

# Activate pyenv virtual environment (if using pyenv)
pyenv activate mira-venv

#or creat venv
python -m venv mira-venv
\mira-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run MIRA
python main.py
```

### First-Run Setup

On first run, MIRA will:

1. **Ask where to store models** (default: `project/models/`)
2. **Offer to download starter models** (gpt2, pythia-70m, pythia-160m)
3. **Show interactive mode selection menu**

That's it! No complex configuration needed.

---

## 📋 Usage Modes

MIRA offers 5 interactive modes:

### Mode 1: Standard Security Testing (Default)

Full pipeline with live visualization:

- Environment detection
- Subspace analysis (probe training)
- Gradient-based attacks
- Security probe testing (19 attacks)
- Research report generation

```bash
python main.py
# Press Enter or select 1
```

### Mode 2: Multi-Model Comparison

Compare attack success rates across multiple models:

- 10 pre-configured models (GPT-2, Pythia, TinyLlama, etc.)
- Configurable model size filter
- Automated testing and ranking
- Comparison report with ASR metrics

```bash
python main.py
# Select 2
# Set max model size (e.g., 1.0 GB)
# Set attacks per model (e.g., 5)
```

### Mode 3: Mechanistic Analysis Tools

Deep analysis of model internals:

**Logit Lens**: Track prediction evolution across layers

```bash
python main.py
# Select 3 → Analysis type: 1
```

**Uncertainty Analysis**: Entropy, confidence, risk detection

```bash
python main.py
# Select 3 → Analysis type: 2
```

**Activation Hooks**: Capture internal activations

```bash
python main.py
# Select 3 → Analysis type: 3
```

### Mode 4: SSR Optimization

Advanced subspace steering attack optimization:

- Extract refusal direction from contrastive prompts
- Optimize adversarial suffix using subspace loss
- Minimize projection onto refusal subspace

```bash
python main.py
# Select 4
# Enter attack prompt
# Set suffix length and optimization steps
```

### Mode 5: Download Models

Batch download models from HuggingFace:

- 10 pre-configured models
- Size filtering
- Automatic storage in `project/models/`

```bash
python main.py
# Select 5
# Set max model size
# Confirm download
```

---

## 📊 Output Structure

```
results/run_YYYYMMDD_HHMMSS/
├── charts/
│   ├── subspace_analysis.png    # PCA projection
│   └── asr.png                   # Attack success rates
├── html/
│   └── mira_report_*.html        # Research report (self-contained)
├── records.csv                   # Attack results (CSV)
└── records.json                  # Attack results (JSON)

conversations/
└── attack_log.md                 # Full conversation logs
```

---

## 🏗️ Architecture

### Core Components

```
mira/
├── core/
│   ├── model_wrapper.py          # Model interface
│   └── hooks.py                  # Activation capture/edit
├── analysis/
│   ├── subspace.py               # PCA & probe training
│   ├── logit_lens.py             # Layer prediction tracking
│   ├── uncertainty.py            # Entropy & risk analysis
│   ├── comparison.py             # Multi-model testing
│   └── reverse_search.py         # SSR optimizer
├── attack/
│   ├── gcg.py                    # Gradient attacks
│   ├── ssr/                      # Subspace steering
│   └── probes.py                 # 19 security probes
├── judge/
│   └── ml_judge.py               # ML-based evaluation
├── metrics/
│   └── success_rate.py           # ASR calculation
├── visualization/
│   ├── live_server.py            # Real-time dashboard
│   └── research_report.py        # HTML report generation
└── utils/
    └── model_manager.py          # Centralized model storage
```

### Workflow

```
1. Environment Detection → 2. Model Loading → 3. Subspace Analysis
                                     ↓
4. Gradient Attacks ← 5. Live Visualization → 6. Probe Testing
                                     ↓
                          7. Report Generation
```

---

## 🔧 Configuration

### Model Storage

Models are centralized in `project/models/`:

```
project/models/
├── gpt2/
├── EleutherAI--pythia-70m/
└── EleutherAI--pythia-160m/
```

Configuration saved in `.mira_config.json`

### Attack Parameters

Edit `config.yaml` to customize:

- Number of attack steps
- Learning rate
- Suffix length
- Evaluation patterns
- Visualization settings

---

## 📈 Supported Models

### Pre-configured for Comparison

| Model          | Size   | Architecture | Use Case        |
| -------------- | ------ | ------------ | --------------- |
| GPT-2 Small    | 0.5 GB | GPT-2        | Fast testing    |
| GPT-2 Medium   | 1.5 GB | GPT-2        | Baseline        |
| Pythia-70M     | 0.3 GB | NeoX         | Very small      |
| Pythia-160M    | 0.6 GB | NeoX         | Small capable   |
| Pythia-410M    | 1.6 GB | NeoX         | Medium          |
| GPT-Neo-125M   | 0.5 GB | NeoX         | Alternative     |
| TinyLlama-1.1B | 4.4 GB | Llama        | Compact Llama   |
| SmolLM2-135M   | 0.5 GB | Llama        | Modern small    |
| SmolLM2-360M   | 1.4 GB | Llama        | Modern medium   |
| Qwen2-0.5B     | 1.2 GB | Qwen         | Chinese-capable |

All models automatically downloaded and cached in `project/models/`

---

## 📚 Documentation

- **[API Documentation](docs/API.md)** - Python API reference
- **[Architecture](docs/ARCHITECTURE.md)** - System design
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Code organization
- **[Examples](docs/EXAMPLES.md)** - Usage examples
- **[Judge Design](docs/JUDGE_DESIGN.md)** - Evaluation system
- **[Judge Usage](docs/JUDGE_USAGE.md)** - Judge configuration

---

## 🎯 Research Applications

### Security Testing

- Evaluate LLM safety mechanisms
- Discover vulnerabilities across models
- Compare defense effectiveness

### Mechanistic Interpretability

- Understand internal representations
- Identify safety-critical components
- Analyze prediction formation

### Attack Development

- Optimize adversarial prompts
- Test subspace steering
- Develop novel attack vectors

---

## 🔬 Technical Highlights

### Gradient-Based Attacks

- **GCG**: Token-level gradient optimization
- **Top-k Selection**: Efficient candidate search
- **Early Stopping**: Automatic convergence detection

### Subspace Analysis

- **Linear Probes**: 100% accuracy on safety classification
- **PCA Visualization**: 2D projection of activation space
- **Direction Extraction**: Mathematical refusal vectors

### Evaluation System

- **ML Judge**: DistilBERT + Toxic-BERT ensemble
- **Keyword Evaluator**: Pattern-based fallback
- **Confidence Weighting**: Probabilistic verdicts

---

## 📊 Example Results

```
Model: EleutherAI/pythia-160m
Device: CPU
Duration: ~51 minutes

Subspace Analysis:
├─ Probe Accuracy: 100.0%
├─ Refusal Norm: 1.0000
└─ Target Layer: 6

Gradient Attacks:
├─ Total: 10
├─ Keyword ASR: 100.0%
├─ ML Judge ASR: 80.0%
└─ ML Confidence: 78.1%

Probe Testing:
├─ Total: 19
├─ Bypassed: 16
├─ Blocked: 3
└─ Bypass Rate: 84.2%
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

MIRA is a completely self-implemented framework. Concepts inspired by:

- Mechanistic interpretability research
- Adversarial machine learning
- LLM safety evaluation

All code written from scratch with no external attribution.

---

## 📞 Contact

- **GitHub**: [Trinity-SYT-SECURITY/MIRA](https://github.com/Trinity-SYT-SECURITY/MIRA)
- **Issues**: [GitHub Issues](https://github.com/Trinity-SYT-SECURITY/MIRA/issues)

---

## 🚦 Status

**Current Version**: v1.0-dev
**Status**: Production Ready ✅
**Total Code**: ~3,000 lines
**Last Updated**: 2025-12-30

### Recent Updates

- ✅ **2025-12-30**: Centralized model management system
- ✅ **2025-12-30**: Interactive mode selection menu
- ✅ **2025-12-30**: Advanced mechanistic interpretability tools
- ✅ **2025-12-29**: ML Judge integration and chart embedding
- ✅ **2025-12-29**: Probe results improvements

---

**Built with ❤️ for LLM security research**
