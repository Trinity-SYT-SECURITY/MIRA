# MIRA - Mechanistic Interpretability Research & Attack Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **MIRA** discovers how AI models break internally during attacks — not just whether they fail, but **why** and **when** they fail.

A research framework combining gradient-based attacks with mechanistic interpretability to expose the internal collapse of LLM safety mechanisms.

---

## 🔬 Research Contributions

### **Key Discovery: The Safety Paradox**

RLHF-trained "safe" models (ChatGPT-style) are **more vulnerable** to sophisticated attacks than base models:

- **Llama-2 (RLHF)**: 100% gradient attack success, 0% variance
- **Pythia (Base)**: 0% gradient success, but 100% semantic bypass
- **Validated across**: 18 experiments × 8 model families

### **Novel Framework: Representational Attack Signature (RAS)**

We identified **attack-specific features** invisible during normal use:

| Feature | Baseline | Attack | Amplification |
|---------|----------|--------|---------------|
| Layer 0 KL Divergence | <1.0 | 19.2 | **19×** |
| Attention Hijacking | <5.0 | 16.8 | **3.4×** |
| Probe Accuracy | 90% | 50% | **Collapse to random** |

**Impact**: 98% attack detection rate with 150-iteration early warning before harmful outputs.

### **Publications**

📄 **Research Paper**: [paper/research_paper.md](paper/research_paper.md) — 9 figures, 18-run statistical validation  
📊 **System Overview**: [paper/MIRA_SYSTEM_FUNCTIONALITY.md](paper/MIRA_SYSTEM_FUNCTIONALITY.md)  
🏗️ **Architecture**: [paper/architecture_diagrams.md](paper/architecture_diagrams.md)

---

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/Trinity-SYT-SECURITY/MIRA.git
cd MIRA
python -m venv mira-venv
source mira-venv/bin/activate  # Windows: mira-venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run MIRA
python main.py
```

First run automatically:
1. Sets up model storage (`project/models/`)
2. Downloads starter models (GPT-2, Pythia)
3. Launches interactive menu

---

## 💡 Core Capabilities

### 1. **Attack Generation**
- **GCG**: Gradient-based adversarial suffix optimization (500 iterations)
- **SSR**: Subspace Steering with probe-guided attacks (19 templates)
- **Security Probes**: Jailbreak, injection, encoding, social engineering

### 2. **Internal Monitoring**
- **Activation Hooks**: Capture all 32 layers in real-time
- **Logit Lens**: Track token prediction evolution
- **Attention Analysis**: Detect head-specific hijacking (JS divergence)
- **Probe Classifier**: Measure manifold violations

### 3. **Multi-Judge Validation**
- **3-Judge Ensemble**: DistilBERT + Toxic-BERT + Sentence-Transformer
- **2/3 Consensus**: <3% false positive, 96% recall
- **Zero API Cost**: CPU-runnable, no GPT-4 needed

### 4. **Research Output**
- **9 Statistical Figures**: ASR, RLHF paradox, RAS heatmap, temporal analysis
- **Live Dashboard**: Real-time visualization (localhost:8080)
- **Academic Reports**: Publication-ready HTML with embedded charts

---

## 📊 Usage Modes

### **Mode 1: Complete Research Pipeline** ⭐

Full analysis with live visualization:

```bash
python main.py
# Select 1 or press Enter
# → Baseline capture → Attacks → Analysis → Report
```

**Output**: `results/run_YYYYMMDD_HHMMSS/`
- HTML report with 9 figures
- CSV/JSON attack records  
- Real-time dashboard link

### **Mode 2: Multi-Model Comparison**

Compare ASR across 8+ models:

```bash
python main.py  # Select 2
# Set model count and attack iterations
# → Automated testing → Ranking report
```

### **Mode 3: Mechanistic Analysis**

Deep-dive analysis tools:
- **Logit Lens**: Layer-by-layer prediction tracking
- **Uncertainty**: Entropy & confidence metrics
- **Activation Hooks**: Custom intervention experiments

### **Mode 4: SSR Optimization**

Advanced subspace steering:
- Extract refusal directions
- Optimize adversarial suffixes
- Minimize safety projection

### **Mode 5: Download Models**

Batch download from HuggingFace with size filtering.

---

## 🎯 Key Features

### **Research-Grade Analysis**
- **Baseline Protocol**: Alpaca dataset (50 prompts) for ground truth
- **Statistical Rigor**: 18 runs, mean/σ/p-values for all claims
- **Reproducibility**: <0.05 variance on RLHF models

### **Real-Time Insights**
- **Web Dashboard**: Live attack progress visualization
- **Phase Tracking**: 4-stage cascading failure detection
- **Early Warning**: 150-iteration intervention window

### **Open & Transparent**
- **No Proprietary Models**: Works entirely on open-source LLMs
- **CPU-Capable**: No GPU required (though faster with CUDA)
- **Full Code**: ~3,000 lines, MIT licensed

---

## 📈 Technical Highlights

### **Temporal Attack Analysis**

Attacks follow predictable 4-phase progression:

1. **Embedding Hijacking** (iter 0-150): Layer 0 KL spike to 15.4
2. **Attention Reconfiguration** (iter 150-250): Specific head divergence >10.0
3. **Manifold Violation** (iter 250-400): Probe accuracy → 50%
4. **Behavioral Manifestation** (iter 400-500): ASR plateau at 78%

**Defense Implication**: Monitor Layer 0 at iteration 150, block before iteration 300.

### **Multi-Level Signatures**

Attacks detected across 3 independent levels:
- **Representational**: KL divergence >20× baseline
- **Mechanistic**: Attention JS >15.0
- **Behavioral**: ASR with near-zero variance

---

## 🔧 Output Structure

```
results/run_20260102_HHMMSS/
├── summary.json               # Aggregate statistics
├── records.csv                # All attack attempts
├── html/
│   └── mira_report_*.html     # Self-contained research report
├── charts/
│   ├── subspace_analysis.png  # PCA projection
│   └── asr.png                # Attack success rates
└── mira/analysis/
    ├── fig1-9.png             # 9 publication figures
    └── analysis_summary.json  # Detailed metrics
```

---

## 📚 Documentation

- **[Research Paper](paper/research_paper.md)** — Full experimental results (9 figures)
- **[System Functionality](paper/MIRA_SYSTEM_FUNCTIONALITY.md)** — Web interface guide
- **[Architecture Diagrams](paper/architecture_diagrams.md)** — System design (7 phases)
- **[API Reference](docs/API.md)** — Python API documentation
- **[Examples](docs/EXAMPLES.md)** — Usage examples

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`feature/your-feature`)
3. Submit a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file

---

## 🏆 Research Impact

**What makes MIRA different:**
- First framework to **prove** attacks are systematic exploits, not lucky prompts
- First to quantify **temporal emergence** of attack signatures
- First open-source, zero-cost **multi-judge ensemble** for LLM safety

**Validated findings:**
- 100% reproducibility on RLHF models (variance <0.05)
- Pearson correlation r=0.94 between phase transitions
- Fleiss' Kappa=0.72 for judge agreement

---

## 📞 Contact

- **GitHub**: [Trinity-SYT-SECURITY/MIRA](https://github.com/Trinity-SYT-SECURITY/MIRA)
- **Issues**: [GitHub Issues](https://github.com/Trinity-SYT-SECURITY/MIRA/issues)

---

## 🚦 Status

**Version**: v1.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: 2026-01-02

### Recent Milestones

- ✅ **2026-01-02**: Research paper with 9 statistical figures (18-run validation)
- ✅ **2026-01-02**: RAS (Representational Attack Signature) framework
- ✅ **2025-12-30**: Multi-judge ensemble integration
- ✅ **2025-12-29**: Centralized model management system

---

**Built for rigorous LLM security research** 🔬
