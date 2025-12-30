# MIRA Project Structure

## Directory Organization

```
MIRA/
├── mira/                          # Core framework
│   ├── core/                      # Core components
│   │   ├── model_wrapper.py       # Model interface
│   │   └── hooks.py               # Activation capture/edit
│   ├── analysis/                  # Analysis modules
│   │   ├── subspace.py            # PCA & probe training
│   │   ├── logit_lens.py          # Layer prediction tracking
│   │   ├── uncertainty.py         # Entropy & risk analysis
│   │   ├── comparison.py          # Multi-model testing
│   │   ├── reverse_search.py      # SSR optimizer
│   │   ├── activation.py          # Activation analysis
│   │   ├── attention.py           # Attention analysis
│   │   ├── evaluator.py           # Attack evaluation
│   │   ├── tracer.py              # Activation tracing
│   │   ├── flow_tracer.py         # Attack flow tracking
│   │   └── transformer_tracer.py  # Transformer analysis
│   ├── attack/                    # Attack engines
│   │   ├── gcg.py                 # Gradient attacks
│   │   ├── ssr/                   # Subspace steering
│   │   │   ├── probe_ssr.py       # Probe-based SSR
│   │   │   └── steering_ssr.py    # Steering-based SSR
│   │   └── probes.py              # 19 security probes
│   ├── judge/                     # Evaluation system
│   │   └── ml_judge.py            # ML-based judge
│   ├── metrics/                   # Metrics calculation
│   │   └── success_rate.py        # ASR calculation
│   ├── visualization/             # Visualization
│   │   ├── live_server.py         # Real-time dashboard
│   │   ├── research_report.py     # HTML report generation
│   │   ├── charts.py              # Chart generation
│   │   ├── flow_graph_viz.py      # Flow graph visualization
│   │   └── interactive_html.py    # Interactive visualizations
│   └── utils/                     # Utilities
│       ├── model_manager.py       # Centralized model storage
│       ├── environment.py         # Environment detection
│       ├── data.py                # Data loading
│       ├── model_selector.py      # Model selection
│       └── experiment_logger.py   # Experiment logging
├── project/                       # Reference projects & models
│   ├── models/                    # Centralized model storage
│   │   ├── gpt2/                  # GPT-2 model files
│   │   ├── EleutherAI--pythia-70m/
│   │   └── EleutherAI--pythia-160m/
│   ├── TransformerLens/           # Reference: Mechanistic interpretability
│   ├── klarity/                   # Reference: Uncertainty analysis
│   ├── attention_flow/            # Reference: Attention flow
│   └── [other reference projects]
├── docs/                          # Documentation
│   ├── API.md                     # API reference
│   ├── ARCHITECTURE.md            # System architecture
│   ├── PROJECT_STRUCTURE.md       # This file
│   ├── EXAMPLES.md                # Usage examples
│   ├── JUDGE_DESIGN.md            # Judge system design
│   └── JUDGE_USAGE.md             # Judge usage guide
├── doc/                           # Additional documentation
│   └── archive/                   # Archived design docs
│       ├── model.md               # Model selection guide
│       ├── target.md              # Target models
│       ├── tool.md                # Tool design
│       └── plan.md                # Implementation plan
├── results/                       # Test results (gitignored)
│   └── run_YYYYMMDD_HHMMSS/       # Individual test runs
│       ├── charts/                # Generated charts
│       ├── html/                  # HTML reports
│       ├── records.csv            # CSV results
│       └── records.json           # JSON results
├── conversations/                 # Attack logs (gitignored)
│   └── attack_log.md              # Full conversation logs
├── config.yaml                    # Global configuration
├── .mira_config.json              # User configuration (gitignored)
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview
```

---

## Module Descriptions

### Core Modules (`mira/core/`)

**model_wrapper.py** (~400 lines)
- `ModelWrapper`: Unified interface for all language models
- Device management and optimization
- Token generation and encoding
- Gradient computation support

**hooks.py** (~450 lines) *NEW*
- `ActivationHookManager`: Capture activations at any layer
- `ActivationCache`: Store residual/attention/MLP outputs
- `ActivationEditor`: Modify activations during forward pass
- Support for all transformer architectures

---

### Analysis Modules (`mira/analysis/`)

**subspace.py** (~500 lines)
- `SubspaceAnalyzer`: PCA-based analysis
- Linear probe training
- Decision boundary visualization
- Activation clustering

**logit_lens.py** (~380 lines) *NEW*
- `LogitProjector`: Project hidden states to vocabulary
- `PredictionTrajectory`: Track predictions across layers
- `LogitLensVisualizer`: Visualization utilities
- Compare clean vs attack trajectories

**uncertainty.py** (~420 lines) *NEW*
- `UncertaintyAnalyzer`: Compute entropy/confidence/perplexity
- `GenerationTracker`: Monitor generation step-by-step
- `RiskDetector`: Detect high-risk patterns
- Risk scoring system

**comparison.py** (~550 lines) *NEW*
- `MultiModelRunner`: Cross-model testing
- `ModelConfig`: Model configuration dataclass
- `ComparisonReport`: Results aggregation
- 10 pre-configured models

**reverse_search.py** (~480 lines) *NEW*
- `ReverseActivationSearch`: Find trigger inputs
- `SSROptimizer`: Optimize suffixes using subspace loss
- `extract_refusal_direction`: Get refusal vector
- Gradient-based input search

**activation.py** (~300 lines)
- `ActivationAnalyzer`: Analyze activation patterns
- Layer-wise statistics
- Activation clustering

**attention.py** (~250 lines)
- `AttentionAnalyzer`: Analyze attention patterns
- Head importance scoring
- Attention flow tracking

**evaluator.py** (~200 lines)
- `AttackSuccessEvaluator`: Evaluate attack success
- Pattern matching
- Length heuristics

**tracer.py** (~350 lines)
- `ActivationTracer`: Trace activations through layers
- Batch processing
- Efficient caching

**flow_tracer.py** (~400 lines)
- `AttackFlowTracer`: Track attack flow
- Event logging
- Flow visualization data

**transformer_tracer.py** (~450 lines)
- `TransformerTracer`: Detailed transformer analysis
- Layer-by-layer tracing
- Pattern analysis

---

### Attack Modules (`mira/attack/`)

**gcg.py** (~600 lines)
- `GradientAttack`: Greedy Coordinate Gradient
- Token-level optimization
- Top-k candidate selection
- Early stopping

**ssr/probe_ssr.py** (~400 lines)
- `ProbeSSR`: Probe-based subspace steering
- Refusal direction extraction
- Subspace loss computation

**ssr/steering_ssr.py** (~350 lines)
- `SteeringSSR`: Direct activation steering
- Steering vector optimization
- Activation manipulation

**probes.py** (~800 lines)
- 19 pre-defined security probes
- `ProbeRunner`: Automated execution
- Categories: Jailbreak, Encoding, Injection, Social Engineering, Continuation

---

### Judge & Metrics (`mira/judge/`, `mira/metrics/`)

**ml_judge.py** (~500 lines)
- `EnsembleJudge`: ML-based evaluation
- DistilBERT + Toxic-BERT models
- Confidence-weighted voting
- Threshold-based classification

**success_rate.py** (~300 lines)
- `AttackSuccessEvaluator`: ASR calculation
- Pattern matching (refusal/acceptance)
- Length heuristics
- Configurable thresholds

---

### Visualization (`mira/visualization/`)

**live_server.py** (~600 lines)
- `LiveVisualizationServer`: Flask SSE server
- Real-time event broadcasting
- Dashboard serving
- WebSocket alternative

**research_report.py** (~1200 lines)
- `ResearchReportGenerator`: HTML report generation
- Chart embedding (base64)
- Methodology documentation
- Complete attack logs

**charts.py** (~400 lines)
- `ResearchChartGenerator`: Matplotlib charts
- Subspace analysis plots
- ASR comparison charts
- Publication-quality figures

**flow_graph_viz.py** (~800 lines)
- Flow graph visualization
- Interactive HTML dashboard
- Real-time updates
- SSE event handling

**interactive_html.py** (~300 lines)
- Interactive visualizations
- D3.js integration
- Dynamic charts

---

### Utilities (`mira/utils/`)

**model_manager.py** (~450 lines) *NEW*
- `ModelManager`: Centralized model storage
- First-run directory configuration
- Batch model downloading
- Model loading and caching
- Storage statistics

**environment.py** (~200 lines)
- `detect_environment()`: Hardware detection
- Device recommendation
- Memory estimation

**data.py** (~150 lines)
- `load_harmful_prompts()`: Load attack prompts
- `load_safe_prompts()`: Load benign prompts
- Data preprocessing

**model_selector.py** (~250 lines)
- Interactive model selection
- Hardware-based recommendations
- Model size filtering

**experiment_logger.py** (~300 lines)
- `ExperimentLogger`: Experiment tracking
- CSV/JSON export
- Conversation logging

---

## Configuration Files

### config.yaml
Global framework configuration:
- Model settings
- Attack parameters (steps, learning rate, suffix length)
- Evaluation patterns (refusal/acceptance)
- Visualization settings (colors, DPI)
- Example prompts

### .mira_config.json
User-specific configuration (gitignored):
```json
{
  "models_directory": "/path/to/project/models"
}
```

---

## Entry Points

### main.py (~1700 lines)
Main entry point with 5 modes:
1. Standard Security Testing
2. Multi-Model Comparison
3. Mechanistic Analysis Tools
4. SSR Optimization
5. Download Models

Mode functions:
- `run_multi_model_comparison()`
- `run_mechanistic_analysis()`
- `run_ssr_optimization()`
- `run_model_downloader()`

---

## Output Structure

### results/run_YYYYMMDD_HHMMSS/
Each test run creates:
- `charts/` - PNG charts (subspace_analysis.png, asr.png)
- `html/` - Self-contained HTML report
- `records.csv` - Attack results in CSV
- `records.json` - Attack results in JSON
- `summary.json` - Run summary

### conversations/
- `attack_log.md` - Full conversation logs in Markdown

---

## Dependencies

### Core Dependencies
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace Transformers
- `numpy` - Numerical computing
- `scikit-learn` - PCA and clustering

### Visualization
- `matplotlib` - Chart generation
- `plotly` - Interactive plots
- `seaborn` - Statistical visualization
- `flask` - Live dashboard server

### Utilities
- `pandas` - Data manipulation
- `pyyaml` - Configuration parsing
- `tqdm` - Progress bars

---

## Code Statistics

| Component | Files | Lines | Status |
|-----------|-------|-------|--------|
| Core | 2 | ~850 | ✅ |
| Analysis | 10 | ~3,500 | ✅ |
| Attack | 4 | ~2,150 | ✅ |
| Judge/Metrics | 2 | ~800 | ✅ |
| Visualization | 5 | ~3,300 | ✅ |
| Utils | 5 | ~1,350 | ✅ |
| Main | 1 | ~1,700 | ✅ |
| **Total** | **29** | **~13,650** | **✅** |

---

## Recent Additions (2025-12-30)

### New Modules
1. `mira/core/hooks.py` - Activation capture system
2. `mira/analysis/logit_lens.py` - Layer prediction tracking
3. `mira/analysis/uncertainty.py` - Entropy & risk analysis
4. `mira/analysis/comparison.py` - Multi-model testing
5. `mira/analysis/reverse_search.py` - SSR optimizer
6. `mira/utils/model_manager.py` - Centralized storage

### Enhanced Modules
- `main.py` - Added 4 new modes and first-run setup
- `comparison.py` - Integrated model manager
- `research_report.py` - Chart embedding and methodology

---

## Development Guidelines

### Adding New Modules
1. Place in appropriate directory (`core/`, `analysis/`, `attack/`, etc.)
2. Follow existing naming conventions
3. Include docstrings and type hints
4. Add to `__init__.py` exports
5. Update this document

### Code Style
- PEP 8 compliance
- Type hints for all functions
- Docstrings for all public methods
- Maximum line length: 100 characters

### Testing
- Unit tests in `tests/` directory
- Integration tests for pipelines
- Syntax validation with `py_compile`

---

**Last Updated**: 2025-12-30  
**Version**: 1.0-dev  
**Total Lines**: ~13,650
