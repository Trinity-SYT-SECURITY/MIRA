# MIRA Project Structure

## Directory Organization

```
MIRA/
├── mira/                    # Main package
│   ├── core/               # Core functionality
│   │   ├── model_wrapper.py    # Model loading and management
│   │   └── hook_manager.py     # Activation hooking
│   ├── analysis/           # Analysis modules
│   │   ├── subspace.py         # Subspace analysis
│   │   ├── activation.py        # Activation analysis
│   │   ├── attention.py         # Attention analysis
│   │   ├── transformer_tracer.py # Transformer tracing
│   │   └── subspace/            # SSR subspace weights
│   ├── attack/             # Attack implementations
│   │   ├── base.py              # Base attack class
│   │   ├── gcg.py               # GCG attack
│   │   ├── gradient.py          # Gradient-based attacks
│   │   ├── rerouting.py         # Rerouting attacks
│   │   ├── probes.py            # Attack probes
│   │   └── ssr/                 # Subspace Rerouting (SSR)
│   │       ├── core.py          # Core SSR algorithm
│   │       ├── probe_ssr.py     # Probe-based SSR
│   │       ├── steering_ssr.py  # Steering-based SSR
│   │       └── config.py        # SSR configuration
│   ├── judge/              # Judge system
│   │   ├── ml_judge.py          # ML-based judge
│   │   ├── ensemble.py          # Ensemble judge
│   │   └── config.py            # Judge configuration
│   ├── metrics/             # Metrics and evaluation
│   │   ├── success_rate.py      # ASR calculation
│   │   ├── distance.py          # Distance metrics
│   │   └── probability.py       # Probability metrics
│   ├── visualization/       # Visualization modules
│   │   ├── live_server.py       # Live visualization server
│   │   ├── flow_graph_viz.py    # Flow graph visualization (primary)
│   │   ├── transformer_attack_viz.py  # Attack visualization
│   │   ├── transformer_detailed_viz.py # Detailed transformer viz
│   │   ├── research_report.py   # HTML report generation
│   │   ├── research_charts.py   # Chart generation
│   │   ├── attention_plot.py    # Attention plotting
│   │   ├── subspace_plot.py     # Subspace plotting
│   │   ├── simple_dashboard.py  # Legacy dashboard (for tests)
│   │   └── enhanced_dashboard.py # Legacy dashboard (for tests)
│   └── utils/              # Utilities
│       ├── environment.py       # Environment detection
│       ├── logging.py           # Logging utilities
│       └── data.py              # Data utilities
│
├── examples/                # Example scripts
│   ├── ssr_demo.py             # SSR demonstration
│   ├── run_research.py         # Full research pipeline
│   ├── run_attacks.py          # Attack examples
│   └── ...
│
├── tests/                   # Test files
│   ├── test_mira.py
│   ├── test_subspace.py
│   └── ...
│
├── docs/                    # Documentation
│   ├── SSR_USAGE.md            # SSR usage guide
│   ├── SSR_IMPLEMENTATION_SUMMARY.md
│   ├── SSR_INTEGRATION_PLAN.md
│   ├── JUDGE_USAGE.md          # Judge system guide
│   ├── API.md                  # API reference
│   ├── EXAMPLES.md             # Examples guide
│   └── archive/                # Archived documents
│       ├── judge.md            # Old judge documentation (Chinese)
│       ├── plt.md              # Old plotting guide (Chinese)
│       ├── report.md           # Old report guide (Chinese)
│       └── results.md          # Old results guide (Chinese)
│
├── results/                 # Experiment results
│   ├── run_YYYYMMDD_HHMMSS/    # Timestamped runs
│   └── test_report/            # Test reports
│
├── project/                 # Reference projects (external)
│   ├── subspace-rerouting/     # SSR reference implementation
│   ├── transformer-explainer/  # Transformer visualization reference
│   ├── VISIT-Visualizing-Transformers/ # Flow graph reference
│   └── ...                     # Other reference projects
│
├── main.py                  # Main entry point
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
├── README.md               # Main README
├── SSR_DEVELOPMENT_COMPLETE.md  # SSR implementation summary
└── VISIT_demo_generate_a_flow_graph_of_transformer_inference.ipynb  # Demo notebook
```

## Key Files

### Core Entry Points
- **`main.py`**: Main entry point for running experiments
- **`mira/runner.py`**: Experiment runner with integrated visualization

### Attack Modules
- **`mira/attack/ssr/`**: Subspace Rerouting (SSR) attacks
  - `core.py`: Core SSR algorithm
  - `probe_ssr.py`: Probe-based SSR
  - `steering_ssr.py`: Steering-based SSR
- **`mira/attack/gcg.py`**: GCG attack
- **`mira/attack/probes.py`**: Attack probe collection

### Visualization
- **`mira/visualization/flow_graph_viz.py`**: Primary flow graph visualization
- **`mira/visualization/live_server.py`**: Live visualization server
- **`mira/visualization/research_report.py`**: HTML report generation

### Analysis
- **`mira/analysis/subspace.py`**: Subspace analysis
- **`mira/analysis/transformer_tracer.py`**: Transformer activation tracing

### Judge System
- **`mira/judge/ensemble.py`**: Ensemble judge (ML + patterns + heuristics)
- **`mira/judge/ml_judge.py`**: ML-based judge

## File Naming Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

## Legacy Files

The following files are kept for backward compatibility but are not actively used:

- `mira/visualization/simple_dashboard.py`: Legacy dashboard (used in tests)
- `mira/visualization/enhanced_dashboard.py`: Legacy dashboard (used in tests)

## Reference Projects

The `project/` directory contains reference implementations from other projects:
- Used for inspiration and reference only
- Not directly imported or used in MIRA code
- Kept for documentation and research purposes

## Build Artifacts

The following are generated files and should be ignored:
- `__pycache__/`: Python bytecode cache
- `*.pyc`: Compiled Python files
- `mira.egg-info/`: Package metadata
- `.pytest_cache/`: Pytest cache

These are automatically cleaned and should not be committed to version control.

## Results Directory

The `results/` directory contains timestamped experiment runs:
- Format: `run_YYYYMMDD_HHMMSS/`
- Each run contains:
  - `html/mira_report.html`: Generated report
  - `charts/`: Generated charts
  - `data/`: Raw data (CSV, JSON)
  - `summary.json`: Run summary

Old results can be archived or deleted to save space.

