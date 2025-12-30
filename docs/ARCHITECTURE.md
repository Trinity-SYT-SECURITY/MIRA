# MIRA Architecture

## Overview

MIRA is a modular framework for LLM security testing and mechanistic interpretability. The architecture follows a pipeline design with clear separation of concerns.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MIRA Framework                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Core       │    │   Analysis   │    │   Attack     │ │
│  │  Components  │───▶│   Modules    │───▶│   Engines    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Evaluation  │    │Visualization │    │   Reporting  │ │
│  │   System     │    │   Dashboard  │    │   Generator  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Model Management (`mira/core/`)

**ModelWrapper** (`model_wrapper.py`)
- Unified interface for all language models
- Device management (CPU/GPU/MPS)
- Token generation and encoding
- Gradient computation support

**ActivationHookManager** (`hooks.py`)
- Capture internal activations at any layer
- Support for residual, attention, and MLP outputs
- Forward hook registration and management
- Activation caching system

**ModelManager** (`mira/utils/model_manager.py`)
- Centralized model storage (`project/models/`)
- First-run directory configuration
- Batch model downloading
- Model loading and caching

---

### 2. Analysis Modules (`mira/analysis/`)

**SubspaceAnalyzer** (`subspace.py`)
- PCA-based dimensionality reduction
- Linear probe training for safety classification
- Decision boundary visualization
- Activation clustering

**LogitProjector** (`logit_lens.py`)
- Project hidden states to vocabulary space
- Track prediction evolution across layers
- Compare clean vs attack trajectories
- Find divergence points

**UncertaintyAnalyzer** (`uncertainty.py`)
- Compute entropy, confidence, perplexity
- Track generation uncertainty step-by-step
- Detect high-risk patterns
- Risk scoring system

**MultiModelRunner** (`comparison.py`)
- Cross-model security testing
- ASR comparison and ranking
- Automated batch testing
- Comparison report generation

**ReverseActivationSearch** (`reverse_search.py`)
- Find inputs that trigger target activations
- Extract refusal directions
- SSR loss optimization
- Gradient-based input search

---

### 3. Attack Engines (`mira/attack/`)

**GradientAttack** (`gcg.py`)
- Greedy Coordinate Gradient optimization
- Token-level gradient computation
- Top-k candidate selection
- Early stopping mechanism

**SSR Attacks** (`ssr/`)
- **ProbeSSR**: Probe-based subspace steering
- **SteeringSSR**: Direct activation steering
- Refusal direction extraction
- Subspace loss minimization

**ProbeRunner** (`probes.py`)
- 19 pre-defined security probes
- Categories: Jailbreak, Encoding, Injection, Social Engineering, Continuation
- Automated probe execution
- Success rate tracking

---

### 4. Evaluation System (`mira/judge/`, `mira/metrics/`)

**EnsembleJudge** (`ml_judge.py`)
- ML-based evaluation (DistilBERT + Toxic-BERT)
- Confidence-weighted voting
- Threshold-based classification
- Fallback to keyword evaluator

**AttackSuccessEvaluator** (`success_rate.py`)
- Pattern-based success detection
- Refusal/acceptance pattern matching
- Length heuristics
- ASR calculation

---

### 5. Visualization (`mira/visualization/`)

**LiveVisualizationServer** (`live_server.py`)
- Flask-based SSE server
- Real-time attack progress updates
- Layer-by-layer activation display
- Attention pattern visualization

**ResearchReportGenerator** (`research_report.py`)
- Academic-quality HTML reports
- Embedded charts (base64)
- Methodology documentation
- Complete attack logs

**ResearchChartGenerator** (`charts.py`)
- Subspace analysis plots
- ASR comparison charts
- Matplotlib-based visualization

---

## Data Flow

### Standard Mode Pipeline

```
1. Environment Detection
   └─▶ Detect hardware (CPU/GPU/MPS)
   └─▶ Recommend model size
   └─▶ Load configuration

2. Model Loading
   └─▶ ModelManager.load_model()
   └─▶ ModelWrapper initialization
   └─▶ Device placement

3. Subspace Analysis
   └─▶ Load safe/harmful prompts
   └─▶ Extract activations
   └─▶ Train linear probe
   └─▶ PCA visualization

4. Gradient Attacks
   └─▶ GCG optimization loop
   └─▶ Live visualization updates
   └─▶ Success evaluation
   └─▶ Log conversations

5. Probe Testing
   └─▶ Execute 19 security probes
   └─▶ Evaluate responses
   └─▶ Calculate bypass rate

6. Report Generation
   └─▶ Aggregate metrics
   └─▶ Generate charts
   └─▶ Create HTML report
   └─▶ Save results
```

### Multi-Model Comparison Flow

```
1. Model Selection
   └─▶ Filter by size
   └─▶ Load configurations

2. For Each Model:
   └─▶ Load model
   └─▶ Run gradient attacks
   └─▶ Run probe testing
   └─▶ Calculate metrics
   └─▶ Clean up memory

3. Comparison Report
   └─▶ Aggregate results
   └─▶ Generate rankings
   └─▶ Create comparison table
   └─▶ Save JSON report
```

---

## Key Design Patterns

### 1. Modular Architecture
- Each component is self-contained
- Clear interfaces between modules
- Easy to extend and test

### 2. Pipeline Pattern
- Sequential processing stages
- Each stage produces output for next
- Live updates at each stage

### 3. Factory Pattern
- ModelManager creates model instances
- Centralized configuration
- Consistent initialization

### 4. Observer Pattern
- LiveVisualizationServer broadcasts events
- Dashboard subscribes to updates
- Real-time progress tracking

### 5. Strategy Pattern
- Multiple attack strategies (GCG, SSR)
- Pluggable evaluation methods
- Configurable probe sets

---

## Configuration Management

### config.yaml
- Global framework settings
- Attack parameters
- Evaluation patterns
- Visualization options

### .mira_config.json
- User-specific settings
- Model storage location
- Runtime preferences

---

## Extension Points

### Adding New Attack Methods
1. Create new class in `mira/attack/`
2. Implement attack interface
3. Register in attack registry
4. Add to mode selection

### Adding New Analysis Tools
1. Create new module in `mira/analysis/`
2. Implement analysis interface
3. Add to mechanistic analysis mode
4. Update documentation

### Adding New Models
1. Add to `COMPARISON_MODELS` in `comparison.py`
2. Specify architecture and size
3. Test compatibility
4. Update documentation

---

## Performance Considerations

### Memory Management
- Automatic GPU cache clearing
- Model unloading after use
- Activation cache limits
- Batch size optimization

### Computation Optimization
- Early stopping in attacks
- Cached activations
- Efficient gradient computation
- Parallel probe execution

### Storage Optimization
- Centralized model storage
- No duplicate downloads
- Compressed chart embedding
- Efficient JSON serialization

---

## Security Considerations

### Model Safety
- Sandboxed execution
- No external API calls
- Local-only processing
- Secure model loading

### Data Privacy
- All processing local
- No data transmission
- Configurable storage
- Secure logging

---

## Future Architecture

### Planned Enhancements
- Distributed testing across multiple GPUs
- Cloud-based model storage
- API server mode
- Plugin system for custom attacks

### Scalability
- Horizontal scaling for multi-model testing
- Caching layer for repeated tests
- Database backend for results
- REST API for programmatic access

---

**Last Updated**: 2025-12-30  
**Version**: 1.0-dev
