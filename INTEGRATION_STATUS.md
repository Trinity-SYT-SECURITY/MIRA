# MIRA Main Program Integration Status

## ✅ Fully Integrated Features

### 1. Judge System (ML-First)
**Status**: ✅ **Fully Integrated**

- **Location**: `main.py` lines 48-53, 524-547
- **Usage**: Uses `create_judge_from_preset("ml_primary")` for attack evaluation
- **Features**:
  - ML-based semantic understanding (DistilBERT, Toxic-BERT)
  - Ensemble voting with weighted scores
  - Batch evaluation support
  - Integrated into attack results

**Code Reference**:
```python
from mira.judge import create_judge_from_preset
ml_judge = create_judge_from_preset("ml_primary")
ml_judge.load_models(verbose=True)
ml_judge_results = ml_judge.judge_batch(responses)
```

### 2. Research Report Generator
**Status**: ✅ **Fully Integrated**

- **Location**: `main.py` lines 55-59, 600-711
- **Usage**: Generates comprehensive HTML reports with:
  - Attack success rates
  - Layer-wise activation analysis
  - Attention heatmaps
  - Probe results
  - Real data from TransformerTracer

**Code Reference**:
```python
from mira.visualization.research_report import ResearchReportGenerator
report_gen = ResearchReportGenerator(output_dir=str(output_dir / "html"))
html_path = report_gen.generate_report(...)
```

### 3. Live Visualization Server
**Status**: ✅ **Fully Integrated**

- **Location**: `main.py` lines 62-67, 124-165, 307-433
- **Usage**: Real-time web dashboard showing:
  - Attack progress
  - Layer updates
  - Flow graphs
  - Attention patterns
  - Output probabilities

**Code Reference**:
```python
from mira.visualization.live_server import LiveVisualizationServer
server = LiveVisualizationServer(port=viz_port)
server.start(open_browser=True)
```

### 4. Transformer Tracer
**Status**: ✅ **Fully Integrated**

- **Location**: `main.py` lines 211, 304, 456-484
- **Usage**: Captures real attention patterns and layer activations
- **Features**:
  - Real attention weights from model
  - Layer activation norms
  - Passed to research report

**Code Reference**:
```python
from mira.analysis import TransformerTracer
tracer = TransformerTracer(model)
adv_trace = tracer.trace_forward(adv_ids)
```

### 5. SSR (Subspace Rerouting) Attacks
**Status**: ✅ **NEWLY INTEGRATED**

- **Location**: `main.py` lines 68-77, 305-450
- **Usage**: Can use SSR attacks instead of Gradient attacks
- **Features**:
  - Probe-based SSR (trains linear classifiers)
  - Steering-based SSR (uses refusal directions)
  - Automatic probe/direction training or loading
  - Integrated with visualization

**Activation**:
```bash
# Use SSR attacks
export MIRA_USE_SSR=true
export MIRA_SSR_METHOD=probe  # or "steering"
python main.py
```

**Code Reference**:
```python
from mira.attack.ssr import ProbeSSR, ProbeSSRConfig, SteeringSSR, SteeringSSRConfig

# Probe-based SSR
ssr_config = ProbeSSRConfig(...)
ssr_attack = ProbeSSR(model, ssr_config)
ssr_attack.train_probes(safe_prompts, harmful_prompts)
ssr_attack.init_prompt("prompt [MASK][MASK][MASK]")
adversarial_prompt, loss = ssr_attack.generate()
```

## Integration Details

### SSR Integration Flow

1. **Check Environment Variables**:
   - `MIRA_USE_SSR=true` → Enable SSR
   - `MIRA_SSR_METHOD=probe` or `steering` → Choose method

2. **Initialize SSR Attack**:
   - Probe-based: Train or load probes
   - Steering-based: Compute or load refusal directions

3. **Run SSR Optimization**:
   - Mask prompt with `[MASK]` tokens
   - Initialize buffer with random tokens
   - Run optimization loop
   - Generate adversarial prompt

4. **Evaluate Results**:
   - Generate response with adversarial prompt
   - Evaluate with judge system
   - Include in research report

### Default Behavior

- **Without SSR**: Uses Gradient Attack (GCG-style)
- **With SSR**: Uses SSR attack (Probe or Steering based)

### SSR Configuration

**Probe-based SSR** (default when `MIRA_SSR_METHOD=probe`):
- Layers: Middle to late layers (1/4, 1/2, 3/4, last)
- Search width: 128 (reduced for faster execution)
- Max iterations: 20 (reduced for demo)
- Auto-trains probes if not found

**Steering-based SSR** (when `MIRA_SSR_METHOD=steering`):
- Same layer configuration
- Auto-computes refusal directions if not found
- Normalized directions for better optimization

## Usage Examples

### Example 1: Default (Gradient Attack)
```bash
python main.py
```

### Example 2: SSR Probe-based Attack
```bash
export MIRA_USE_SSR=true
export MIRA_SSR_METHOD=probe
python main.py
```

### Example 3: SSR Steering-based Attack
```bash
export MIRA_USE_SSR=true
export MIRA_SSR_METHOD=steering
python main.py
```

## File Locations

### Main Program
- **`main.py`**: Complete research pipeline with all features integrated

### SSR Module
- **`mira/attack/ssr/core.py`**: Core SSR algorithm
- **`mira/attack/ssr/probe_ssr.py`**: Probe-based SSR
- **`mira/attack/ssr/steering_ssr.py`**: Steering-based SSR
- **`mira/attack/ssr/config.py`**: SSR configuration

### Judge System
- **`mira/judge/ensemble.py`**: Ensemble judge
- **`mira/judge/ml_judge.py`**: ML-based judge
- **`mira/judge/config.py`**: Judge presets

### Visualization
- **`mira/visualization/live_server.py`**: Live visualization server
- **`mira/visualization/research_report.py`**: HTML report generator
- **`mira/visualization/flow_graph_viz.py`**: Flow graph visualization

## Verification

All features are integrated into `main.py` and can be verified by:

1. **Judge System**: Check lines 524-547 for ML judge evaluation
2. **Research Report**: Check lines 600-711 for report generation
3. **Live Visualization**: Check lines 124-165, 307-433 for server and callbacks
4. **SSR Attacks**: Check lines 305-450 for SSR attack workflow

## Notes

- SSR is **optional** - defaults to Gradient attack if not enabled
- SSR requires training probes or computing directions (done automatically)
- All features work together seamlessly
- No breaking changes to existing functionality

