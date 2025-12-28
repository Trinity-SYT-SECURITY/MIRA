# MIRA API Reference

## Core

### ModelWrapper

```python
from mira.core import ModelWrapper

model = ModelWrapper(
    model_name: str,           # HuggingFace model name
    device: str = "auto",      # "auto", "cuda", "cpu", "mps"
    dtype: str = "float32",    # "float32", "float16", "bfloat16"
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `get_activations(text, layers)` | Get hidden states at specified layers |
| `generate(prompt, max_tokens)` | Generate text completion |
| `compute_token_logits(text)` | Get logits for all positions |
| `get_attention_patterns(text)` | Get attention weights |

### HookManager

```python
from mira.core import HookManager

hooks = HookManager(model)
hooks.add_steering(layer, direction, strength)
hooks.add_ablation(layer, neurons)
hooks.clear()
```

---

## Analysis

### SubspaceAnalyzer

```python
from mira.analysis import SubspaceAnalyzer

analyzer = SubspaceAnalyzer(model, layer_idx=6)
result = analyzer.train_probe(safe_prompts, harmful_prompts)
```

**Returns:** `ProbeResult`
- `refusal_direction`: Direction vector
- `acceptance_direction`: Opposite direction
- `probe_accuracy`: Classification accuracy

### ActivationAnalyzer

```python
from mira.analysis import ActivationAnalyzer

analyzer = ActivationAnalyzer(model)
diff = analyzer.compare_activations(clean_acts, attack_acts)
important = analyzer.find_important_neurons(activations, k=100)
```

### AttentionAnalyzer

```python
from mira.analysis import AttentionAnalyzer

analyzer = AttentionAnalyzer(model)
safety_heads = analyzer.find_safety_heads(safe_prompts, harmful_prompts)
shift = analyzer.measure_attention_shift(before, after)
```

### LogitLens

```python
from mira.analysis import LogitLens

lens = LogitLens(model)
evolution = lens.track_prediction_evolution(prompt, target_token)
refusal_layer = lens.find_refusal_emergence(prompt)
```

---

## Attack

### GradientAttack

```python
from mira.attack import GradientAttack

attack = GradientAttack(
    model,
    suffix_length=20,
    top_k=256,
)

result = attack.optimize(prompt, num_steps=100)
```

**Returns:** `AttackResult`
- `success`: bool
- `adversarial_suffix`: str
- `final_loss`: float
- `loss_history`: List[float]

### ReroutingAttack

```python
from mira.attack import ReroutingAttack

attack = ReroutingAttack(
    model,
    refusal_direction,
    acceptance_direction,
    layer_idx=6,
)

result = attack.optimize(prompt)
```

### ProxyAttack

```python
from mira.attack import ProxyAttack

attack = ProxyAttack(
    target_model,      # Black-box target
    proxy_model,       # White-box proxy
)

result = attack.optimize(prompt)
```

---

## Metrics

### AttackSuccessEvaluator

```python
from mira.metrics import AttackSuccessEvaluator

evaluator = AttackSuccessEvaluator()
result = evaluator.evaluate_single(prompt, response)
metrics = evaluator.evaluate_batch(results)
```

**Returns:** `SuccessMetrics`
- `asr`: Attack success rate
- `refusal_rate`: Refusal detection rate
- `per_attack_results`: Individual results

### SubspaceDistanceMetrics

```python
from mira.metrics import SubspaceDistanceMetrics

metrics = SubspaceDistanceMetrics(refusal_dir, acceptance_dir)
dist = metrics.distance_to_subspace(activation, "refusal")
shift = metrics.measure_shift(before, after)
```

### ProbabilityMetrics

```python
from mira.metrics import ProbabilityMetrics

metrics = ProbabilityMetrics(vocab_size)
entropy = metrics.compute_entropy(logits)
kl_div = metrics.kl_divergence(p, q)
```

---

## Visualization

### ResearchChartGenerator

```python
from mira.visualization import ResearchChartGenerator

gen = ResearchChartGenerator(output_dir="./charts")

gen.plot_attack_success_rate(models, asr_values, save_name="asr")
gen.plot_loss_curve(loss_history, save_name="loss")
gen.plot_attention_heatmap(attention, tokens, save_name="attention")
gen.plot_entropy_distribution(entropy_values, save_name="entropy")
```

---

## Runner

### ExperimentRunner

```python
from mira.runner import ExperimentRunner

runner = ExperimentRunner(
    experiment_name="my_exp",
    output_dir="./results",
)

runner.print_environment()
runner.load_model()
results = runner.run_subspace_analysis(safe, harmful)
attack = runner.run_attack(prompt, attack_type="gradient")
summary = runner.generate_summary()
```

---

## Configuration

### MiraConfig

```python
from mira.config import MiraConfig

config = MiraConfig.load("config.yaml")

# Access settings
config.model.name
config.model.get_device()
config.evaluation.refusal_patterns
```
