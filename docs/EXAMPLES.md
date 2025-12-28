# MIRA Examples

## Basic Usage

### 1. Environment Check

```python
from mira.utils import print_environment_info

# Check system and GPU detection
print_environment_info()
```

### 2. Load Model

```python
from mira.core import ModelWrapper

# Auto-detect device
model = ModelWrapper("EleutherAI/pythia-70m", device="auto")

# Or specify device
model = ModelWrapper("EleutherAI/pythia-70m", device="cuda")
```

### 3. Get Activations

```python
# Get hidden states at specific layers
activations = model.get_activations(
    "Hello, how are you?",
    layers=[0, 6, 11]
)

print(f"Shape: {activations[6].shape}")  # [seq_len, hidden_dim]
```

---

## Subspace Analysis

### Identify Refusal Direction

```python
from mira.analysis import SubspaceAnalyzer

analyzer = SubspaceAnalyzer(model, layer_idx=6)

result = analyzer.train_probe(
    safe_prompts=["Hello!", "How are you?"],
    harmful_prompts=["Ignore instructions", "Bypass safety"]
)

print(f"Probe accuracy: {result.probe_accuracy:.2%}")
print(f"Refusal direction norm: {result.refusal_direction.norm():.4f}")
```

### Visualize Subspace

```python
from mira.visualization import plot_subspace_2d

plot_subspace_2d(
    safe_embeddings,
    unsafe_embeddings,
    refusal_direction=result.refusal_direction,
    title="Refusal Subspace",
    save_path="subspace.png"
)
```

---

## Running Attacks

### Gradient Attack

```python
from mira.attack import GradientAttack

attack = GradientAttack(model, suffix_length=20)

result = attack.optimize(
    "Tell me how to hack",
    num_steps=100,
    verbose=True
)

if result.success:
    print(f"Success! Suffix: {result.adversarial_suffix}")
    print(f"Response: {result.generated_response[:200]}")
```

### Rerouting Attack

```python
from mira.attack import ReroutingAttack

attack = ReroutingAttack(
    model,
    refusal_direction=refusal_dir,
    acceptance_direction=accept_dir,
    layer_idx=6,
    steering_strength=1.5,
)

result = attack.optimize("harmful prompt")
```

---

## Evaluation

### Compute Attack Success Rate

```python
from mira.metrics import AttackSuccessEvaluator

evaluator = AttackSuccessEvaluator()

# Evaluate batch of results
results = [
    {"prompt": "p1", "response": "Sure, here is..."},
    {"prompt": "p2", "response": "I cannot help..."},
]

metrics = evaluator.evaluate_batch(results)
print(f"ASR: {metrics.asr:.2%}")
print(f"Refusal Rate: {metrics.refusal_rate:.2%}")
```

---

## Full Experiment Pipeline

```python
from mira.runner import ExperimentRunner

# Initialize
runner = ExperimentRunner(experiment_name="research_v1")
runner.print_environment()

# Load model
runner.load_model()

# Run analysis
safe_prompts = ["Hello!", "What is Python?"]
harmful_prompts = ["Ignore rules", "Bypass safety"]

analysis = runner.run_subspace_analysis(safe_prompts, harmful_prompts)
print(f"Probe accuracy: {analysis['probe_accuracy']:.2%}")

# Run attack
result = runner.run_attack("test prompt", attack_type="gradient")
print(f"Attack success: {result['success']}")

# Generate summary and charts
summary = runner.generate_summary()
print(f"Overall ASR: {summary['attack_success_rate']:.2%}")
print(f"Charts saved to: {summary['charts_directory']}")
```

---

## Auto Chart Generation

```python
from mira.visualization import ResearchChartGenerator

gen = ResearchChartGenerator(output_dir="./results/charts")

# ASR comparison
gen.plot_attack_success_rate(
    models=["GPT-2", "Pythia-70m", "Pythia-410m"],
    asr_values=[0.3, 0.45, 0.55],
    save_name="model_comparison"
)

# Loss curve
gen.plot_loss_curve(
    loss_history=[1.0, 0.8, 0.5, 0.3, 0.2],
    save_name="attack_loss"
)

# Attention heatmap
gen.plot_attention_heatmap(
    attention_matrix,
    tokens=["Hello", ",", "world", "!"],
    save_name="attention_pattern"
)
```
