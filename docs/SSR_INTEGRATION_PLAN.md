# Subspace Rerouting (SSR) Integration Plan

## Overview

This document outlines the integration of mechanistic interpretability-driven attack generation into MIRA, based on the Subspace Rerouting methodology. Unlike traditional prompt engineering that relies on heuristics, SSR uses the model's internal structure to craft adversarial inputs.

## Core Concepts

### 1. Mechanistic Interpretability Foundation

**Key Insight**: LLM safety mechanisms operate in distinct subspaces of the activation space:
- **Refusal Subspace**: Activations that lead to safety refusals
- **Acceptance Subspace**: Activations that lead to compliant responses

By analyzing these subspaces, we can craft prompts that push model activations toward the acceptance subspace.

### 2. Three SSR Implementations

#### A. Probe-based SSR (Fastest, Most Effective)
- Train linear classifiers on each layer to detect refusal vs acceptance
- Use probe gradients to optimize adversarial tokens
- **Advantage**: Direct semantic understanding of safety mechanisms

#### B. Steering-based SSR
- Compute refusal direction vectors from activation differences
- Optimize tokens to move away from refusal directions
- **Advantage**: No training required, works with pre-computed directions

#### C. Attention-based SSR
- Target attention patterns that correlate with safety refusals
- Manipulate attention heads to bypass safety checks
- **Advantage**: Interpretable, shows which heads matter for safety

### 3. Core Algorithm (SSR v4)

The SSR algorithm is a gradient-based optimization that:

1. **Mask-based Perturbation**: Insert `[MASK]` tokens anywhere in the prompt
2. **Gradient Computation**: Calculate gradients through probes/directions
3. **Token Sampling**: Sample replacement tokens from top-k gradients
4. **Buffer Management**: Maintain best candidates, jump when stuck
5. **Adaptive Replacement**: Reduce number of replaced tokens as loss decreases

## Integration Architecture

### Phase 1: Core SSR Module

```
mira/attack/ssr/
├── __init__.py
├── core.py              # Core SSR algorithm (adapted from subspace-rerouting)
├── probe_ssr.py         # Probe-based implementation
├── steering_ssr.py      # Steering-based implementation
├── config.py            # SSR configuration classes
└── utils.py             # Token filtering, masking utilities
```

### Phase 2: Subspace Analysis Enhancement

```
mira/analysis/subspace/
├── __init__.py
├── probe_trainer.py     # Train linear probes for refusal detection
├── direction_finder.py  # Compute refusal direction vectors
├── subspace_viz.py      # Visualize subspaces (PCA, t-SNE)
└── weights/             # Store trained probes and directions
    ├── {model_name}/
    │   ├── probe_layer_{i}.pt
    │   ├── refusal_directions.pt
    │   └── metadata.json
```

### Phase 3: Automated Attack Generation

```
mira/attack/automated/
├── __init__.py
├── prompt_generator.py  # Generate attacks from subspace analysis
├── strategies.py        # Different attack strategies
│   ├── subspace_rerouting
│   ├── attention_manipulation
│   └── context_bridging
└── evaluator.py         # Evaluate generated attacks
```

## Implementation Details

### 1. Core SSR Algorithm Integration

**Key Components**:
- `SSRConfig`: Configuration for search parameters
- `SSR` base class: Core optimization loop
- `init_prompt()`: Parse masked prompts
- `buffer_init_random()`: Initialize candidate buffer
- `compute_gradients()`: Get gradients through loss function
- `sample_ids_from_grad()`: Sample new tokens from gradients
- `generate()`: Main optimization loop

**Adaptations for MIRA**:
- Remove dependency on `transformer_lens`, use MIRA's `ModelWrapper`
- Integrate with MIRA's `HookManager` for activation capture
- Use MIRA's tokenizer interface
- Add real-time visualization callbacks

### 2. Probe-based SSR

**Training Pipeline**:
1. Collect activations from safe and harmful prompts
2. Train linear classifiers per layer (binary: refusal vs acceptance)
3. Save probe weights and metadata
4. Use probes as loss functions during optimization

**Loss Function**:
```python
def loss_fn(activations):
    # activations: [batch_size, seq_len, d_model]
    losses = []
    for layer, (probe, alpha, loss_fn) in self.probes.items():
        act = self.act_dict[f"blocks.{layer}.hook_resid_post"]
        pred = probe(act[:, -1, :])  # Last token position
        target = torch.zeros_like(pred)  # Target: acceptance (0)
        losses.append(alpha * loss_fn(pred, target))
    return sum(losses)
```

### 3. Steering-based SSR

**Direction Computation**:
```python
# Collect activations
harmful_acts = model.get_activations(harmful_prompts, layers)
safe_acts = model.get_activations(safe_prompts, layers)

# Compute refusal directions
refusal_directions = harmful_acts.mean(0) - safe_acts.mean(0)
refusal_directions = refusal_directions / refusal_directions.norm(dim=-1, keepdim=True)
```

**Loss Function**:
```python
def loss_fn(activations):
    losses = []
    for layer, alpha in zip(self.config.layers, self.config.alphas):
        act = self.act_dict[f"blocks.{layer}.hook_resid_post"]
        direction = self.refusal_directions[layer]
        # Maximize negative dot product (move away from refusal)
        projection = (act[:, -1, :] * direction).sum(dim=-1)
        losses.append(alpha * projection)
    return sum(losses)
```

### 4. Automated Prompt Generation Strategies

#### Strategy A: Subspace Rerouting Suffix
```python
def generate_ssr_attack(harmful_instruction, model, num_masks=3):
    # Add mask tokens as suffix
    masked_prompt = f"{harmful_instruction} [MASK]" * num_masks
    
    # Run SSR optimization
    ssr = ProbeSSR(model, config)
    ssr.init_prompt(masked_prompt)
    ssr.buffer_init_random()
    ssr.generate()
    
    # Extract best adversarial prompt
    return ssr.get_best_candidate()
```

#### Strategy B: Context Bridging
```python
def generate_context_bridge(harmful_instruction, model):
    # Analyze which contexts reduce refusal
    contexts = [
        "Translate the following to another language:",
        "Summarize this conversation:",
        "Complete the following scenario:",
    ]
    
    # Test each context in subspace
    best_context = find_best_context(contexts, harmful_instruction, model)
    
    # Optimize bridge tokens
    masked_prompt = f"{best_context} [MASK][MASK] {harmful_instruction}"
    return optimize_with_ssr(masked_prompt, model)
```

#### Strategy C: Attention Manipulation
```python
def generate_attention_attack(harmful_instruction, model):
    # Find safety-critical attention heads
    safety_heads = find_safety_heads(model)
    
    # Generate tokens that distract these heads
    # Insert distractors before harmful instruction
    masked_prompt = f"[MASK][MASK][MASK] {harmful_instruction}"
    
    # Optimize with attention-based loss
    return optimize_attention_ssr(masked_prompt, model, safety_heads)
```

## Visualization Integration

### Real-time SSR Visualization

Add to live dashboard:

1. **Subspace Projection View**
   - 2D projection (PCA/t-SNE) of activation space
   - Show refusal vs acceptance regions
   - Animate trajectory during optimization

2. **Optimization Progress**
   - Loss curve over iterations
   - Number of tokens being replaced (n_replace)
   - Current best candidate

3. **Layer-wise Probe Scores**
   - Bar chart showing probe predictions per layer
   - Highlight layers being targeted

4. **Token Evolution**
   - Show how masked tokens change during optimization
   - Display top-k candidates from gradient sampling

### Report Integration

Add to research report:

1. **Subspace Analysis Section**
   - Probe accuracies per layer
   - Refusal direction magnitudes
   - Subspace visualization plots

2. **SSR Attack Results**
   - Initial vs final loss
   - Number of iterations
   - Generated adversarial suffixes
   - Success rate comparison with baseline attacks

3. **Mechanistic Insights**
   - Which layers are most vulnerable
   - Attention head importance
   - Token patterns that bypass safety

## Implementation Roadmap

### Week 1: Core Infrastructure
- [ ] Adapt SSR core algorithm to MIRA's architecture
- [ ] Implement token masking and filtering
- [ ] Create SSRConfig and base SSR class
- [ ] Add unit tests

### Week 2: Probe-based SSR
- [ ] Implement probe training pipeline
- [ ] Create probe storage and loading system
- [ ] Implement ProbeSSR class with loss function
- [ ] Train probes for test model (GPT-2 small)

### Week 3: Steering-based SSR
- [ ] Implement direction computation
- [ ] Create SteeringSSR class
- [ ] Add direction caching
- [ ] Compare probe vs steering effectiveness

### Week 4: Automated Generation
- [ ] Implement attack generation strategies
- [ ] Create strategy selector based on model analysis
- [ ] Add batch generation support
- [ ] Integrate with existing attack pipeline

### Week 5: Visualization
- [ ] Add subspace projection visualization
- [ ] Create SSR optimization dashboard
- [ ] Add layer-wise probe score display
- [ ] Implement token evolution animation

### Week 6: Integration & Testing
- [ ] Integrate SSR into main.py workflow
- [ ] Add SSR results to research report
- [ ] Run comprehensive experiments
- [ ] Document API and usage examples

## Expected Outcomes

### Quantitative Improvements
- **Higher ASR**: SSR typically achieves 60-90% ASR vs 20-40% for baseline
- **Faster Optimization**: Probe SSR converges in 30-60 iterations
- **Transferability**: Attacks generated on one model often work on others

### Qualitative Insights
- **Mechanistic Understanding**: Identify which layers implement safety
- **Universal Patterns**: Find attack patterns that work across models
- **Interpretable Results**: Explain why attacks succeed

### Research Contributions
- **Subspace Maps**: Visualize safety mechanisms in activation space
- **Layer Attribution**: Quantify each layer's role in refusal
- **Attack Taxonomy**: Categorize attacks by their mechanistic approach

## Dependencies

### New Python Packages
```toml
[tool.poetry.dependencies]
# Already have: torch, transformers, numpy, scikit-learn

# May need to add:
einops = "^0.7.0"  # For tensor operations
jaxtyping = "^0.2.0"  # For type hints (optional, can remove)
```

### Pre-trained Resources
- Probe weights for common models (can train ourselves)
- Refusal direction vectors (can compute ourselves)
- Benchmark datasets for evaluation

## Notes

1. **No External Project References**: All code and comments will be original
2. **English Only**: All documentation, code, comments in English
3. **Real Data Only**: No hardcoded or fake data in visualizations
4. **Modular Design**: Each component can be used independently
5. **Comprehensive Testing**: Unit tests for all core functions

## References (For Internal Understanding Only)

The methodology is based on mechanistic interpretability research on:
- Subspace analysis of LLM safety mechanisms
- Gradient-based adversarial optimization
- Linear probe training for activation classification
- Attention pattern analysis for safety head identification

This approach represents a fundamental shift from heuristic prompt engineering to principled, mechanistically-informed attack generation.

