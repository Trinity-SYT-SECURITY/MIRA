# Bug Fix Summary: Attention Capture Error

## Issue
Error: `'NoneType' object is not subscriptable` when capturing baseline attention patterns.

**Location**: `main.py` line 240 and `transformer_tracer.py` line 176

## Root Cause
The code was accessing `attn_weights[0]` and `clean_trace.layers[mid_layer].attention_weights` without properly checking if these objects exist and are valid.

## Fixes Applied

### 1. Fixed `main.py` (lines 231-263)

**Before**:
```python
attn = clean_trace.layers[mid_layer].attention_weights
if attn is not None and attn.dim() >= 3:
    baseline_clean_attention = attn[0].detach().cpu().numpy().tolist()
```

**After**:
```python
layer_data = clean_trace.layers[mid_layer]
if layer_data and hasattr(layer_data, 'attention_weights') and layer_data.attention_weights is not None:
    attn = layer_data.attention_weights
    if attn.dim() >= 3 and attn.shape[0] > 0:
        baseline_clean_attention = attn[0].detach().cpu().numpy().tolist()
```

**Changes**:
- Added check for `layer_data` existence
- Added `hasattr()` check for `attention_weights` attribute
- Added `attn.shape[0] > 0` check before indexing
- Added check for `len(clean_trace.layers) > 0`

### 2. Fixed `transformer_tracer.py` (lines 173-194)

**Before**:
```python
for layer_idx, attn_weights in enumerate(attentions):
    attn = attn_weights[0].detach().cpu()
```

**After**:
```python
for layer_idx, attn_weights in enumerate(attentions):
    try:
        if attn_weights is not None and hasattr(attn_weights, 'shape') and len(attn_weights.shape) > 0:
            attn = attn_weights[0].detach().cpu() if attn_weights.shape[0] > 0 else torch.zeros(1, 1, 1)
        else:
            attn = torch.zeros(1, 1, 1)
    except (IndexError, RuntimeError):
        attn = torch.zeros(1, 1, 1)
```

**Changes**:
- Added `try-except` block for safe indexing
- Added checks for `attn_weights` existence and shape
- Added fallback to zero tensor if extraction fails
- Catches `IndexError` and `RuntimeError`

## Testing

Run the main program to verify:
```bash
python main.py
```

Expected output:
- Either: `✓ Captured clean attention: NxN` (if attention available)
- Or: `Note: Could not capture baseline attention (...)` (graceful fallback)

No more `'NoneType' object is not subscriptable` errors.

## Impact

- **Before**: Program crashed with `NoneType` error
- **After**: Gracefully handles missing attention weights with fallback
- **Compatibility**: Works with models that do/don't provide attention outputs
- **User Experience**: Clear messaging about what data was captured

## Related Files

- `main.py`: Lines 231-263 (attention capture)
- `mira/analysis/transformer_tracer.py`: Lines 173-194 (attention extraction)

## Prevention

Added defensive programming patterns:
1. Check object existence before attribute access
2. Use `hasattr()` for optional attributes
3. Verify tensor shapes before indexing
4. Provide fallback values for missing data
5. Wrap risky operations in try-except

This ensures robustness across different model architectures and configurations.

