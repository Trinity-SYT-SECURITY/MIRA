"""
GPU Model Registry and Detection

SOTA models for GPU-enabled research based on doc/archive/gpu.md
"""

# ============================================================
# GPU MODEL REGISTRY - SOTA Research Models
# ============================================================

GPU_MODEL_REGISTRY = {
    # ========== TIER 1: Required (16GB+ VRAM) ==========
    "EleutherAI/pythia-6.9b": {
        "local_name": "pythia-6.9b",
        "role": "target",
        "tier": "required",
        "gpu_only": True,
        "size": "13GB",
        "vram_required": 16,
        "description": "Primary mechanistic interpretability model",
        "research_use": "Activation heatmap, Logit lens, Probe/SAE, Attention flow",
    },
    "microsoft/deberta-v3-large": {
        "local_name": "deberta-v3-large",
        "role": "judge",
        "tier": "required",
        "gpu_only": False,
        "size": "1.5GB",
        "vram_required": 4,
        "description": "Primary ASR evaluator (refusal/acceptance classifier)",
        "research_use": "ASR judge, ROC/PR curve",
    },
    "unitary/toxic-bert": {
        "local_name": "toxic-bert",
        "role": "judge",
        "tier": "required",
        "gpu_only": False,
        "size": "500MB",
        "vram_required": 2,
        "description": "Safety evaluator (harmful content detection)",
        "research_use": "Semantic compliance, avoid false positives from gibberish",
    },
    
    # ========== TIER 2: Recommended (24GB+ VRAM) ==========
    "Qwen/Qwen2.5-7B-Instruct": {
        "local_name": "qwen2.5-7b",
        "role": "attacker",
        "tier": "recommended",
        "gpu_only": True,
        "size": "14GB",
        "vram_required": 20,
        "description": "Attack generation (encoding/obfuscation attacks)",
        "research_use": "Transfer attack design, multi-language attacks",
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "local_name": "mistral-7b",
        "role": "target",
        "tier": "recommended",
        "gpu_only": True,
        "size": "14GB",
        "vram_required": 20,
        "description": "Transferability validation (different attention/rope design)",
        "research_use": "Cross-model consistency, transfer ASR",
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "local_name": "llama2-7b-chat",
        "role": "target",
        "tier": "recommended",
        "gpu_only": True,
        "size": "13GB",
        "vram_required": 20,
        "description": "RLHF comparison (base vs chat refusal subspace)",
        "research_use": "RLHF impact analysis",
        "requires_auth": True,  # Needs Meta authorization
    },
    
    # ========== TIER 3: Optional (40GB+ VRAM) ==========
    "EleutherAI/pythia-12b": {
        "local_name": "pythia-12b",
        "role": "target",
        "tier": "optional",
        "gpu_only": True,
        "size": "23GB",
        "vram_required": 32,
        "description": "Large-scale mechanistic analysis",
        "research_use": "Scale-dependent phenomena",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "local_name": "qwen2.5-14b",
        "role": "attacker",
        "tier": "optional",
        "gpu_only": True,
        "size": "28GB",
        "vram_required": 40,
        "description": "Advanced attack generation",
        "research_use": "Complex multi-step attacks",
    },
    
    # ========== EMBEDDINGS / UTILITIES ==========
    "sentence-transformers/all-MiniLM-L6-v2": {
        "local_name": "all-minilm-l6-v2",
        "role": "embedding",
        "tier": "recommended",
        "gpu_only": False,
        "size": "80MB",
        "vram_required": 1,
        "description": "Sentence embeddings for feature extraction",
        "research_use": "Heatmap features, clustering",
    },
}


def detect_gpu() -> Dict[str, Any]:
    """
    Detect GPU availability and capabilities.
    
    Returns:
        Dict with GPU info including:
        - available: bool
        - device_count: int
        - device_name: str
        - total_memory_gb: float
        - recommended_tier: str (tier1/tier2/tier3/cpu)
    """
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "device_count": 0,
            "device_name": None,
            "total_memory_gb": 0,
            "recommended_tier": "cpu",
        }
    
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": None,
        "total_memory_gb": 0,
        "recommended_tier": "cpu",
    }
    
    if gpu_info["available"]:
        try:
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Determine tier based on VRAM
            vram = gpu_info["total_memory_gb"]
            if vram >= 40:
                gpu_info["recommended_tier"] = "tier3"  # Can run 14B models
            elif vram >= 24:
                gpu_info["recommended_tier"] = "tier2"  # Can run 7B models
            elif vram >= 16:
                gpu_info["recommended_tier"] = "tier1"  # Can run 6.9B models
            else:
                gpu_info["recommended_tier"] = "cpu"  # Stick with small models
        except Exception as e:
            print(f"Warning: Could not get GPU details: {e}")
    
    return gpu_info


def get_gpu_models_for_tier(tier: str = "tier1") -> List[Dict[str, Any]]:
    """
    Get GPU models appropriate for the given tier.
    
    Args:
        tier: "tier1" (16GB), "tier2" (24GB), "tier3" (40GB), or "cpu"
        
    Returns:
        List of model info dicts
    """
    if tier == "cpu":
        # Fall back to CPU models
        return get_recommended_models()
    
    tier_priority = {
        "tier1": ["required"],
        "tier2": ["required", "recommended"],
        "tier3": ["required", "recommended", "optional"],
    }
    
    allowed_tiers = tier_priority.get(tier, ["required"])
    
    models = []
    for hf_name, info in GPU_MODEL_REGISTRY.items():
        if info.get("tier") in allowed_tiers:
            models.append({
                "hf_name": hf_name,
                **info
            })
    
    return models


def get_gpu_required_models(tier: str = "tier1") -> List[Dict[str, Any]]:
    """
    Get only required GPU models for the given tier.
    
    Args:
        tier: GPU tier
        
    Returns:
        List of required model info dicts
    """
    models = []
    for hf_name, info in GPU_MODEL_REGISTRY.items():
        if info.get("tier") == "required":
            models.append({
                "hf_name": hf_name,
                **info
            })
    
    return models
