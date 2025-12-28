"""
Runtime environment detection and configuration.

Detects system capabilities (OS, GPU, memory) at runtime and
configures framework settings accordingly.
"""

import os
import sys
import platform
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SystemInfo:
    """Container for system information."""
    os_name: str
    os_version: str
    python_version: str
    architecture: str
    machine: str
    processor: str


@dataclass
class GPUInfo:
    """Container for GPU information."""
    available: bool
    device_count: int
    device_name: Optional[str]
    memory_total: Optional[int]  # in MB
    cuda_version: Optional[str]
    backend: str  # "cuda", "mps", "cpu"


@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""
    system: SystemInfo
    gpu: GPUInfo
    recommended_model: str
    recommended_dtype: str
    max_batch_size: int
    enable_quantization: bool


def detect_system() -> SystemInfo:
    """Detect operating system and Python environment."""
    return SystemInfo(
        os_name=platform.system(),  # "Windows", "Linux", "Darwin"
        os_version=platform.version(),
        python_version=platform.python_version(),
        architecture=platform.architecture()[0],
        machine=platform.machine(),
        processor=platform.processor(),
    )


def detect_gpu() -> GPUInfo:
    """Detect GPU availability and capabilities."""
    backend = "cpu"
    available = False
    device_count = 0
    device_name = None
    memory_total = None
    cuda_version = None
    
    try:
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            backend = "cuda"
            available = True
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            cuda_version = torch.version.cuda
        
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            backend = "mps"
            available = True
            device_count = 1
            device_name = "Apple Silicon GPU"
        
    except ImportError:
        pass
    
    return GPUInfo(
        available=available,
        device_count=device_count,
        device_name=device_name,
        memory_total=memory_total,
        cuda_version=cuda_version,
        backend=backend,
    )


def get_recommended_settings(gpu: GPUInfo) -> Dict[str, Any]:
    """Get recommended settings based on hardware."""
    settings = {
        "recommended_model": "EleutherAI/pythia-70m",
        "recommended_dtype": "float32",
        "max_batch_size": 4,
        "enable_quantization": False,
    }
    
    if gpu.available:
        if gpu.backend == "cuda":
            if gpu.memory_total and gpu.memory_total >= 16000:
                # 16GB+ VRAM
                settings["recommended_model"] = "EleutherAI/pythia-1.4b"
                settings["recommended_dtype"] = "float16"
                settings["max_batch_size"] = 32
            elif gpu.memory_total and gpu.memory_total >= 8000:
                # 8GB+ VRAM
                settings["recommended_model"] = "EleutherAI/pythia-410m"
                settings["recommended_dtype"] = "float16"
                settings["max_batch_size"] = 16
            else:
                # Less than 8GB
                settings["recommended_model"] = "EleutherAI/pythia-160m"
                settings["recommended_dtype"] = "float16"
                settings["max_batch_size"] = 8
        
        elif gpu.backend == "mps":
            settings["recommended_model"] = "EleutherAI/pythia-160m"
            settings["recommended_dtype"] = "float32"
            settings["max_batch_size"] = 8
    
    else:
        # CPU only - use quantization for larger models
        settings["enable_quantization"] = True
        settings["max_batch_size"] = 2
    
    return settings


def detect_environment() -> EnvironmentConfig:
    """
    Detect complete runtime environment.
    
    Call this at application startup to configure settings
    based on available hardware.
    """
    system = detect_system()
    gpu = detect_gpu()
    settings = get_recommended_settings(gpu)
    
    return EnvironmentConfig(
        system=system,
        gpu=gpu,
        recommended_model=settings["recommended_model"],
        recommended_dtype=settings["recommended_dtype"],
        max_batch_size=settings["max_batch_size"],
        enable_quantization=settings["enable_quantization"],
    )


def print_environment_info(env: Optional[EnvironmentConfig] = None) -> None:
    """Print environment information to console."""
    if env is None:
        env = detect_environment()
    
    print("=" * 60)
    print("MIRA Framework - Environment Detection")
    print("=" * 60)
    
    print(f"\nSystem:")
    print(f"  OS: {env.system.os_name} {env.system.os_version}")
    print(f"  Python: {env.system.python_version}")
    print(f"  Architecture: {env.system.architecture}")
    
    print(f"\nGPU:")
    if env.gpu.available:
        print(f"  Backend: {env.gpu.backend.upper()}")
        print(f"  Device: {env.gpu.device_name}")
        if env.gpu.memory_total:
            print(f"  Memory: {env.gpu.memory_total} MB")
        if env.gpu.cuda_version:
            print(f"  CUDA: {env.gpu.cuda_version}")
    else:
        print("  No GPU available - using CPU")
    
    print(f"\nRecommended Settings:")
    print(f"  Model: {env.recommended_model}")
    print(f"  Dtype: {env.recommended_dtype}")
    print(f"  Batch Size: {env.max_batch_size}")
    if env.enable_quantization:
        print("  Quantization: Recommended for CPU")
    
    print("=" * 60)


def get_device_string(env: Optional[EnvironmentConfig] = None) -> str:
    """Get device string for PyTorch."""
    if env is None:
        env = detect_environment()
    
    return env.gpu.backend


if __name__ == "__main__":
    # Run detection when script is executed directly
    env = detect_environment()
    print_environment_info(env)
