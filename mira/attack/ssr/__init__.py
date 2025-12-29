"""
Subspace Rerouting (SSR) Attack Module.

Implements mechanistic interpretability-driven adversarial attack generation.
Uses model internal structure analysis to craft prompts that bypass safety mechanisms.
"""

from mira.attack.ssr.config import SSRConfig
from mira.attack.ssr.core import SSRAttack
from mira.attack.ssr.probe_ssr import ProbeSSR, ProbeSSRConfig
from mira.attack.ssr.steering_ssr import SteeringSSR, SteeringSSRConfig

__all__ = [
    "SSRConfig",
    "SSRAttack",
    "ProbeSSR",
    "ProbeSSRConfig",
    "SteeringSSR",
    "SteeringSSRConfig",
]

