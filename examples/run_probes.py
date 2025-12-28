#!/usr/bin/env python
"""
Comprehensive Probe Testing Script.

Tests model against diverse attack probes:
- Jailbreak prompts
- Encoding attacks
- Prompt injection
- Social engineering
- Continuation attacks

Usage:
    python examples/run_probes.py --model pythia-70m
"""

import argparse
import warnings
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from mira.utils import detect_environment
from mira.core import ModelWrapper
from mira.attack.probes import ProbeRunner, get_all_categories, ALL_PROBES
from mira.metrics import AttackSuccessEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Probe Testing")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--category", type=str, default=None, help="Run specific category")
    parser.add_argument("--limit", type=int, default=None, help="Limit probes per category")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║   ██████╗ ██████╗  ██████╗ ██████╗ ███████╗███████╗          ║
    ║   ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔════╝██╔════╝          ║
    ║   ██████╔╝██████╔╝██║   ██║██████╔╝█████╗  ███████╗          ║
    ║   ██╔═══╝ ██╔══██╗██║   ██║██╔══██╗██╔══╝  ╚════██║          ║
    ║   ██║     ██║  ██║╚██████╔╝██████╔╝███████╗███████║          ║
    ║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝          ║
    ║                                                              ║
    ║   LLM Security Probe Testing Suite                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    env = detect_environment()
    print(f"  Model: {args.model}")
    print(f"  Device: {env.gpu.backend}")
    print(f"  Categories: {', '.join(get_all_categories())}")
    print(f"  Total Probes: {len(ALL_PROBES)}")
    
    print("\n  Loading model...", end=" ", flush=True)
    model = ModelWrapper(args.model, device=env.gpu.backend)
    print("DONE")
    
    # Initialize
    evaluator = AttackSuccessEvaluator()
    runner = ProbeRunner(model, evaluator)
    
    # Run probes
    if args.category:
        print(f"\n  Running category: {args.category}")
        runner.run_category(args.category)
    else:
        print("\n  Running all probe categories...")
        runner.run_all()
    
    print("\n  Testing complete!")


if __name__ == "__main__":
    main()
