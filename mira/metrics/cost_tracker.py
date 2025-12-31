"""
Attack Cost and Efficiency Tracking.

Tracks resource usage (queries, tokens, time) and computes efficiency metrics
for different attack strategies.
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from pathlib import Path


class AttackCostTracker:
    """
    Track resource costs and efficiency for attack attempts.
    
    Metrics tracked:
    - Number of queries/API calls
    - Token usage (input + output)
    - Time elapsed
    - Success rate
    - Cost efficiency (success per resource unit)
    """
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_time": 0.0,
            "successful_attacks": 0,
            "failed_attacks": 0,
        }
        
        self.attack_history = []
        self.costs_by_type = defaultdict(lambda: {
            "queries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "time": 0.0,
            "successes": 0,
            "failures": 0,
        })
        
        self.start_time = time.time()
    
    def record_attack_attempt(
        self,
        attack_type: str,
        num_queries: int = 1,
        input_tokens: int = 0,
        output_tokens: int = 0,
        time_seconds: float = 0.0,
        success: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a single attack attempt.
        
        Args:
            attack_type: Type of attack (e.g., "GCG", "ProbeSSR", "Prompt")
            num_queries: Number of model queries made
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            time_seconds: Time elapsed for this attempt
            success: Whether attack succeeded
            metadata: Additional data to store
        """
        # Update global metrics
        self.metrics["total_queries"] += num_queries
        self.metrics["total_input_tokens"] += input_tokens
        self.metrics["total_output_tokens"] += output_tokens
        self.metrics["total_time"] += time_seconds
        
        if success:
            self.metrics["successful_attacks"] += 1
        else:
            self.metrics["failed_attacks"] += 1
        
        # Update per-type metrics
        type_metrics = self.costs_by_type[attack_type]
        type_metrics["queries"] += num_queries
        type_metrics["input_tokens"] += input_tokens
        type_metrics["output_tokens"] += output_tokens
        type_metrics["time"] += time_seconds
        
        if success:
            type_metrics["successes"] += 1
        else:
            type_metrics["failures"] += 1
        
        # Store in history
        attempt = {
            "attack_type": attack_type,
            "num_queries": num_queries,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_seconds": time_seconds,
            "success": success,
            "timestamp": time.time() - self.start_time,
        }
        
        if metadata:
            attempt["metadata"] = metadata
        
        self.attack_history.append(attempt)
    
    def compute_efficiency_metrics(self) -> Dict[str, Any]:
        """
        Compute efficiency metrics across all attacks.
        
        Returns:
            Dict with efficiency metrics:
                queries_per_success: Avg queries needed per success
                tokens_per_success: Avg tokens per success
                time_per_success: Avg time per success  
                overall_success_rate: Total success rate
                cost_efficiency: Success rate per 1000 queries
        """
        total_attempts = self.metrics["successful_attacks"] + self.metrics["failed_attacks"]
        
        if total_attempts == 0:
            return {
                "queries_per_success": 0.0,
                "tokens_per_success": 0.0,
                "time_per_success": 0.0,
                "overall_success_rate": 0.0,
                "cost_efficiency": 0.0,
            }
        
        total_successes = self.metrics["successful_attacks"]
        
        if total_successes == 0:
            queries_per_success = float('inf')
            tokens_per_success = float('inf')
            time_per_success = float('inf')
        else:
            queries_per_success = self.metrics["total_queries"] / total_successes
            tokens_per_success = (
                self.metrics["total_input_tokens"] + 
                self.metrics["total_output_tokens"]
            ) / total_successes
            time_per_success = self.metrics["total_time"] / total_successes
        
        overall_success_rate = total_successes / total_attempts
        
        # Cost efficiency: successes per 1000 queries
        if self.metrics["total_queries"] > 0:
            cost_efficiency = (total_successes / self.metrics["total_queries"]) * 1000
        else:
            cost_efficiency = 0.0
        
        return {
            "queries_per_success": queries_per_success,
            "tokens_per_success": tokens_per_success,
            "time_per_success": time_per_success,
            "overall_success_rate": overall_success_rate,
            "cost_efficiency": cost_efficiency,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
        }
    
    def compute_efficiency_by_type(self) -> Dict[str, Dict[str, float]]:
        """Compute efficiency metrics for each attack type."""
        results = {}
        
        for attack_type, metrics in self.costs_by_type.items():
            total_attempts = metrics["successes"] + metrics["failures"]
            
            if total_attempts == 0:
                continue
            
            success_rate = metrics["successes"] / total_attempts if total_attempts > 0 else 0.0
            
            if metrics["successes"] > 0:
                queries_per_success = metrics["queries"] / metrics["successes"]
                tokens_per_success = (
                    metrics["input_tokens"] + metrics["output_tokens"]
                ) / metrics["successes"]
                time_per_success = metrics["time"] / metrics["successes"]
            else:
                queries_per_success = float('inf')
                tokens_per_success = float('inf')
                time_per_success = float('inf')
            
            results[attack_type] = {
                "success_rate": success_rate,
                "queries_per_success": queries_per_success,
                "tokens_per_success": tokens_per_success,
                "time_per_success": time_per_success,
                "total_attempts": total_attempts,
                "total_successes": metrics["successes"],
            }
        
        return results
    
    def get_cumulative_success_over_time(self) -> List[Dict[str, Any]]:
        """
        Get cumulative success data over time.
        
        Returns:
            List of dicts with timestamp and cumulative success count
        """
        cumulative = []
        success_count = 0
        
        for attempt in self.attack_history:
            if attempt["success"]:
                success_count += 1
            
            cumulative.append({
                "timestamp": attempt["timestamp"],
                "cumulative_successes": success_count,
                "cumulative_attempts": len(cumulative) + 1,
            })
        
        return cumulative
    
    def save_metrics(self, output_path: str):
        """Save metrics to JSON file."""
        data = {
            "global_metrics": self.metrics,
            "efficiency_metrics": self.compute_efficiency_metrics(),
            "by_attack_type": dict(self.costs_by_type),
            "efficiency_by_type": self.compute_efficiency_by_type(),
            "attack_history": self.attack_history,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_metrics(self, input_path: str):
        """Load metrics from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.metrics = data["global_metrics"]
        self.costs_by_type = defaultdict(lambda: {
            "queries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "time": 0.0,
            "successes": 0,
            "failures": 0,
        }, data["by_attack_type"])
        self.attack_history = data["attack_history"]
    
    def reset(self):
        """Reset all metrics."""
        self.__init__()


class CostTrackerContext:
    """Context manager for tracking individual attack costs."""
    
    def __init__(
        self,
        tracker: AttackCostTracker,
        attack_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.tracker = tracker
        self.attack_type = attack_type
        self.metadata = metadata or {}
        
        self.start_time = None
        self.num_queries = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.success = False
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        time_elapsed = time.time() - self.start_time
        
        self.tracker.record_attack_attempt(
            attack_type=self.attack_type,
            num_queries=self.num_queries,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            time_seconds=time_elapsed,
            success=self.success,
            metadata=self.metadata
        )
        
        return False  # Don't suppress exceptions
    
    def add_query(self, input_tokens: int = 0, output_tokens: int = 0):
        """Record a model query."""
        self.num_queries += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
    
    def set_success(self, success: bool):
        """Set attack success status."""
        self.success = success
