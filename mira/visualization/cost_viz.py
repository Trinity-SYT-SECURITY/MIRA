"""
Cost and Efficiency Visualization for Attack Strategies.

Generates charts showing resource usage, efficiency metrics,
and cost-benefit analysis for different attack types.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Any
from pathlib import Path


def plot_cost_efficiency(
    cost_data: Dict[str, Any],
    output_path: str
):
    """
    Generate comprehensive cost/efficiency visualization.
    
    Creates 4-panel chart showing:
    1. Queries vs Success Rate (scatter)
    2. Time vs ASR (line chart)
    3. Cost Efficiency Comparison (bar chart)
    4. Cumulative Success Over Time (area chart)
    
    Args:
        cost_data: Dict with cost metrics from AttackCostTracker
        output_path: Path to save visualization
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Extract data
    by_type = cost_data.get("efficiency_by_type", {})
    cumulative = cost_data.get("cumulative_success", [])
    
    # Panel 1: Queries vs Success Rate
    ax1 = fig.add_subplot(gs[0, 0])
    plot_queries_vs_success(ax1, by_type)
    
    # Panel 2: Time vs Success  
    ax2 = fig.add_subplot(gs[0, 1])
    plot_time_vs_success(ax2, by_type)
    
    # Panel 3: Cost Efficiency Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    plot_efficiency_comparison(ax3, by_type)
    
    # Panel 4: Cumulative Success
    ax4 = fig.add_subplot(gs[1, 1])
    plot_cumulative_success(ax4, cumulative)
    
    plt.suptitle("Attack Cost & Efficiency Analysis", fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_queries_vs_success(ax, by_type: Dict[str, Dict[str, float]]):
    """Plot queries per success vs success rate."""
    attack_types = list(by_type.keys())
    queries = [by_type[t].get("queries_per_success", 0) for t in attack_types]
    success_rates = [by_type[t].get("success_rate", 0) * 100 for t in attack_types]
    
    # Filter out infinite values
    valid_data = [(t, q, s) for t, q, s in zip(attack_types, queries, success_rates) 
                  if q != float('inf') and q < 1000]
    
    if not valid_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return
    
    attack_types, queries, success_rates = zip(*valid_data)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(attack_types)))
    
    ax.scatter(queries, success_rates, s=200, c=colors, alpha=0.7, edgecolors='black', linewidths=1.5)
    
    # Add labels
    for i, (t, q, s) in enumerate(zip(attack_types, queries, success_rates)):
        ax.annotate(t, (q, s), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel("Queries per Success", fontsize=11, fontweight='bold')
    ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight='bold')
    ax.set_title("Efficiency: Queries vs Success Rate", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add efficiency region annotation
    ax.axhspan(80, 100, alpha=0.1, color='green', label='High Success')
    ax.legend(loc='best', fontsize=9)


def plot_time_vs_success(ax, by_type: Dict[str, Dict[str, float]]):
    """Plot time per success for each attack type."""
    attack_types = list(by_type.keys())
    times = [by_type[t].get("time_per_success", 0) for t in attack_types]
    success_rates = [by_type[t].get("success_rate", 0) * 100 for t in attack_types]
    
    # Filter out infinite values
    valid_data = [(t, time, s) for t, time, s in zip(attack_types, times, success_rates) 
                  if time != float('inf') and time < 1000]
    
    if not valid_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return
    
    attack_types, times, success_rates = zip(*valid_data)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(attack_types)))
    
    bars = ax.bar(attack_types, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add success rate as text on bars
    for bar, sr in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sr:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel("Attack Type", fontsize=11, fontweight='bold')
    ax.set_ylabel("Time per Success (s)", fontsize=11, fontweight='bold')
    ax.set_title("Time Efficiency by Attack Type", fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)


def plot_efficiency_comparison(ax, by_type: Dict[str, Dict[str, float]]):
    """Plot multi-dimensional efficiency comparison."""
    attack_types = list(by_type.keys())
    
    if not attack_types:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return
    
    # Normalize metrics for comparison
    success_rates = np.array([by_type[t].get("success_rate", 0) for t in attack_types])
    
    # Inverse of costs (lower is better -> higher normalized score)
    queries = np.array([by_type[t].get("queries_per_success", 1) for t in attack_types])
    queries = np.where(queries == float('inf'), 1000, queries)
    query_efficiency = 1.0 / (queries + 1)  # Normalize
    
    times = np.array([by_type[t].get("time_per_success", 1) for t in attack_types])
    times = np.where(times == float('inf'), 1000, times)
    time_efficiency = 1.0 / (times + 1)
    
    x = np.arange(len(attack_types))
    width = 0.25
    
    bars1 = ax.bar(x - width, success_rates, width, label='Success Rate', color='green', alpha=0.7)
    bars2 = ax.bar(x, query_efficiency * max(success_rates), width, 
                   label='Query Efficiency', color='blue', alpha=0.7)
    bars3 = ax.bar(x + width, time_efficiency * max(success_rates), width, 
                   label='Time Efficiency', color='orange', alpha=0.7)
    
    ax.set_xlabel("Attack Type", fontsize=11, fontweight='bold')
    ax.set_ylabel("Normalized Score", fontsize=11, fontweight='bold')
    ax.set_title("Multi-Dimensional Efficiency Comparison", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


def plot_cumulative_success(ax, cumulative_data: List[Dict[str, Any]]):
    """Plot cumulative success over time."""
    if not cumulative_data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return
    
    timestamps = [d["timestamp"] for d in cumulative_data]
    successes = [d["cumulative_successes"] for d in cumulative_data]
    attempts = [d["cumulative_attempts"] for d in cumulative_data]
    
    # Plot
    ax.plot(timestamps, successes, 'g-', linewidth=2, label='Successful Attacks')
    ax.fill_between(timestamps, 0, successes, alpha=0.3, color='green')
    
    ax.plot(timestamps, attempts, 'b--', linewidth=1.5, alpha=0.7, label='Total Attempts')
    
    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax.set_title("Cumulative Success Over Time", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add success rate annotation
    if attempts[-1] > 0:
        final_rate = (successes[-1] / attempts[-1]) * 100
        ax.text(0.65, 0.95, f'Final Success Rate: {final_rate:.1f}%',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_cost_breakdown(
    by_type: Dict[str, Dict[str, float]],
    output_path: str
):
    """
    Plot detailed cost breakdown by attack type.
    
    Args:
        by_type: Per-type metrics
        output_path: Path to save chart
    """
    attack_types = list(by_type.keys())
    
    if not attack_types:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Total queries by type (filter out zero queries to avoid NaN)
    queries_raw = [by_type[t].get("queries", 0) for t in attack_types]
    valid_indices = [i for i, q in enumerate(queries_raw) if q > 0]
    
    if valid_indices:
        queries = [queries_raw[i] / 1000 for i in valid_indices]
        query_labels = [attack_types[i] for i in valid_indices]
        colors = plt.cm.Set3(np.arange(len(queries)))
        
        ax1.pie(queries, labels=query_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title("Query Distribution by Attack Type", fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, "No query data available", ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Query Distribution by Attack Type", fontsize=12, fontweight='bold')
    
    # Success vs Failure counts
    successes = [by_type[t].get("total_successes", 0) for t in attack_types]
    failures = [by_type[t].get("total_attempts", 0) - by_type[t].get("total_successes", 0) 
                for t in attack_types]
    
    x = np.arange(len(attack_types))
    width = 0.35
    
    ax2.bar(x - width/2, successes, width, label='Success', color='green', alpha=0.7)
    ax2.bar(x + width/2, failures, width, label='Failure', color='red', alpha=0.7)
    
    ax2.set_xlabel("Attack Type", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax2.set_title("Success vs Failure by Attack Type", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(attack_types, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_cost_report(
    cost_tracker,
    output_dir: str
) -> Dict[str, str]:
    """
    Generate complete cost visualization report.
    
    Args:
        cost_tracker: AttackCostTracker instance
        output_dir: Directory to save visualizations
        
    Returns:
        Dict mapping chart names to file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    cost_data = {
        "efficiency_by_type": cost_tracker.compute_efficiency_by_type(),
        "cumulative_success": cost_tracker.get_cumulative_success_over_time(),
    }
    
    # Generate charts
    paths = {}
    
    main_path = os.path.join(output_dir, "cost_efficiency_analysis.png")
    plot_cost_efficiency(cost_data, main_path)
    paths["main_analysis"] = main_path
    
    breakdown_path = os.path.join(output_dir, "cost_breakdown.png")
    plot_cost_breakdown(cost_data["efficiency_by_type"], breakdown_path)
    paths["breakdown"] = breakdown_path
    
    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "cost_metrics.json")
    cost_tracker.save_metrics(metrics_path)
    paths["metrics_json"] = metrics_path
    
    return paths
