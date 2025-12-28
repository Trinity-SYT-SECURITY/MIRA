"""
Subspace visualization tools.

Provides visualization functions for:
- 2D/3D embedding space plots
- Subspace boundaries and directions
- Attack trajectory visualization
"""

from typing import List, Optional, Dict, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def plot_subspace_2d(
    safe_embeddings: torch.Tensor,
    unsafe_embeddings: torch.Tensor,
    refusal_direction: Optional[torch.Tensor] = None,
    title: str = "Subspace Visualization",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create 2D visualization of safe vs unsafe embedding clusters.
    
    Args:
        safe_embeddings: Safe prompt embeddings (n_safe, hidden_dim)
        unsafe_embeddings: Unsafe prompt embeddings (n_unsafe, hidden_dim)
        refusal_direction: Optional direction vector to display
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Combine embeddings for PCA
    all_embeds = torch.cat([safe_embeddings, unsafe_embeddings], dim=0)
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeds.cpu().numpy())
    
    n_safe = len(safe_embeddings)
    safe_2d = reduced[:n_safe]
    unsafe_2d = reduced[n_safe:]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    ax.scatter(
        safe_2d[:, 0], safe_2d[:, 1],
        c='#2ecc71', marker='o', s=100, alpha=0.7,
        label='Safe/Accepted', edgecolors='white', linewidth=1
    )
    ax.scatter(
        unsafe_2d[:, 0], unsafe_2d[:, 1],
        c='#e74c3c', marker='s', s=100, alpha=0.7,
        label='Unsafe/Refused', edgecolors='white', linewidth=1
    )
    
    # Plot direction if provided
    if refusal_direction is not None:
        # Project direction to 2D
        direction_np = refusal_direction.cpu().numpy()
        direction_2d = pca.transform(direction_np.reshape(1, -1))[0]
        
        # Normalize and scale for visibility
        direction_2d = direction_2d / (np.linalg.norm(direction_2d) + 1e-10)
        scale = max(abs(reduced).max() * 0.3, 1.0)
        
        # Draw arrow from center
        center = reduced.mean(axis=0)
        ax.annotate(
            '', xy=center + direction_2d * scale, xytext=center,
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=2)
        )
        ax.text(
            center[0] + direction_2d[0] * scale * 1.1,
            center[1] + direction_2d[1] * scale * 1.1,
            'Refusal Direction', fontsize=10, color='#3498db'
        )
    
    # Labels and styling
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_subspace_3d(
    safe_embeddings: torch.Tensor,
    unsafe_embeddings: torch.Tensor,
    title: str = "3D Subspace Visualization",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create 3D visualization of embedding clusters.
    
    Args:
        safe_embeddings: Safe prompt embeddings
        unsafe_embeddings: Unsafe prompt embeddings
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    all_embeds = torch.cat([safe_embeddings, unsafe_embeddings], dim=0)
    
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(all_embeds.cpu().numpy())
    
    n_safe = len(safe_embeddings)
    safe_3d = reduced[:n_safe]
    unsafe_3d = reduced[n_safe:]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        safe_3d[:, 0], safe_3d[:, 1], safe_3d[:, 2],
        c='#2ecc71', marker='o', s=80, alpha=0.7,
        label='Safe/Accepted'
    )
    ax.scatter(
        unsafe_3d[:, 0], unsafe_3d[:, 1], unsafe_3d[:, 2],
        c='#e74c3c', marker='s', s=80, alpha=0.7,
        label='Unsafe/Refused'
    )
    
    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_zlabel('PC3', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trajectory(
    trajectory_embeddings: List[torch.Tensor],
    labels: Optional[List[str]] = None,
    title: str = "Attack Trajectory",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the trajectory of embeddings during an attack.
    
    Args:
        trajectory_embeddings: List of embedding tensors at each step
        labels: Optional labels for each step
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    if len(trajectory_embeddings) < 2:
        raise ValueError("Need at least 2 embeddings for trajectory")
    
    # Stack and reduce
    all_embeds = torch.stack(trajectory_embeddings)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeds.cpu().numpy())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color gradient from green to red
    n_points = len(reduced)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_points))
    
    # Plot trajectory line
    ax.plot(reduced[:, 0], reduced[:, 1], 'k--', alpha=0.3, linewidth=1)
    
    # Plot points with gradient
    for i, (point, color) in enumerate(zip(reduced, colors)):
        marker = 'o' if i not in [0, n_points-1] else ('>' if i == n_points-1 else 's')
        ax.scatter(point[0], point[1], c=[color], marker=marker, s=150, zorder=5)
        
        if labels and i < len(labels):
            ax.annotate(
                labels[i], (point[0], point[1]),
                textcoords="offset points", xytext=(10, 5),
                fontsize=9
            )
    
    # Mark start and end
    ax.annotate(
        'Start', (reduced[0, 0], reduced[0, 1]),
        textcoords="offset points", xytext=(-15, -15),
        fontsize=10, fontweight='bold', color='#2ecc71'
    )
    ax.annotate(
        'End', (reduced[-1, 0], reduced[-1, 1]),
        textcoords="offset points", xytext=(10, 10),
        fontsize=10, fontweight='bold', color='#e74c3c'
    )
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_loss_curve(
    loss_history: List[float],
    title: str = "Attack Optimization",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the loss curve during attack optimization.
    
    Args:
        loss_history: List of loss values
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    steps = list(range(len(loss_history)))
    ax.plot(steps, loss_history, 'b-', linewidth=2, alpha=0.8)
    ax.fill_between(steps, loss_history, alpha=0.2)
    
    # Mark minimum
    min_idx = np.argmin(loss_history)
    min_val = loss_history[min_idx]
    ax.scatter([min_idx], [min_val], c='red', s=100, zorder=5, label=f'Min: {min_val:.4f}')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
