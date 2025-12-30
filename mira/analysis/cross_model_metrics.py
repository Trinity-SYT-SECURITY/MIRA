"""
Cross-Model Internal Metrics Analysis.

Compares internal model states across models:
- Refusal direction similarity
- Entropy patterns
- Layer divergence points
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine


class CrossModelAnalyzer:
    """Compare internal model states across models."""
    
    def __init__(self):
        """Initialize cross-model analyzer."""
        pass
    
    def compute_refusal_direction_similarity(
        self,
        model_results: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute cosine similarity matrix of refusal directions.
        
        Args:
            model_results: List of result dicts with internal_metrics
        
        Returns:
            (similarity_matrix, model_names)
            similarity_matrix: NxN numpy array of cosine similarities
            model_names: List of model names in order
        """
        # Extract refusal directions
        refusal_dirs = []
        model_names = []
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            internal = result.get("internal_metrics", {})
            refusal_dir = internal.get("refusal_direction")
            
            if refusal_dir is not None:
                refusal_dirs.append(np.array(refusal_dir))
                model_names.append(result.get("model_name", "unknown"))
        
        if len(refusal_dirs) < 2:
            return np.array([[1.0]]), model_names
        
        # Compute pairwise cosine similarity
        n = len(refusal_dirs)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Cosine similarity = 1 - cosine distance
                    similarity_matrix[i, j] = 1.0 - cosine(refusal_dirs[i], refusal_dirs[j])
        
        return similarity_matrix, model_names
    
    def analyze_entropy_patterns(
        self,
        model_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare entropy drop patterns across models.
        
        Args:
            model_results: List of result dicts with internal_metrics
        
        Returns:
            {
                "model_name": {
                    "delta_entropy_success": mean,
                    "delta_entropy_failure": mean,
                    "entropy_collapse_ratio": ratio,
                    "statistical_significance": p_value,
                },
                "summary": {
                    "models_with_collapse": count,
                    "mean_collapse_ratio": value,
                }
            }
        """
        entropy_analysis = {}
        collapse_ratios = []
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            internal = result.get("internal_metrics", {})
            entropy_by_attack = internal.get("entropy_by_attack", {})
            
            success_entropies = entropy_by_attack.get("successful", [])
            failed_entropies = entropy_by_attack.get("failed", [])
            
            if not success_entropies or not failed_entropies:
                continue
            
            # Calculate mean delta entropy
            delta_success = np.mean(success_entropies) if success_entropies else 0.0
            delta_failure = np.mean(failed_entropies) if failed_entropies else 0.0
            
            # Entropy collapse ratio (how much more entropy drops on success)
            if delta_failure != 0:
                collapse_ratio = abs(delta_success / delta_failure)
            else:
                collapse_ratio = abs(delta_success) if delta_success != 0 else 1.0
            
            # Statistical significance test
            if len(success_entropies) > 1 and len(failed_entropies) > 1:
                _, p_value = stats.ttest_ind(success_entropies, failed_entropies)
            else:
                p_value = 1.0
            
            entropy_analysis[model_name] = {
                "delta_entropy_success": float(delta_success),
                "delta_entropy_failure": float(delta_failure),
                "entropy_collapse_ratio": float(collapse_ratio),
                "statistical_significance": float(p_value),
                "significant": p_value < 0.05,
            }
            
            collapse_ratios.append(collapse_ratio)
        
        # Summary statistics
        entropy_analysis["summary"] = {
            "models_with_collapse": sum(
                1 for data in entropy_analysis.values()
                if isinstance(data, dict) and data.get("entropy_collapse_ratio", 1.0) > 2.0
            ),
            "mean_collapse_ratio": float(np.mean(collapse_ratios)) if collapse_ratios else 1.0,
            "models_analyzed": len(collapse_ratios),
        }
        
        return entropy_analysis
    
    def identify_common_failure_layers(
        self,
        model_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Find if models fail at similar layer depths.
        
        Args:
            model_results: List of result dicts with internal_metrics
        
        Returns:
            {
                "model_name": {
                    "divergence_layer": int,
                    "divergence_depth_ratio": float,  # 0-1
                },
                "summary": {
                    "common_divergence_range": (min, max),
                    "clustering": bool,
                }
            }
        """
        layer_analysis = {}
        divergence_ratios = []
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            internal = result.get("internal_metrics", {})
            
            divergence_layer = internal.get("layer_divergence_point", -1)
            
            # Get model layer count from layer_activations
            layer_acts = result.get("layer_activations", {})
            clean_acts = layer_acts.get("clean", [])
            n_layers = len(clean_acts) if clean_acts else 12  # Default
            
            if divergence_layer >= 0 and n_layers > 0:
                divergence_ratio = divergence_layer / n_layers
                
                layer_analysis[model_name] = {
                    "divergence_layer": divergence_layer,
                    "total_layers": n_layers,
                    "divergence_depth_ratio": float(divergence_ratio),
                }
                
                divergence_ratios.append(divergence_ratio)
        
        # Check for clustering
        if len(divergence_ratios) >= 2:
            std_dev = float(np.std(divergence_ratios))
            mean_ratio = float(np.mean(divergence_ratios))
            clustering = std_dev < 0.2  # Tight clustering if std < 0.2
            
            layer_analysis["summary"] = {
                "common_divergence_range": (
                    float(np.min(divergence_ratios)),
                    float(np.max(divergence_ratios))
                ),
                "mean_divergence_ratio": mean_ratio,
                "std_divergence_ratio": std_dev,
                "clustering": clustering,
                "models_analyzed": len(divergence_ratios),
            }
        else:
            layer_analysis["summary"] = {
                "models_analyzed": len(divergence_ratios),
                "clustering": False,
            }
        
        return layer_analysis
    
    def generate_similarity_matrix_html(
        self,
        similarity_matrix: np.ndarray,
        model_names: List[str]
    ) -> str:
        """
        Generate HTML heatmap for refusal direction similarity matrix.
        
        Args:
            similarity_matrix: NxN similarity matrix
            model_names: List of model names
        
        Returns:
            HTML string with heatmap
        """
        if similarity_matrix.shape[0] == 0:
            return "<p>No similarity data available.</p>"
        
        n = len(model_names)
        
        # Build heatmap cells
        cells_html = ""
        for i in range(n):
            row_html = f'<tr><th class="row-header">{model_names[i]}</th>'
            for j in range(n):
                sim = similarity_matrix[i, j]
                # Color intensity based on similarity
                if i == j:
                    cell_class = "diagonal"
                elif sim > 0.8:
                    cell_class = "high-sim"
                elif sim > 0.5:
                    cell_class = "med-sim"
                else:
                    cell_class = "low-sim"
                
                row_html += f'<td class="sim-cell {cell_class}" title="{model_names[i]} vs {model_names[j]}: {sim:.3f}">{sim:.2f}</td>'
            row_html += '</tr>'
            cells_html += row_html
        
        # Build column headers
        col_headers = '<tr><th></th>'
        for name in model_names:
            col_headers += f'<th class="col-header">{name}</th>'
        col_headers += '</tr>'
        
        html = f'''
        <div class="similarity-matrix">
            <style>
                .similarity-matrix table {{
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 0.85rem;
                }}
                .similarity-matrix th, .similarity-matrix td {{
                    padding: 10px;
                    text-align: center;
                    border: 1px solid var(--border-color, #333);
                }}
                .similarity-matrix th {{
                    background: var(--bg-secondary, #2a2a4e);
                    font-weight: 600;
                }}
                .similarity-matrix .row-header, .similarity-matrix .col-header {{
                    font-size: 0.8rem;
                    max-width: 120px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}
                .similarity-matrix .sim-cell {{
                    cursor: help;
                    font-weight: 500;
                }}
                .similarity-matrix .diagonal {{
                    background: rgba(129, 140, 248, 0.3);
                }}
                .similarity-matrix .high-sim {{
                    background: rgba(34, 197, 94, 0.3);
                }}
                .similarity-matrix .med-sim {{
                    background: rgba(245, 158, 11, 0.2);
                }}
                .similarity-matrix .low-sim {{
                    background: rgba(239, 68, 68, 0.2);
                }}
            </style>
            <table>
                <thead>
                    {col_headers}
                </thead>
                <tbody>
                    {cells_html}
                </tbody>
            </table>
            <div class="legend" style="margin-top: 16px; font-size: 0.85rem; color: var(--text-secondary, #999);">
                <strong>Cosine Similarity:</strong>
                1.0 = Identical directions |
                &gt;0.8 = Highly similar |
                0.5-0.8 = Moderately similar |
                &lt;0.5 = Different
            </div>
        </div>
'''
        
        return html
    
    def generate_entropy_comparison_html(
        self,
        entropy_analysis: Dict[str, Any]
    ) -> str:
        """Generate HTML visualization for entropy pattern comparison."""
        if not entropy_analysis or "summary" not in entropy_analysis:
            return "<p>No entropy data available.</p>"
        
        # Build table rows
        rows_html = ""
        for model_name, data in entropy_analysis.items():
            if model_name == "summary" or not isinstance(data, dict):
                continue
            
            delta_success = data.get("delta_entropy_success", 0.0)
            delta_failure = data.get("delta_entropy_failure", 0.0)
            collapse_ratio = data.get("entropy_collapse_ratio", 1.0)
            p_value = data.get("statistical_significance", 1.0)
            significant = data.get("significant", False)
            
            sig_marker = "✓" if significant else "–"
            sig_class = "significant" if significant else ""
            
            rows_html += f'''
            <tr class="{sig_class}">
                <td><strong>{model_name}</strong></td>
                <td>{delta_success:.3f}</td>
                <td>{delta_failure:.3f}</td>
                <td class="collapse-ratio">{collapse_ratio:.2f}x</td>
                <td>{p_value:.4f}</td>
                <td class="sig-marker">{sig_marker}</td>
            </tr>
'''
        
        summary = entropy_analysis.get("summary", {})
        models_with_collapse = summary.get("models_with_collapse", 0)
        mean_collapse = summary.get("mean_collapse_ratio", 1.0)
        
        html = f'''
        <div class="entropy-comparison">
            <style>
                .entropy-comparison table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 0.9rem;
                }}
                .entropy-comparison th, .entropy-comparison td {{
                    padding: 12px;
                    text-align: center;
                    border: 1px solid var(--border-color, #333);
                }}
                .entropy-comparison th {{
                    background: var(--bg-secondary, #2a2a4e);
                    font-weight: 600;
                }}
                .entropy-comparison tr.significant {{
                    background: rgba(34, 197, 94, 0.1);
                }}
                .entropy-comparison .collapse-ratio {{
                    font-weight: 600;
                    color: var(--warning, #f59e0b);
                }}
                .entropy-comparison .sig-marker {{
                    font-size: 1.2rem;
                    color: var(--success, #22c55e);
                }}
            </style>
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>ΔEntropy<br>(Success)</th>
                        <th>ΔEntropy<br>(Failure)</th>
                        <th>Collapse<br>Ratio</th>
                        <th>p-value</th>
                        <th>Sig.</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <div class="summary" style="margin-top: 16px; padding: 16px; background: var(--bg-tertiary, #1a1a2e); border-radius: 8px;">
                <strong>Key Finding:</strong>
                {models_with_collapse} of {summary.get("models_analyzed", 0)} models show entropy collapse (ratio &gt; 2.0) during successful attacks.
                Mean collapse ratio: {mean_collapse:.2f}x
            </div>
            <div class="legend" style="margin-top: 12px; font-size: 0.85rem; color: var(--text-secondary, #999);">
                <strong>Interpretation:</strong>
                Higher collapse ratio indicates successful attacks cause significantly more entropy drop than failed attacks.
                p &lt; 0.05 indicates statistical significance.
            </div>
        </div>
'''
        
        return html
