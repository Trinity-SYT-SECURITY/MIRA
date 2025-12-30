"""
Comprehensive Visualization Methods for Research Report.

Generates detailed visualizations for academic-quality multi-model comparison:
- Layer activation heatmaps
- Attention pattern analysis
- Probe accuracy curves
- Logit lens evolution
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class ComprehensiveVisualizer:
    """Generate comprehensive visualizations for research report."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.colors = {
            "primary": "#818cf8",
            "success": "#22c55e",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "info": "#06b6d4",
        }
    
    def generate_layer_activation_heatmap(
        self,
        model_results: List[Dict[str, Any]],
        max_layers: int = 32
    ) -> str:
        """
        Generate comprehensive layer activation heatmap for all models.
        
        Shows: All layers × All models with clean vs attack comparison
        """
        if not model_results:
            return "<p>No layer activation data available.</p>"
        
        # Build heatmap HTML
        rows_html = ""
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            layer_acts = result.get("layer_activations", {})
            clean_acts = layer_acts.get("clean", [])
            attack_acts = layer_acts.get("attack", [])
            
            if not clean_acts or not attack_acts:
                continue
            
            # Build cells for each layer
            cells_html = ""
            max_val = max(max(clean_acts) if clean_acts else 0, max(attack_acts) if attack_acts else 0)
            
            for i in range(min(len(clean_acts), max_layers)):
                clean_val = clean_acts[i]
                attack_val = attack_acts[i] if i < len(attack_acts) else 0
                
                # Calculate difference
                diff = attack_val - clean_val
                diff_pct = (diff / clean_val * 100) if clean_val > 0 else 0
                
                # Color based on difference
                if abs(diff_pct) < 10:
                    cell_class = "neutral"
                elif diff_pct > 0:
                    cell_class = "increased"
                else:
                    cell_class = "decreased"
                
                # Intensity based on absolute value
                intensity = min(attack_val / max_val, 1.0) if max_val > 0 else 0
                
                cells_html += f'''
                <td class="heatmap-cell {cell_class}" 
                    style="opacity: {0.3 + intensity * 0.7};"
                    title="Layer {i}&#10;Clean: {clean_val:.3f}&#10;Attack: {attack_val:.3f}&#10;Diff: {diff_pct:+.1f}%">
                    <span class="cell-value">{attack_val:.2f}</span>
                </td>
'''
            
            rows_html += f'''
            <tr>
                <th class="row-header">{model_name}</th>
                {cells_html}
            </tr>
'''
        
        # Generate column headers
        col_headers = '<tr><th>Model</th>'
        for i in range(max_layers):
            col_headers += f'<th class="layer-header">L{i}</th>'
        col_headers += '</tr>'
        
        html = f'''
        <div class="layer-heatmap">
            <style>
                .layer-heatmap table {{
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 0.75rem;
                    width: 100%;
                    overflow-x: auto;
                }}
                .layer-heatmap th, .layer-heatmap td {{
                    padding: 8px 4px;
                    text-align: center;
                    border: 1px solid var(--border-color, #333);
                    min-width: 40px;
                }}
                .layer-heatmap th {{
                    background: var(--bg-secondary, #2a2a4e);
                    font-weight: 600;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                .layer-heatmap .row-header {{
                    position: sticky;
                    left: 0;
                    background: var(--bg-secondary, #2a2a4e);
                    z-index: 11;
                    text-align: left;
                    max-width: 150px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }}
                .layer-heatmap .heatmap-cell {{
                    cursor: help;
                    transition: all 0.2s;
                }}
                .layer-heatmap .heatmap-cell:hover {{
                    transform: scale(1.1);
                    z-index: 100;
                    box-shadow: 0 0 10px rgba(129, 140, 248, 0.5);
                }}
                .layer-heatmap .neutral {{
                    background: rgba(100, 100, 100, 0.3);
                }}
                .layer-heatmap .increased {{
                    background: rgba(239, 68, 68, 0.5);
                }}
                .layer-heatmap .decreased {{
                    background: rgba(34, 197, 94, 0.5);
                }}
                .layer-heatmap .cell-value {{
                    font-size: 0.7rem;
                    font-weight: 500;
                }}
            </style>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        {col_headers}
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            <div class="legend" style="margin-top: 16px; font-size: 0.85rem;">
                <strong>Legend:</strong>
                <span style="color: #ef4444;">■</span> Increased activation (attack > clean) |
                <span style="color: #22c55e;">■</span> Decreased activation (attack < clean) |
                <span style="color: #666;">■</span> Neutral (< 10% change)
                <br><strong>Opacity:</strong> Darker = higher absolute activation value
            </div>
        </div>
'''
        
        return html
    
    def generate_attention_difference_maps(
        self,
        model_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate attention difference heatmaps (attack - clean).
        
        Shows how attention patterns change during attacks.
        """
        if not model_results:
            return "<p>No attention data available.</p>"
        
        sections_html = ""
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            attention_data = result.get("attention_data", {})
            clean_attn = attention_data.get("baseline_clean")
            attack_attn = attention_data.get("baseline_attack")
            
            if not clean_attn or not attack_attn:
                continue
            
            # Calculate difference
            try:
                clean_arr = np.array(clean_attn)
                attack_arr = np.array(attack_attn)
                
                if clean_arr.shape != attack_arr.shape:
                    continue
                
                diff_arr = attack_arr - clean_arr
                
                # Generate heatmap
                heatmap_html = self._generate_attention_heatmap_html(
                    diff_arr,
                    title=f"{model_name} - Attention Difference (Attack - Clean)",
                    is_difference=True
                )
                
                sections_html += f'''
                <div class="attention-section">
                    <h4>{model_name}</h4>
                    {heatmap_html}
                </div>
'''
            except Exception:
                continue
        
        if not sections_html:
            return "<p>No attention difference data available.</p>"
        
        return f'''
        <div class="attention-differences">
            <style>
                .attention-section {{
                    margin: 24px 0;
                    padding: 16px;
                    background: var(--bg-tertiary, #1a1a2e);
                    border-radius: 8px;
                }}
                .attention-section h4 {{
                    margin-top: 0;
                    color: var(--text-primary, #e0e0e0);
                }}
            </style>
            {sections_html}
        </div>
'''
    
    def _generate_attention_heatmap_html(
        self,
        attention_matrix: np.ndarray,
        title: str = "Attention Pattern",
        is_difference: bool = False
    ) -> str:
        """Generate HTML heatmap for attention matrix."""
        if attention_matrix.size == 0:
            return "<p>No data</p>"
        
        # Normalize for visualization
        if is_difference:
            # Center around 0 for difference maps
            vmax = max(abs(attention_matrix.min()), abs(attention_matrix.max()))
            vmin = -vmax
        else:
            vmin = attention_matrix.min()
            vmax = attention_matrix.max()
        
        rows, cols = attention_matrix.shape
        cells_html = ""
        
        for i in range(min(rows, 16)):  # Limit to 16x16 for readability
            row_html = "<tr>"
            for j in range(min(cols, 16)):
                val = attention_matrix[i, j]
                
                # Normalize to 0-1
                if vmax != vmin:
                    normalized = (val - vmin) / (vmax - vmin)
                else:
                    normalized = 0.5
                
                # Color based on value
                if is_difference:
                    if val > 0:
                        color = f"rgba(239, 68, 68, {normalized})"  # Red for increase
                    else:
                        color = f"rgba(34, 197, 94, {1-normalized})"  # Green for decrease
                else:
                    color = f"rgba(129, 140, 248, {normalized})"  # Blue for attention
                
                row_html += f'''
                <td style="background: {color}; padding: 8px; border: 1px solid #333; min-width: 30px; text-align: center; font-size: 0.7rem;"
                    title="[{i},{j}]: {val:.4f}">
                    {val:.2f}
                </td>
'''
            row_html += "</tr>"
            cells_html += row_html
        
        return f'''
        <div class="attention-heatmap">
            <p style="font-weight: 600; margin-bottom: 8px;">{title}</p>
            <div style="overflow-x: auto;">
                <table style="border-collapse: collapse; font-size: 0.75rem;">
                    {cells_html}
                </table>
            </div>
        </div>
'''
    
    def generate_probe_accuracy_curves(
        self,
        model_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate per-layer probe accuracy curves.
        
        Shows how well probes can detect attacks at each layer.
        """
        # This would require per-layer probe data
        # For now, return placeholder
        return '''
        <div class="probe-curves">
            <p class="section-description">
                Per-layer probe accuracy analysis requires additional data collection.
                Future enhancement: Train probes at each layer to measure attack signal emergence.
            </p>
        </div>
'''
    
    def generate_token_probability_evolution(
        self,
        logit_lens_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate token probability evolution across layers.
        
        Shows how token probabilities change through the model.
        """
        if not logit_lens_results:
            return "<p>No logit lens data available.</p>"
        
        # Placeholder for now - would need full logit lens data
        return '''
        <div class="logit-evolution">
            <p class="section-description">
                Token probability evolution visualization requires full logit lens analysis.
                Future enhancement: Track top-k token probabilities across all layers.
            </p>
        </div>
'''
