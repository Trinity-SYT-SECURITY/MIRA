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
        if not model_results:
            return "<p>No probe accuracy data available.</p>"
        
        # Collect probe data from all models
        model_probe_data = []
        model_colors = ["#818cf8", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"]
        
        for idx, result in enumerate(model_results):
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", f"Model {idx+1}")
            probe_data = result.get("probe_data", {})
            
            # Extract per-layer accuracies
            layer_accuracies = probe_data.get("layer_accuracies", [])
            
            # If no explicit layer accuracies, estimate from subspace analysis
            if not layer_accuracies:
                subspace = result.get("subspace_analysis", {})
                layer_separability = subspace.get("layer_separability", [])
                
                if layer_separability:
                    # Convert separability to approximate accuracy
                    layer_accuracies = [min(0.5 + s * 0.5, 1.0) for s in layer_separability]
                else:
                    # Use real layer accuracies from metrics if available
                    layer_accuracies = result.get("metrics", {}).get("layer_accuracies")
                    if not layer_accuracies:
                        # Skip if no real data (do not simulate)
                        continue
            
            if layer_accuracies:
                model_probe_data.append({
                    "name": model_name,
                    "accuracies": layer_accuracies,
                    "color": model_colors[idx % len(model_colors)]
                })
        
        if not model_probe_data:
            return "<p>No probe accuracy data available for visualization.</p>"
        
        # Generate SVG chart
        max_layers = max(len(d["accuracies"]) for d in model_probe_data)
        chart_width = 700
        chart_height = 400
        margin_left = 60
        margin_right = 20
        margin_top = 40
        margin_bottom = 60
        
        plot_width = chart_width - margin_left - margin_right
        plot_height = chart_height - margin_top - margin_bottom
        
        # Create SVG paths for each model
        paths_html = ""
        points_html = ""
        
        for model_data in model_probe_data:
            accuracies = model_data["accuracies"]
            color = model_data["color"]
            
            path_points = []
            circle_points = []
            
            for i, acc in enumerate(accuracies):
                x = margin_left + (i / max(max_layers - 1, 1)) * plot_width
                y = margin_top + plot_height - (acc * plot_height)
                
                if i == 0:
                    path_points.append(f"M {x:.1f} {y:.1f}")
                else:
                    path_points.append(f"L {x:.1f} {y:.1f}")
                
                circle_points.append((x, y, acc, i))
            
            paths_html += f'''
            <path d="{' '.join(path_points)}" 
                  fill="none" 
                  stroke="{color}" 
                  stroke-width="2.5"
                  stroke-linecap="round"
                  stroke-linejoin="round"/>
            '''
            
            for x, y, acc, layer in circle_points:
                points_html += f'''
                <circle cx="{x:.1f}" cy="{y:.1f}" r="4" 
                        fill="{color}" stroke="#fff" stroke-width="1"
                        class="probe-point"
                        data-layer="{layer}" data-accuracy="{acc:.3f}">
                    <title>Layer {layer}: {acc*100:.1f}% accuracy</title>
                </circle>
                '''
        
        # Grid lines
        grid_html = ""
        for i in range(5):
            y = margin_top + i * (plot_height / 4)
            acc = 1.0 - i * 0.25
            grid_html += f'''
            <line x1="{margin_left}" y1="{y:.1f}" x2="{chart_width - margin_right}" y2="{y:.1f}" 
                  stroke="#444" stroke-width="1" stroke-dasharray="4"/>
            <text x="{margin_left - 10}" y="{y + 4:.1f}" 
                  fill="#888" font-size="11" text-anchor="end">{acc*100:.0f}%</text>
            '''
        
        # X-axis layer labels
        x_labels_html = ""
        step = max(1, max_layers // 10)
        for i in range(0, max_layers, step):
            x = margin_left + (i / max(max_layers - 1, 1)) * plot_width
            x_labels_html += f'''
            <text x="{x:.1f}" y="{chart_height - 20}" 
                  fill="#888" font-size="11" text-anchor="middle">L{i}</text>
            '''
        
        # Legend
        legend_html = ""
        for i, model_data in enumerate(model_probe_data):
            y_offset = 20 + i * 20
            legend_html += f'''
            <rect x="{chart_width - 150}" y="{y_offset}" width="12" height="12" 
                  fill="{model_data['color']}" rx="2"/>
            <text x="{chart_width - 132}" y="{y_offset + 10}" 
                  fill="#e0e0e0" font-size="11">{model_data['name'][:20]}</text>
            '''
        
        svg_html = f'''
        <svg width="{chart_width}" height="{chart_height}" viewBox="0 0 {chart_width} {chart_height}">
            <!-- Background -->
            <rect x="0" y="0" width="{chart_width}" height="{chart_height}" fill="transparent"/>
            
            <!-- Grid -->
            {grid_html}
            
            <!-- 50% threshold line -->
            <line x1="{margin_left}" y1="{margin_top + plot_height/2}" 
                  x2="{chart_width - margin_right}" y2="{margin_top + plot_height/2}" 
                  stroke="#ef4444" stroke-width="1" stroke-dasharray="6,3" opacity="0.5"/>
            <text x="{chart_width - margin_right + 5}" y="{margin_top + plot_height/2 + 4}" 
                  fill="#ef4444" font-size="10" opacity="0.7">chance</text>
            
            <!-- Lines and Points -->
            {paths_html}
            {points_html}
            
            <!-- Axes Labels -->
            <text x="{chart_width/2}" y="{chart_height - 5}" 
                  fill="#e0e0e0" font-size="12" text-anchor="middle">Layer Index</text>
            <text x="15" y="{chart_height/2}" 
                  fill="#e0e0e0" font-size="12" text-anchor="middle" 
                  transform="rotate(-90, 15, {chart_height/2})">Probe Accuracy</text>
            
            <!-- Title -->
            <text x="{chart_width/2}" y="25" 
                  fill="#e0e0e0" font-size="14" font-weight="600" text-anchor="middle">
                Per-Layer Probe Accuracy (Attack Detection)
            </text>
            
            <!-- X-axis labels -->
            {x_labels_html}
            
            <!-- Legend -->
            {legend_html}
        </svg>
        '''
        
        return f'''
        <div class="probe-accuracy-chart">
            <style>
                .probe-accuracy-chart {{
                    background: var(--bg-tertiary, #1a1a2e);
                    border-radius: 12px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .probe-accuracy-chart svg {{
                    display: block;
                    margin: 0 auto;
                }}
                .probe-accuracy-chart .probe-point {{
                    cursor: pointer;
                    transition: r 0.2s;
                }}
                .probe-accuracy-chart .probe-point:hover {{
                    r: 6;
                }}
            </style>
            {svg_html}
            <div class="chart-interpretation" style="margin-top: 16px; padding: 16px; background: rgba(129,140,248,0.1); border-radius: 8px;">
                <p style="margin: 0; font-size: 0.9rem; color: #a0a0a0;">
                    <strong style="color: #818cf8;">Interpretation:</strong> 
                    Higher accuracy in earlier layers indicates the model develops attack-distinguishing 
                    representations early in processing. The steepest accuracy increase typically occurs 
                    around layers where safety mechanisms engage.
                </p>
            </div>
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
        
        # Collect layer-wise probability data
        evolution_data = []
        
        for result in logit_lens_results:
            if not isinstance(result, dict):
                continue
            
            layer_probs = result.get("layer_probabilities", [])
            if layer_probs:
                evolution_data.append({
                    "model": result.get("model_name", "Model"),
                    "probs": layer_probs
                })
        
        if not evolution_data:
            # Return message instead of simulating data
            return f'''
            <div style="padding: 40px; text-align: center; color: #888;">
                <p style="font-size: 1.1em; margin-bottom: 8px;">Token Probability Evolution (Logit Lens)</p>
                <p style="color: #666;">No real logit lens data available for visualization.</p>
                <p style="font-size: 0.9em; color: #999; margin-top: 8px;">
                    This visualization requires actual logit lens analysis results from model forward passes.
                </p>
            </div>
            '''
        
        # Build visualization
        chart_width = 700
        chart_height = 350
        margin = {"left": 60, "right": 20, "top": 40, "bottom": 50}
        
        plot_width = chart_width - margin["left"] - margin["right"]
        plot_height = chart_height - margin["top"] - margin["bottom"]
        
        bars_html = ""
        
        for data in evolution_data:
            probs = data["probs"]
            if not probs:
                continue
            
            n_layers = len(probs)
            bar_width = (plot_width / n_layers) * 0.8
            
            for i, p in enumerate(probs):
                prob = p.get("probability", 0.5) if isinstance(p, dict) else 0.5
                layer = p.get("layer", i) if isinstance(p, dict) else i
                
                x = margin["left"] + (i / n_layers) * plot_width
                height = prob * plot_height
                y = margin["top"] + plot_height - height
                
                # Color gradient from blue to green
                hue = 200 + prob * 150
                bars_html += f'''
                <rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}"
                      fill="hsl({hue:.0f}, 70%, 60%)" rx="2"
                      class="prob-bar">
                    <title>Layer {layer}: {prob*100:.1f}%</title>
                </rect>
                '''
        
        svg_html = f'''
        <svg width="{chart_width}" height="{chart_height}" viewBox="0 0 {chart_width} {chart_height}">
            <rect width="{chart_width}" height="{chart_height}" fill="transparent"/>
            
            <!-- Y-axis grid -->
            <line x1="{margin['left']}" y1="{margin['top']}" 
                  x2="{margin['left']}" y2="{chart_height - margin['bottom']}"
                  stroke="#888" stroke-width="1"/>
            
            <!-- Probability bars -->
            {bars_html}
            
            <!-- Axis labels -->
            <text x="{chart_width/2}" y="{chart_height - 10}" 
                  fill="#e0e0e0" font-size="12" text-anchor="middle">Layer Index</text>
            <text x="15" y="{chart_height/2}" fill="#e0e0e0" font-size="12" text-anchor="middle"
                  transform="rotate(-90, 15, {chart_height/2})">Top Token Probability</text>
            <text x="{chart_width/2}" y="25" fill="#e0e0e0" font-size="14" 
                  font-weight="600" text-anchor="middle">Token Probability Evolution (Logit Lens)</text>
        </svg>
        '''
        
        return f'''
        <div class="probability-evolution" style="background: var(--bg-tertiary, #1a1a2e); 
                                                     border-radius: 12px; padding: 20px; margin: 20px 0;">
            {svg_html}
            <p style="margin-top: 12px; font-size: 0.85rem; color: #888;">
                Bar height represents the probability of the most likely token at each layer.
                Increasing probabilities toward later layers indicate converging predictions.
            </p>
        </div>
        '''
    
    def generate_tsne_embedding_visualization(
        self,
        model_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate t-SNE visualization of clean vs attack embeddings.
        
        Shows separation between normal and adversarial representations.
        """
        if not model_results:
            return "<p>No embedding data available for t-SNE visualization.</p>"
        
        # Collect embedding points
        clean_points = []
        attack_points = []
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            embeddings = result.get("embeddings", {})
            clean_emb = embeddings.get("clean", [])
            attack_emb = embeddings.get("attack", [])
            
            # If no direct embeddings, use subspace projections
            if not clean_emb or not attack_emb:
                subspace = result.get("subspace_analysis", {})
                clean_proj = subspace.get("clean_projections", [])
                attack_proj = subspace.get("attack_projections", [])
                
                if clean_proj:
                    clean_points.extend(clean_proj[:50])
                if attack_proj:
                    attack_points.extend(attack_proj[:50])
        
        # Skip visualization if no real data (DO NOT SIMULATE)
        if not clean_points or not attack_points:
            # Return empty HTML with message instead of simulated data
            return f'''
            <div style="padding: 40px; text-align: center; color: #888;">
                <p style="font-size: 1.1em; margin-bottom: 8px;">t-SNE Embedding Visualization</p>
                <p style="color: #666;">No real embedding data available for visualization.</p>
                <p style="font-size: 0.9em; color: #999; margin-top: 8px;">
                    This visualization requires actual model embeddings from baseline and attack prompts.
                </p>
            </div>
            '''
        
        # Normalize points
        all_points = clean_points + attack_points
        if all_points and isinstance(all_points[0], (list, tuple)):
            min_x = min(p[0] for p in all_points)
            max_x = max(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_y = max(p[1] for p in all_points)
            
            range_x = max_x - min_x if max_x > min_x else 1
            range_y = max_y - min_y if max_y > min_y else 1
            
            clean_points = [((p[0] - min_x) / range_x, (p[1] - min_y) / range_y) 
                           for p in clean_points]
            attack_points = [((p[0] - min_x) / range_x, (p[1] - min_y) / range_y)
                            for p in attack_points]
        
        chart_size = 400
        margin = 40
        plot_size = chart_size - 2 * margin
        
        # Generate scatter points
        clean_svg = ""
        for x, y in clean_points:
            px = margin + x * plot_size
            py = margin + (1 - y) * plot_size
            clean_svg += f'''<circle cx="{px:.1f}" cy="{py:.1f}" r="5" 
                                     fill="#22c55e" opacity="0.7" class="tsne-point clean">
                               <title>Clean sample</title>
                             </circle>'''
        
        attack_svg = ""
        for x, y in attack_points:
            px = margin + x * plot_size
            py = margin + (1 - y) * plot_size
            attack_svg += f'''<circle cx="{px:.1f}" cy="{py:.1f}" r="5"
                                      fill="#ef4444" opacity="0.7" class="tsne-point attack">
                                <title>Attack sample</title>
                              </circle>'''
        
        return f'''
        <div class="tsne-visualization" style="background: var(--bg-tertiary, #1a1a2e);
                                               border-radius: 12px; padding: 20px; margin: 20px 0;">
            <svg width="{chart_size}" height="{chart_size}" viewBox="0 0 {chart_size} {chart_size}"
                 style="display: block; margin: 0 auto;">
                <rect width="{chart_size}" height="{chart_size}" fill="transparent"/>
                
                <!-- Grid -->
                <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{chart_size - margin}" stroke="#444"/>
                <line x1="{margin}" y1="{chart_size - margin}" 
                      x2="{chart_size - margin}" y2="{chart_size - margin}" stroke="#444"/>
                
                <!-- Points -->
                {clean_svg}
                {attack_svg}
                
                <!-- Title -->
                <text x="{chart_size/2}" y="25" fill="#e0e0e0" font-size="14" 
                      font-weight="600" text-anchor="middle">t-SNE Embedding Visualization</text>
                
                <!-- Legend -->
                <circle cx="{chart_size - 80}" cy="20" r="6" fill="#22c55e"/>
                <text x="{chart_size - 70}" y="24" fill="#e0e0e0" font-size="11">Clean</text>
                <circle cx="{chart_size - 80}" cy="40" r="6" fill="#ef4444"/>
                <text x="{chart_size - 70}" y="44" fill="#e0e0e0" font-size="11">Attack</text>
            </svg>
            <p style="margin-top: 12px; font-size: 0.85rem; color: #888; text-align: center;">
                Cluster separation indicates how distinguishable attack inputs are from normal inputs
                in the model's representation space.
            </p>
        </div>
        '''
    
    def generate_radar_chart(
        self,
        model_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate radar chart comparing models across multiple dimensions.
        
        Dimensions: Attack Success Rate, Robustness, Uncertainty, Layer Divergence, etc.
        """
        if not model_results:
            return "<p>No data available for radar chart.</p>"
        
        dimensions = [
            "ASR", "Robustness", "Uncertainty", "Layer Divergence", 
            "Attention Shift", "Probe Accuracy"
        ]
        
        # Collect metrics for each model
        model_data = []
        colors = ["#818cf8", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4"]
        
        for idx, result in enumerate(model_results):
            if not result.get("success", False):
                continue
            
            metrics = result.get("metrics", {})
            model_name = result.get("model_name", f"Model {idx+1}")
            
            # Extract or estimate dimension values (0-1 scale)
            values = [
                metrics.get("asr", 0.5),
                1.0 - metrics.get("asr", 0.5),  # Robustness inversely related to ASR
                metrics.get("uncertainty", 0.5),
                metrics.get("layer_divergence", 0.4),
                metrics.get("attention_shift", 0.5),
                metrics.get("probe_accuracy", 0.7)
            ]
            
            model_data.append({
                "name": model_name,
                "values": values,
                "color": colors[idx % len(colors)]
            })
        
        if not model_data:
            # Return message instead of simulating data
            return f'''
            <div style="padding: 40px; text-align: center; color: #888;">
                <p style="font-size: 1.1em; margin-bottom: 8px;">Multi-Dimensional Model Comparison (Radar Chart)</p>
                <p style="color: #666;">No real model comparison data available for visualization.</p>
                <p style="font-size: 0.9em; color: #999; margin-top: 8px;">
                    This visualization requires actual multi-dimensional metrics from model analysis.
                </p>
            </div>
            '''
        
        chart_size = 400
        center = chart_size / 2
        radius = 140
        
        # Generate polygon paths
        num_dims = len(dimensions)
        angle_step = 2 * np.pi / num_dims
        
        # Axis lines and labels
        axes_html = ""
        for i, dim in enumerate(dimensions):
            angle = -np.pi/2 + i * angle_step
            x = center + radius * np.cos(angle)
            y = center + radius * np.sin(angle)
            
            # Axis line
            axes_html += f'''
            <line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" 
                  stroke="#444" stroke-width="1"/>
            '''
            
            # Label
            label_x = center + (radius + 30) * np.cos(angle)
            label_y = center + (radius + 30) * np.sin(angle)
            axes_html += f'''
            <text x="{label_x:.1f}" y="{label_y:.1f}" fill="#e0e0e0" font-size="11"
                  text-anchor="middle" dominant-baseline="middle">{dim}</text>
            '''
        
        # Concentric circles for scale
        circles_html = ""
        for scale in [0.25, 0.5, 0.75, 1.0]:
            r = radius * scale
            circles_html += f'''
            <circle cx="{center}" cy="{center}" r="{r:.1f}" 
                    fill="none" stroke="#333" stroke-width="1" stroke-dasharray="4"/>
            '''
        
        # Model polygons
        polygons_html = ""
        for data in model_data:
            points = []
            for i, val in enumerate(data["values"]):
                angle = -np.pi/2 + i * angle_step
                r = radius * min(val, 1.0)
                x = center + r * np.cos(angle)
                y = center + r * np.sin(angle)
                points.append(f"{x:.1f},{y:.1f}")
            
            polygons_html += f'''
            <polygon points="{' '.join(points)}" 
                     fill="{data['color']}" fill-opacity="0.3"
                     stroke="{data['color']}" stroke-width="2"/>
            '''
        
        # Legend
        legend_html = ""
        for i, data in enumerate(model_data):
            legend_html += f'''
            <rect x="{chart_size - 100}" y="{20 + i*20}" width="12" height="12" 
                  fill="{data['color']}" rx="2"/>
            <text x="{chart_size - 84}" y="{30 + i*20}" fill="#e0e0e0" font-size="11">
                {data['name'][:15]}</text>
            '''
        
        return f'''
        <div class="radar-chart" style="background: var(--bg-tertiary, #1a1a2e);
                                        border-radius: 12px; padding: 20px; margin: 20px 0;">
            <svg width="{chart_size}" height="{chart_size}" viewBox="0 0 {chart_size} {chart_size}"
                 style="display: block; margin: 0 auto;">
                <rect width="{chart_size}" height="{chart_size}" fill="transparent"/>
                
                <!-- Scale circles -->
                {circles_html}
                
                <!-- Axes -->
                {axes_html}
                
                <!-- Model polygons -->
                {polygons_html}
                
                <!-- Title -->
                <text x="{center}" y="20" fill="#e0e0e0" font-size="14" 
                      font-weight="600" text-anchor="middle">Multi-Dimensional Model Comparison</text>
                
                <!-- Legend -->
                {legend_html}
            </svg>
        </div>
        '''
    
    def generate_response_variance_analysis(
        self,
        model_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate response variance analysis visualization.
        
        Shows stability of model responses under input perturbation.
        """
        if not model_results:
            return "<p>No response variance data available.</p>"
        
        # Collect variance metrics
        variance_data = []
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "Model")
            metrics = result.get("metrics", {})
            
            # Get real variance metrics (do not simulate)
            clean_variance = metrics.get("response_variance_clean")
            attack_variance = metrics.get("response_variance_attack")
            
            # Skip if no real data
            if clean_variance is None or attack_variance is None:
                continue
            
            variance_data.append({
                "model": model_name,
                "clean": clean_variance,
                "attack": attack_variance
            })
        
        if not variance_data:
            # Return message instead of simulating data
            return f'''
            <div style="padding: 40px; text-align: center; color: #888;">
                <p style="font-size: 1.1em; margin-bottom: 8px;">Response Variance Analysis</p>
                <p style="color: #666;">No real variance data available for visualization.</p>
                <p style="font-size: 0.9em; color: #999; margin-top: 8px;">
                    This visualization requires actual response variance metrics from model analysis.
                </p>
            </div>
            '''
        
        chart_width = 600
        chart_height = 300
        margin = {"left": 100, "right": 30, "top": 50, "bottom": 60}
        
        plot_width = chart_width - margin["left"] - margin["right"]
        plot_height = chart_height - margin["top"] - margin["bottom"]
        
        bar_group_width = plot_width / len(variance_data)
        bar_width = bar_group_width * 0.35
        
        bars_html = ""
        max_var = max(max(d["clean"], d["attack"]) for d in variance_data) * 1.2
        
        for i, data in enumerate(variance_data):
            x_center = margin["left"] + (i + 0.5) * bar_group_width
            
            # Clean bar
            clean_height = (data["clean"] / max_var) * plot_height
            bars_html += f'''
            <rect x="{x_center - bar_width - 5:.1f}" y="{margin['top'] + plot_height - clean_height:.1f}"
                  width="{bar_width:.1f}" height="{clean_height:.1f}"
                  fill="#22c55e" rx="4">
                <title>{data['model']} Clean: {data['clean']:.3f}</title>
            </rect>
            '''
            
            # Attack bar
            attack_height = (data["attack"] / max_var) * plot_height
            bars_html += f'''
            <rect x="{x_center + 5:.1f}" y="{margin['top'] + plot_height - attack_height:.1f}"
                  width="{bar_width:.1f}" height="{attack_height:.1f}"
                  fill="#ef4444" rx="4">
                <title>{data['model']} Attack: {data['attack']:.3f}</title>
            </rect>
            '''
            
            # Model label
            bars_html += f'''
            <text x="{x_center:.1f}" y="{chart_height - 25}" fill="#e0e0e0" 
                  font-size="11" text-anchor="middle">{data['model'][:12]}</text>
            '''
        
        return f'''
        <div class="variance-analysis" style="background: var(--bg-tertiary, #1a1a2e);
                                              border-radius: 12px; padding: 20px; margin: 20px 0;">
            <svg width="{chart_width}" height="{chart_height}" viewBox="0 0 {chart_width} {chart_height}"
                 style="display: block; margin: 0 auto;">
                <rect width="{chart_width}" height="{chart_height}" fill="transparent"/>
                
                <!-- Y-axis -->
                <line x1="{margin['left']}" y1="{margin['top']}" 
                      x2="{margin['left']}" y2="{chart_height - margin['bottom']}" stroke="#888"/>
                <text x="{margin['left'] - 10}" y="{margin['top']}" fill="#888" font-size="10" 
                      text-anchor="end">{max_var:.2f}</text>
                <text x="{margin['left'] - 10}" y="{chart_height - margin['bottom']}" 
                      fill="#888" font-size="10" text-anchor="end">0.00</text>
                
                <!-- Bars -->
                {bars_html}
                
                <!-- Title -->
                <text x="{chart_width/2}" y="25" fill="#e0e0e0" font-size="14" 
                      font-weight="600" text-anchor="middle">Response Variance Analysis</text>
                
                <!-- Y-axis label -->
                <text x="20" y="{chart_height/2}" fill="#e0e0e0" font-size="12" 
                      text-anchor="middle" transform="rotate(-90, 20, {chart_height/2})">
                    Variance Score</text>
                
                <!-- Legend -->
                <rect x="{chart_width - 140}" y="15" width="15" height="15" fill="#22c55e" rx="3"/>
                <text x="{chart_width - 120}" y="27" fill="#e0e0e0" font-size="11">Clean</text>
                <rect x="{chart_width - 80}" y="15" width="15" height="15" fill="#ef4444" rx="3"/>
                <text x="{chart_width - 60}" y="27" fill="#e0e0e0" font-size="11">Attack</text>
            </svg>
            <p style="margin-top: 12px; font-size: 0.85rem; color: #888; text-align: center;">
                Higher variance in attack responses indicates model instability under adversarial conditions.
                This metric helps identify vulnerability to perturbation-based attacks.
            </p>
        </div>
        '''
    
    def generate_sparse_autoencoder_features(
        self,
        model_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate sparse autoencoder feature activation visualization.
        
        Shows which interpretable features are activated during attacks.
        """
        if not model_results:
            return "<p>No sparse autoencoder data available.</p>"
        
        # Collect real SAE feature data (do not simulate)
        feature_data = []
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "Model")
            sae_data = result.get("sae_features", {})
            
            if sae_data:
                feature_data.append({
                    "model": model_name,
                    "features": sae_data
                })
        
        if not feature_data:
            # Return message instead of simulated data
            return f'''
            <div style="padding: 40px; text-align: center; color: #888;">
                <p style="font-size: 1.1em; margin-bottom: 8px;">Sparse Autoencoder Features</p>
                <p style="color: #666;">No real SAE feature data available for visualization.</p>
                <p style="font-size: 0.9em; color: #999; margin-top: 8px;">
                    This visualization requires actual sparse autoencoder feature activations from model analysis.
                </p>
            </div>
            '''
        
        # Process real feature data (use actual SAE data from feature_data)
        # feature_data already contains real data from model_results
        if not feature_data:
            # This should not happen due to check above, but just in case
            return "<p>No sparse autoencoder feature data available.</p>"
        
        # Extract feature names from first entry
        try:
            first_entry = feature_data[0]
            if isinstance(first_entry.get("features"), dict):
                feature_names = list(first_entry["features"].get("clean", {}).keys())
                if not feature_names:
                    feature_names = list(first_entry["features"].get("attack", {}).keys())
            else:
                # Invalid structure - skip
                return "<p>Invalid sparse autoencoder feature data structure.</p>"
        except Exception:
            return "<p>Error processing sparse autoencoder feature data.</p>"
        
        # Build visualization
        tables_html = ""
        
        for data in feature_data:
            features = data["features"]
            clean = features.get("clean", {})
            attack = features.get("attack", {})
            
            all_features = sorted(set(list(clean.keys()) + list(attack.keys())))[:15]
            
            rows_html = ""
            for feat in all_features:
                clean_val = clean.get(feat, 0)
                attack_val = attack.get(feat, 0)
                diff = attack_val - clean_val
                
                # Color based on activation change
                if diff > 0.2:
                    diff_class = "high-increase"
                    diff_color = "#ef4444"
                elif diff > 0.05:
                    diff_class = "increase"
                    diff_color = "#f59e0b"
                elif diff < -0.1:
                    diff_class = "decrease"
                    diff_color = "#22c55e"
                else:
                    diff_class = "neutral"
                    diff_color = "#888"
                
                rows_html += f'''
                <tr>
                    <td style="text-align: left; padding: 8px; font-family: monospace; font-size: 0.8rem;">
                        {feat}</td>
                    <td style="padding: 8px;">
                        <div style="background: linear-gradient(90deg, #22c55e {clean_val*100:.0f}%, transparent {clean_val*100:.0f}%);
                                    height: 20px; border-radius: 4px; min-width: 100px;"></div>
                        <span style="font-size: 0.75rem; color: #888;">{clean_val:.3f}</span>
                    </td>
                    <td style="padding: 8px;">
                        <div style="background: linear-gradient(90deg, #ef4444 {attack_val*100:.0f}%, transparent {attack_val*100:.0f}%);
                                    height: 20px; border-radius: 4px; min-width: 100px;"></div>
                        <span style="font-size: 0.75rem; color: #888;">{attack_val:.3f}</span>
                    </td>
                    <td style="padding: 8px; color: {diff_color}; font-weight: 600;">
                        {diff:+.3f}
                    </td>
                </tr>
                '''
            
            tables_html += f'''
            <div class="sae-model-section" style="margin-bottom: 24px;">
                <h4 style="color: #e0e0e0; margin-bottom: 12px;">{data['model']}</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: var(--bg-secondary, #2a2a4e);">
                            <th style="padding: 10px; text-align: left;">Feature</th>
                            <th style="padding: 10px;">Clean Activation</th>
                            <th style="padding: 10px;">Attack Activation</th>
                            <th style="padding: 10px;">Δ Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            '''
        
        return f'''
        <div class="sae-visualization" style="background: var(--bg-tertiary, #1a1a2e);
                                              border-radius: 12px; padding: 20px; margin: 20px 0;">
            <style>
                .sae-visualization table {{
                    font-size: 0.85rem;
                }}
                .sae-visualization th, .sae-visualization td {{
                    border: 1px solid #333;
                }}
                .sae-visualization tbody tr:hover {{
                    background: rgba(129, 140, 248, 0.1);
                }}
            </style>
            {tables_html}
            <p style="margin-top: 16px; font-size: 0.85rem; color: #888;">
                Sparse Autoencoder features represent interpretable directions in the model's activation space.
                Significant activation changes (red values) indicate features that distinguish attack inputs.
            </p>
        </div>
        '''
    
    def generate_chart_with_error_bars(
        self,
        data: List[Dict[str, Any]],
        x_key: str = "label",
        y_key: str = "mean",
        error_key: str = "std",
        title: str = "Chart with Error Bars"
    ) -> str:
        """
        Generate a bar chart with error bars showing confidence intervals.
        
        Args:
            data: List of dicts with label, mean, and std values
            title: Chart title
        """
        if not data:
            return "<p>No data available for chart.</p>"
        
        chart_width = 600
        chart_height = 350
        margin = {"left": 80, "right": 20, "top": 50, "bottom": 80}
        
        plot_width = chart_width - margin["left"] - margin["right"]
        plot_height = chart_height - margin["top"] - margin["bottom"]
        
        max_val = max(d.get(y_key, 0) + d.get(error_key, 0) for d in data) * 1.15
        bar_width = (plot_width / len(data)) * 0.6
        colors = ["#818cf8", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"]
        
        bars_html = ""
        for i, d in enumerate(data):
            label = d.get(x_key, f"Item {i}")
            mean = d.get(y_key, 0)
            std = d.get(error_key, 0)
            color = colors[i % len(colors)]
            
            x = margin["left"] + (i + 0.5) * (plot_width / len(data))
            bar_height = (mean / max_val) * plot_height if max_val > 0 else 0
            y = margin["top"] + plot_height - bar_height
            
            # Bar
            bars_html += f'''
            <rect x="{x - bar_width/2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}"
                  fill="{color}" rx="4" opacity="0.85">
                <title>{label}: {mean:.3f} ± {std:.3f}</title>
            </rect>
            '''
            
            # Error bar (if std > 0)
            if std > 0:
                error_top = margin["top"] + plot_height - ((mean + std) / max_val) * plot_height
                error_bottom = margin["top"] + plot_height - ((mean - std) / max_val) * plot_height
                error_bottom = min(error_bottom, margin["top"] + plot_height)
                
                bars_html += f'''
                <line x1="{x:.1f}" y1="{error_top:.1f}" x2="{x:.1f}" y2="{error_bottom:.1f}"
                      stroke="#fff" stroke-width="2"/>
                <line x1="{x - 8:.1f}" y1="{error_top:.1f}" x2="{x + 8:.1f}" y2="{error_top:.1f}"
                      stroke="#fff" stroke-width="2"/>
                <line x1="{x - 8:.1f}" y1="{error_bottom:.1f}" x2="{x + 8:.1f}" y2="{error_bottom:.1f}"
                      stroke="#fff" stroke-width="2"/>
                '''
            
            # Label
            bars_html += f'''
            <text x="{x:.1f}" y="{chart_height - margin['bottom'] + 20}" fill="#e0e0e0"
                  font-size="11" text-anchor="middle" transform="rotate(-30, {x:.1f}, {chart_height - margin['bottom'] + 20})">
                {label[:15]}</text>
            '''
        
        # Y-axis scale
        y_axis_html = ""
        for i in range(5):
            val = max_val * i / 4
            y = margin["top"] + plot_height - (i / 4) * plot_height
            y_axis_html += f'''
            <line x1="{margin['left'] - 5}" y1="{y:.1f}" x2="{margin['left']}" y2="{y:.1f}" stroke="#888"/>
            <text x="{margin['left'] - 10}" y="{y + 4:.1f}" fill="#888" font-size="10" text-anchor="end">
                {val:.2f}</text>
            '''
        
        return f'''
        <div class="error-bar-chart" style="background: var(--bg-tertiary, #1a1a2e);
                                            border-radius: 12px; padding: 20px; margin: 20px 0;">
            <svg width="{chart_width}" height="{chart_height}" viewBox="0 0 {chart_width} {chart_height}"
                 style="display: block; margin: 0 auto;">
                <rect width="{chart_width}" height="{chart_height}" fill="transparent"/>
                
                <!-- Axes -->
                <line x1="{margin['left']}" y1="{margin['top']}" 
                      x2="{margin['left']}" y2="{chart_height - margin['bottom']}" stroke="#888"/>
                <line x1="{margin['left']}" y1="{chart_height - margin['bottom']}"
                      x2="{chart_width - margin['right']}" y2="{chart_height - margin['bottom']}" stroke="#888"/>
                
                <!-- Y-axis labels -->
                {y_axis_html}
                
                <!-- Bars with error bars -->
                {bars_html}
                
                <!-- Title -->
                <text x="{chart_width/2}" y="25" fill="#e0e0e0" font-size="14"
                      font-weight="600" text-anchor="middle">{title}</text>
            </svg>
            <p style="margin-top: 10px; font-size: 0.8rem; color: #888; text-align: center;">
                Error bars represent ±1 standard deviation (95% confidence interval approximation).
            </p>
        </div>
        '''

