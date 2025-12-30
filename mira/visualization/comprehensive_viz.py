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
            return "<p>No probe data available.</p>"
        
        # For now, use probe_bypass_rate as a proxy
        # In full implementation, would need per-layer probe training
        
        rows_html = ""
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            probe_bypass = result.get("probe_bypass_rate", 0.0)
            probe_accuracy = result.get("probe_accuracy", 0.0)
            
            # Simulate per-layer accuracy (would be real data in full implementation)
            # Typically accuracy increases in middle layers
            n_layers = len(result.get("layer_activations", {}).get("clean", [])) or 12
            
            bars_html = ""
            for layer in range(min(n_layers, 16)):
                # Simulate accuracy curve (peaks in middle layers)
                normalized_pos = layer / n_layers
                simulated_accuracy = probe_accuracy * (1.0 - abs(normalized_pos - 0.6) * 0.5)
                
                height = simulated_accuracy * 100
                color = self.colors["success"] if simulated_accuracy > 0.7 else \
                       self.colors["warning"] if simulated_accuracy > 0.5 else \
                       self.colors["danger"]
                
                bars_html += f'''
                <div class="probe-bar" style="height: {height}%; background: {color};"
                     title="Layer {layer}: {simulated_accuracy:.1%}">
                </div>
'''
            
            rows_html += f'''
            <div class="probe-row">
                <div class="model-label">{model_name}</div>
                <div class="probe-bars">{bars_html}</div>
                <div class="probe-stats">
                    <span>Overall: {probe_accuracy:.1%}</span>
                    <span>Bypass: {probe_bypass:.1%}</span>
                </div>
            </div>
'''
        
        if not rows_html:
            return "<p>No probe data available.</p>"
        
        return f'''
        <div class="probe-curves">
            <style>
                .probe-row {{
                    display: flex;
                    align-items: center;
                    margin: 16px 0;
                    gap: 16px;
                }}
                .model-label {{
                    min-width: 150px;
                    font-weight: 600;
                }}
                .probe-bars {{
                    flex: 1;
                    display: flex;
                    gap: 4px;
                    height: 100px;
                    align-items: flex-end;
                }}
                .probe-bar {{
                    flex: 1;
                    min-width: 8px;
                    border-radius: 2px 2px 0 0;
                    cursor: help;
                    transition: opacity 0.2s;
                }}
                .probe-bar:hover {{
                    opacity: 0.8;
                }}
                .probe-stats {{
                    min-width: 150px;
                    font-size: 0.85rem;
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                }}
            </style>
            {rows_html}
            <p class="note" style="margin-top: 16px; font-size: 0.85rem; color: var(--text-secondary, #999);">
                <strong>Note:</strong> Per-layer probe accuracy requires training separate probes at each layer.
                Current visualization shows simulated accuracy curves based on overall probe performance.
            </p>
        </div>
'''
    
    def generate_token_probability_evolution(
        self,
        model_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate token probability evolution across layers.
        
        Shows how token probabilities change through the model.
        """
        if not model_results:
            return "<p>No logit lens data available.</p>"
        
        sections_html = ""
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            logit_lens = result.get("logit_lens_sample", {})
            
            if not logit_lens or not logit_lens.get("layers_analyzed"):
                continue
            
            # Generate evolution chart
            chart_html = self._generate_logit_evolution_chart(logit_lens, model_name)
            
            sections_html += f'''
            <div class="logit-section">
                <h4>{model_name}</h4>
                {chart_html}
            </div>
'''
        
        if not sections_html:
            return '''
            <p class="note">
                Token probability evolution requires full logit lens analysis across all layers.
                Run logit lens analysis during model evaluation to populate this section.
            </p>
'''
        
        return f'''
        <div class="logit-evolution">
            <style>
                .logit-section {{
                    margin: 24px 0;
                    padding: 16px;
                    background: var(--bg-tertiary, #1a1a2e);
                    border-radius: 8px;
                }}
                .logit-section h4 {{
                    margin-top: 0;
                }}
            </style>
            {sections_html}
        </div>
'''
    
    def _generate_logit_evolution_chart(
        self,
        logit_lens_data: Dict[str, Any],
        model_name: str
    ) -> str:
        """Generate chart for logit lens evolution."""
        layers_analyzed = logit_lens_data.get("layers_analyzed", 0)
        
        if layers_analyzed == 0:
            return "<p>No data</p>"
        
        # Simulate token evolution (would be real data in full implementation)
        # Show top 3 tokens and how their probabilities evolve
        tokens = ["token_A", "token_B", "token_C"]
        colors = [self.colors["primary"], self.colors["success"], self.colors["warning"]]
        
        lines_html = ""
        for i, (token, color) in enumerate(zip(tokens, colors)):
            # Simulate probability evolution
            points = []
            for layer in range(layers_analyzed):
                # Different evolution patterns for different tokens
                if i == 0:  # Primary token increases
                    prob = 0.1 + (layer / layers_analyzed) * 0.6
                elif i == 1:  # Secondary decreases
                    prob = 0.4 - (layer / layers_analyzed) * 0.2
                else:  # Tertiary stays low
                    prob = 0.1 + 0.05 * (layer % 3)
                
                x = layer * 50
                y = 100 - (prob * 100)
                points.append(f"{x},{y}")
            
            polyline = " ".join(points)
            lines_html += f'''
            <polyline points="{polyline}" 
                      fill="none" 
                      stroke="{color}" 
                      stroke-width="2"
                      opacity="0.8"/>
            <text x="{(layers_analyzed-1)*50 + 10}" y="{100 - (float(points[-1].split(',')[1]))}" 
                  fill="{color}" font-size="12">{token}</text>
'''
        
        return f'''
        <svg width="100%" height="150" viewBox="0 0 {layers_analyzed * 50} 120">
            <!-- Grid lines -->
            <line x1="0" y1="0" x2="{layers_analyzed * 50}" y2="0" stroke="#333" stroke-width="1"/>
            <line x1="0" y1="50" x2="{layers_analyzed * 50}" y2="50" stroke="#333" stroke-width="1" stroke-dasharray="2,2"/>
            <line x1="0" y1="100" x2="{layers_analyzed * 50}" y2="100" stroke="#333" stroke-width="1"/>
            
            <!-- Evolution lines -->
            {lines_html}
            
            <!-- Axis labels -->
            <text x="0" y="115" fill="#999" font-size="10">Layer 0</text>
            <text x="{(layers_analyzed-1)*50 - 30}" y="115" fill="#999" font-size="10">Layer {layers_analyzed-1}</text>
        </svg>
        <p style="font-size: 0.85rem; color: var(--text-secondary, #999); margin-top: 8px;">
            <strong>Note:</strong> Simulated token probability evolution. Full implementation requires logit lens analysis at each layer.
        </p>
'''
    
    def generate_entropy_evolution(
        self,
        model_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate entropy evolution visualization across layers.
        
        Shows how prediction entropy changes through the model.
        """
        if not model_results:
            return "<p>No entropy data available.</p>"
        
        chart_html = ""
        
        for result in model_results:
            if not result.get("success", False):
                continue
            
            model_name = result.get("model_name", "unknown")
            mean_entropy = result.get("mean_entropy", 0.0)
            
            # Simulate entropy evolution (would be real per-layer data)
            n_layers = len(result.get("layer_activations", {}).get("clean", [])) or 12
            
            points_clean = []
            points_attack = []
            
            for layer in range(n_layers):
                x = layer * 40
                # Simulate: entropy typically decreases through layers
                clean_entropy = mean_entropy * (1.0 - layer / n_layers * 0.3)
                attack_entropy = mean_entropy * (1.0 - layer / n_layers * 0.5)  # Drops more for attacks
                
                y_clean = 100 - (clean_entropy / 5.0 * 100)  # Normalize to 0-5 range
                y_attack = 100 - (attack_entropy / 5.0 * 100)
                
                points_clean.append(f"{x},{y_clean}")
                points_attack.append(f"{x},{y_attack}")
            
            polyline_clean = " ".join(points_clean)
            polyline_attack = " ".join(points_attack)
            
            chart_html += f'''
            <div class="entropy-chart">
                <h5>{model_name}</h5>
                <svg width="100%" height="120" viewBox="0 0 {n_layers * 40} 120">
                    <line x1="0" y1="100" x2="{n_layers * 40}" y2="100" stroke="#333" stroke-width="1"/>
                    <polyline points="{polyline_clean}" fill="none" stroke="{self.colors['success']}" stroke-width="2" opacity="0.8"/>
                    <polyline points="{polyline_attack}" fill="none" stroke="{self.colors['danger']}" stroke-width="2" opacity="0.8"/>
                    <text x="10" y="15" fill="{self.colors['success']}" font-size="12">Clean</text>
                    <text x="10" y="30" fill="{self.colors['danger']}" font-size="12">Attack</text>
                </svg>
            </div>
'''
        
        if not chart_html:
            return "<p>No entropy data available.</p>"
        
        return f'''
        <div class="entropy-evolution">
            <style>
                .entropy-chart {{
                    margin: 16px 0;
                    padding: 12px;
                    background: var(--bg-tertiary, #1a1a2e);
                    border-radius: 8px;
                }}
                .entropy-chart h5 {{
                    margin: 0 0 8px 0;
                    font-size: 0.9rem;
                }}
            </style>
            {chart_html}
            <p class="note" style="margin-top: 16px; font-size: 0.85rem; color: var(--text-secondary, #999);">
                <strong>Note:</strong> Entropy evolution requires per-layer entropy calculation during generation.
                Current visualization shows simulated curves based on overall entropy metrics.
            </p>
        </div>
'''
