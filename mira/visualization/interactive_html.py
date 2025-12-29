"""
Interactive HTML Visualization for Transformer Internals.

Generates rich, interactive HTML visualizations for:
- Attention patterns (heatmaps)
- Layer-wise processing flow
- Token probability distributions
- Attack flow comparison
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np


# HTML template for interactive visualization
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>MIRA - Transformer Visualization</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .header h1 { 
            font-size: 2.5em;
            background: linear-gradient(45deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .section {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
        }
        .section h2 {
            color: #00d2ff;
            margin-bottom: 20px;
            border-bottom: 2px solid #3a7bd5;
            padding-bottom: 10px;
        }
        .attention-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .attention-head {
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 15px;
        }
        .attention-head h3 { color: #a0a0a0; margin-bottom: 10px; }
        .heatmap {
            display: grid;
            gap: 2px;
        }
        .heatmap-cell {
            width: 100%;
            aspect-ratio: 1;
            border-radius: 3px;
            transition: transform 0.2s;
        }
        .heatmap-cell:hover {
            transform: scale(1.5);
            z-index: 10;
        }
        .layer-flow {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .layer-bar {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .layer-label {
            width: 80px;
            font-weight: bold;
        }
        .layer-progress {
            flex: 1;
            height: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        .layer-fill {
            height: 100%;
            border-radius: 15px;
            transition: width 0.3s ease;
        }
        .layer-fill.refusal { background: linear-gradient(90deg, #ff6b6b, #c92a2a); }
        .layer-fill.neutral { background: linear-gradient(90deg, #868e96, #495057); }
        .layer-fill.acceptance { background: linear-gradient(90deg, #51cf66, #2f9e44); }
        .token-prob {
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            border-radius: 20px;
            background: rgba(58, 123, 213, 0.3);
            border: 1px solid #3a7bd5;
        }
        .token-prob .prob { 
            font-size: 0.8em; 
            color: #00d2ff;
            margin-left: 5px;
        }
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        .comparison-col h3 {
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 15px;
        }
        .comparison-col.clean h3 { background: rgba(201, 42, 42, 0.3); }
        .comparison-col.attack h3 { background: rgba(47, 158, 68, 0.3); }
        .tooltip {
            position: absolute;
            background: rgba(0,0,0,0.9);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            z-index: 100;
        }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        .live { animation: pulse 1s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MIRA Transformer Visualization</h1>
            <p>{{subtitle}}</p>
        </div>
        {{content}}
    </div>
    <script>
        // Interactive tooltips
        document.querySelectorAll('.heatmap-cell').forEach(cell => {
            cell.addEventListener('mouseenter', (e) => {
                const value = e.target.dataset.value;
                const from = e.target.dataset.from;
                const to = e.target.dataset.to;
                // Show tooltip logic here
            });
        });
    </script>
</body>
</html>
'''


class InteractiveViz:
    """
    Generates interactive HTML visualizations.
    
    Creates rich, browser-based visualizations for transformer analysis.
    """
    
    def __init__(self, output_dir: str = "./viz"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for HTML output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_attention_heatmap(
        self,
        attention: np.ndarray,
        tokens: List[str],
        layer: int,
        head: int,
    ) -> str:
        """
        Generate HTML for attention heatmap.
        
        Args:
            attention: Attention weights [seq, seq]
            tokens: Token strings
            layer: Layer index
            head: Attention head index
            
        Returns:
            HTML string
        """
        n = len(tokens)
        cells = []
        
        for i in range(min(n, 15)):
            row = []
            for j in range(min(n, 15)):
                val = float(attention[i, j])
                # Color intensity based on value
                r = int(255 * val)
                g = int(100 * (1 - val))
                b = int(180 * (1 - val))
                color = f"rgb({r},{g},{b})"
                
                row.append(f'''<div class="heatmap-cell" 
                    style="background:{color}; grid-column:{j+1}; grid-row:{i+1};"
                    data-value="{val:.3f}"
                    data-from="{tokens[i]}"
                    data-to="{tokens[j]}"
                    title="{tokens[i]} → {tokens[j]}: {val:.3f}">
                </div>''')
            cells.extend(row)
        
        return f'''
        <div class="attention-head">
            <h3>Layer {layer}, Head {head}</h3>
            <div class="heatmap" style="grid-template-columns: repeat({min(n,15)}, 1fr);">
                {"".join(cells)}
            </div>
        </div>
        '''
    
    def generate_layer_flow(
        self,
        layer_states: List[Dict],
    ) -> str:
        """
        Generate HTML for layer-by-layer flow visualization.
        
        Args:
            layer_states: List of layer state dicts
            
        Returns:
            HTML string
        """
        bars = []
        
        for state in layer_states:
            layer = state.get("layer", 0)
            direction = state.get("direction", "neutral")
            refusal = state.get("refusal_score", 0)
            acceptance = state.get("acceptance_score", 0)
            
            # Determine bar width and color
            if direction == "refusal":
                width = min(100, abs(refusal) * 100)
                css_class = "refusal"
            elif direction == "acceptance":
                width = min(100, abs(acceptance) * 100)
                css_class = "acceptance"
            else:
                width = 30
                css_class = "neutral"
            
            icon = "❌" if direction == "refusal" else ("✅" if direction == "acceptance" else "○")
            
            bars.append(f'''
            <div class="layer-bar">
                <span class="layer-label">Layer {layer}</span>
                <div class="layer-progress">
                    <div class="layer-fill {css_class}" style="width: {width}%;"></div>
                </div>
                <span>{icon}</span>
            </div>
            ''')
        
        return f'''
        <div class="section">
            <h2>Processing Flow</h2>
            <div class="layer-flow">
                {"".join(bars)}
            </div>
        </div>
        '''
    
    def generate_token_probabilities(
        self,
        token_probs: List[Tuple[str, float]],
        title: str = "Top Predictions",
    ) -> str:
        """
        Generate HTML for token probability display.
        
        Args:
            token_probs: List of (token, probability) tuples
            title: Section title
            
        Returns:
            HTML string
        """
        tokens_html = []
        
        for token, prob in token_probs[:10]:
            tokens_html.append(f'''
            <span class="token-prob">
                {token}<span class="prob">{prob:.1%}</span>
            </span>
            ''')
        
        return f'''
        <div class="section">
            <h2>{title}</h2>
            <div>
                {"".join(tokens_html)}
            </div>
        </div>
        '''
    
    def generate_comparison(
        self,
        clean_states: List[Dict],
        attack_states: List[Dict],
    ) -> str:
        """
        Generate side-by-side comparison of clean vs attacked.
        
        Args:
            clean_states: Clean prompt layer states
            attack_states: Attacked prompt layer states
            
        Returns:
            HTML string
        """
        clean_bars = self._generate_mini_flow(clean_states)
        attack_bars = self._generate_mini_flow(attack_states)
        
        return f'''
        <div class="section">
            <h2>Attack Comparison</h2>
            <div class="comparison">
                <div class="comparison-col clean">
                    <h3>Original (Blocked)</h3>
                    {clean_bars}
                </div>
                <div class="comparison-col attack">
                    <h3>Attacked (Bypassed)</h3>
                    {attack_bars}
                </div>
            </div>
        </div>
        '''
    
    def _generate_mini_flow(self, states: List[Dict]) -> str:
        """Generate mini flow bars."""
        bars = []
        for state in states:
            direction = state.get("direction", "neutral")
            css_class = direction
            bars.append(f'''<div class="layer-progress" style="height:20px;margin:3px 0;">
                <div class="layer-fill {css_class}" style="width:100%;"></div>
            </div>''')
        return "".join(bars)
    
    def save_visualization(
        self,
        content: str,
        filename: str = "visualization.html",
        subtitle: str = "",
    ) -> str:
        """
        Save complete visualization to HTML file.
        
        Args:
            content: HTML content sections
            filename: Output filename
            subtitle: Page subtitle
            
        Returns:
            Path to saved file
        """
        html = HTML_TEMPLATE.replace("{{content}}", content)
        html = html.replace("{{subtitle}}", subtitle)
        
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        
        return str(path)
    
    def create_full_report(
        self,
        attention_data: Optional[List[Dict]] = None,
        layer_states: Optional[List[Dict]] = None,
        token_probs: Optional[List[Tuple[str, float]]] = None,
        comparison: Optional[Tuple[List[Dict], List[Dict]]] = None,
        prompt: str = "",
        model_name: str = "",
    ) -> str:
        """
        Create complete interactive report.
        
        Args:
            attention_data: Attention patterns per head
            layer_states: Layer-by-layer states
            token_probs: Token probabilities
            comparison: Clean vs attack states tuple
            prompt: Input prompt
            model_name: Model name
            
        Returns:
            Path to saved HTML file
        """
        sections = []
        
        # Prompt info
        sections.append(f'''
        <div class="section">
            <h2>Analysis Target</h2>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Prompt:</strong> {prompt}</p>
        </div>
        ''')
        
        # Layer flow
        if layer_states:
            sections.append(self.generate_layer_flow(layer_states))
        
        # Token probabilities
        if token_probs:
            sections.append(self.generate_token_probabilities(token_probs))
        
        # Attention heatmaps
        if attention_data:
            heads_html = []
            for data in attention_data[:4]:  # Show 4 heads
                heads_html.append(self.generate_attention_heatmap(
                    data["attention"],
                    data["tokens"],
                    data["layer"],
                    data["head"],
                ))
            
            sections.append(f'''
            <div class="section">
                <h2>Attention Patterns</h2>
                <div class="attention-grid">
                    {"".join(heads_html)}
                </div>
            </div>
            ''')
        
        # Comparison
        if comparison:
            sections.append(self.generate_comparison(comparison[0], comparison[1]))
        
        return self.save_visualization(
            "".join(sections),
            filename="mira_report.html",
            subtitle=f"Analysis of: {prompt[:50]}...",
        )


def generate_live_html(
    layer_idx: int,
    n_layers: int,
    direction: str,
    refusal_score: float,
    acceptance_score: float,
) -> str:
    """
    Generate HTML snippet for live updates.
    
    Args:
        layer_idx: Current layer
        n_layers: Total layers
        direction: Current direction
        refusal_score: Refusal score
        acceptance_score: Acceptance score
        
    Returns:
        HTML snippet for injection
    """
    progress = (layer_idx + 1) / n_layers * 100
    
    if direction == "refusal":
        color = "#ff6b6b"
        icon = "❌"
    elif direction == "acceptance":
        color = "#51cf66"
        icon = "✅"
    else:
        color = "#868e96"
        icon = "○"
    
    return f'''
    <div style="display:flex;align-items:center;gap:10px;margin:5px 0;">
        <span style="width:60px;font-weight:bold;">Layer {layer_idx}</span>
        <div style="flex:1;height:25px;background:#333;border-radius:12px;overflow:hidden;">
            <div style="width:{progress}%;height:100%;background:{color};transition:width 0.3s;"></div>
        </div>
        <span>{icon}</span>
    </div>
    '''
