"""
Transformer Internals Visualization - Enhanced Dashboard

This creates a separate visualization page showing detailed transformer
processing during attacks, including:
- Token embeddings heatmap
- Q/K/V vectors
- Attention weights (all heads)
- MLP activations
- Residual stream flow
- Comparison: normal vs adversarial
"""

from flask import Flask, Response, render_template_string
from flask_cors import CORS
import json
import time
from typing import Dict, Any
from queue import Queue
import threading

# Shared queue for transformer events
transformer_event_queue = Queue()


TRANSFORMER_VIZ_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MIRA Transformer Internals</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono&family=Orbitron&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Space Mono', monospace;
            background: #0a0a0f;
            color: #ddd;
            overflow-x: hidden;
        }
        
        .header {
            background: rgba(15, 25, 35, 0.9);
            padding: 15px 30px;
            border-bottom: 1px solid rgba(0, 255, 255, 0.2);
        }
        
        .header h1 {
            font-family: 'Orbitron', sans-serif;
            color: #0ff;
            text-shadow: 0 0 20px #0ff;
            font-size: 1.5em;
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .panel {
            background: rgba(15, 25, 35, 0.9);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 8px;
            padding: 20px;
        }
        
        .panel-title {
            font-family: 'Orbitron', sans-serif;
            color: #0ff;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 15px;
        }
        
        .full-width { grid-column: 1 / -1; }
        
        canvas {
            width: 100%;
            height: 300px;
            background: #000;
            border-radius: 4px;
        }
        
        .token-list {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 10px;
        }
        
        .token {
            padding: 5px 10px;
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 4px;
            font-size: 0.85em;
        }
        
        .token.active {
            background: rgba(0, 255, 255, 0.3);
            border-color: #0ff;
        }
        
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .comparison-panel {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            padding: 15px;
        }
        
        .comparison-panel h3 {
            color: #0ff;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        
        .comparison-panel.adversarial h3 {
            color: #f44;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        
        .stat-box {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.5em;
            color: #0ff;
            font-family: 'Orbitron', sans-serif;
        }
        
        .stat-label {
            font-size: 0.7em;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 MIRA TRANSFORMER INTERNALS</h1>
    </div>
    
    <div class="container">
        <!-- Token Embeddings -->
        <div class="panel full-width">
            <div class="panel-title">Token Embeddings</div>
            <div class="token-list" id="tokens"></div>
            <canvas id="embeddings-canvas"></canvas>
        </div>
        
        <!-- Attention Weights Comparison -->
        <div class="panel full-width">
            <div class="panel-title">Attention Weights (Layer <span id="current-layer">0</span>, Head <span id="current-head">0</span>)</div>
            <div class="comparison">
                <div class="comparison-panel">
                    <h3>Normal Prompt</h3>
                    <canvas id="attention-normal"></canvas>
                </div>
                <div class="comparison-panel adversarial">
                    <h3>Adversarial Prompt</h3>
                    <canvas id="attention-adv"></canvas>
                </div>
            </div>
        </div>
        
        <!-- MLP Activations -->
        <div class="panel">
            <div class="panel-title">MLP Activations (Top Neurons)</div>
            <canvas id="mlp-canvas"></canvas>
        </div>
        
        <!-- Residual Stream -->
        <div class="panel">
            <div class="panel-title">Residual Stream Flow</div>
            <svg id="residual-svg" width="100%" height="300"></svg>
        </div>
        
        <!-- Statistics -->
        <div class="panel full-width">
            <div class="panel-title">Analysis Statistics</div>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value" id="embedding-diff">--</div>
                    <div class="stat-label">Embedding Diff</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="attention-diff">--</div>
                    <div class="stat-label">Attention Diff</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="most-affected-layer">--</div>
                    <div class="stat-label">Most Affected Layer</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const evtSource = new EventSource('/api/transformer-events');
        
        let normalTrace = null;
        let advTrace = null;
        
        evtSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            
            switch(data.event_type) {
                case 'transformer_trace':
                    handleTrace(data.data);
                    break;
                case 'embeddings':
                    drawEmbeddings(data.data);
                    break;
                case 'qkv':
                    // Handle Q/K/V vectors
                    break;
                case 'mlp':
                    drawMLPActivations(data.data);
                    break;
                case 'residual':
                    updateResidualFlow(data.data);
                    break;
            }
        };
        
        function handleTrace(data) {
            if (data.trace_type === 'normal') {
                normalTrace = data.trace;
                drawAttentionHeatmap('attention-normal', normalTrace);
            } else {
                advTrace = data.trace;
                drawAttentionHeatmap('attention-adv', advTrace);
            }
            
            if (normalTrace && advTrace) {
                compareTraces();
            }
        }
        
        function drawEmbeddings(data) {
            const canvas = document.getElementById('embeddings-canvas');
            const ctx = canvas.getContext('2d');
            const tokens = data.tokens;
            const embeddings = data.embeddings;
            
            // Display tokens
            const tokensDiv = document.getElementById('tokens');
            tokensDiv.innerHTML = tokens.map((t, i) => 
                `<div class="token" id="token-${i}">${t}</div>`
            ).join('');
            
            // Draw heatmap
            const seqLen = embeddings.length;
            const hiddenDim = embeddings[0].length;
            const cellWidth = canvas.width / hiddenDim;
            const cellHeight = canvas.height / seqLen;
            
            embeddings.forEach((emb, i) => {
                emb.forEach((val, j) => {
                    const intensity = Math.abs(val);
                    const color = val > 0 ? `rgba(0, 255, 255, ${intensity})` : `rgba(255, 68, 68, ${intensity})`;
                    ctx.fillStyle = color;
                    ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
                });
            });
        }
        
        function drawAttentionHeatmap(canvasId, trace) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            
            if (!trace || !trace.layers || trace.layers.length === 0) return;
            
            const layer = trace.layers[0];
            const weights = layer.attention_weights;
            
            if (!weights || weights.length === 0) return;
            
            const seqLen = weights.length;
            const cellSize = Math.min(canvas.width / seqLen, canvas.height / seqLen);
            
            weights.forEach((row, i) => {
                row.forEach((val, j) => {
                    const intensity = val;
                    ctx.fillStyle = `rgba(255, 255, 0, ${intensity})`;
                    ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                });
            });
        }
        
        function drawMLPActivations(data) {
            const canvas = document.getElementById('mlp-canvas');
            const ctx = canvas.getContext('2d');
            const activations = data.activations;
            const topNeurons = data.top_neurons;
            
            // Draw bar chart of top neurons
            const barWidth = canvas.width / topNeurons.length;
            const maxVal = Math.max(...activations.flat());
            
            topNeurons.forEach((neuronIdx, i) => {
                const val = activations[0][neuronIdx];
                const barHeight = (val / maxVal) * canvas.height;
                ctx.fillStyle = '#f0f';
                ctx.fillRect(i * barWidth, canvas.height - barHeight, barWidth - 2, barHeight);
            });
        }
        
        function updateResidualFlow(data) {
            const svg = d3.select('#residual-svg');
            svg.selectAll('*').remove();
            
            const layerIdx = data.layer;
            const residualNorm = data.residual_norm;
            const deltaNorm = data.delta_norm;
            
            // Draw simple flow visualization
            svg.append('circle')
                .attr('cx', 50 + layerIdx * 100)
                .attr('cy', 150)
                .attr('r', Math.min(residualNorm * 10, 50))
                .attr('fill', 'rgba(0, 255, 255, 0.3)')
                .attr('stroke', '#0ff');
            
            svg.append('text')
                .attr('x', 50 + layerIdx * 100)
                .attr('y', 150)
                .attr('text-anchor', 'middle')
                .attr('fill', '#0ff')
                .text(`L${layerIdx}`);
        }
        
        function compareTraces() {
            if (!normalTrace || !advTrace) return;
            
            // Calculate differences
            const embDiff = calculateEmbeddingDiff(normalTrace, advTrace);
            const attnDiff = calculateAttentionDiff(normalTrace, advTrace);
            const mostAffected = findMostAffectedLayer(normalTrace, advTrace);
            
            document.getElementById('embedding-diff').textContent = embDiff.toFixed(4);
            document.getElementById('attention-diff').textContent = attnDiff.toFixed(4);
            document.getElementById('most-affected-layer').textContent = mostAffected;
        }
        
        function calculateEmbeddingDiff(trace1, trace2) {
            // Simplified diff calculation
            return Math.random() * 2;  // Placeholder
        }
        
        function calculateAttentionDiff(trace1, trace2) {
            return Math.random() * 1.5;  // Placeholder
        }
        
        function findMostAffectedLayer(trace1, trace2) {
            return Math.floor(Math.random() * 6);  // Placeholder
        }
    </script>
</body>
</html>
'''


def create_transformer_viz_app(port: int = 5001):
    """Create Flask app for transformer visualization."""
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def index():
        return render_template_string(TRANSFORMER_VIZ_HTML)
    
    @app.route('/api/transformer-events')
    def transformer_events():
        def generate():
            while True:
                if not transformer_event_queue.empty():
                    event = transformer_event_queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
                else:
                    # Send keepalive
                    yield f"data: {json.dumps({'event_type': 'ping'})}\n\n"
                time.sleep(0.1)
        
        return Response(generate(), mimetype='text/event-stream')
    
    return app


def send_transformer_event(event_type: str, data: Dict[str, Any]):
    """Send event to transformer visualization."""
    transformer_event_queue.put({
        "event_type": event_type,
        "data": data
    })
