"""
Real-time Web Visualization Server.

Starts a local web server that displays live LLM processing during attacks:
- Attention heatmaps
- Layer-by-layer activation flow
- Token probabilities
- Attack progress

Usage:
    python -m mira.visualization.live_server
    # Opens browser at http://localhost:5000
"""

import json
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from queue import Queue
import time

try:
    from flask import Flask, render_template_string, jsonify, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Install flask: pip install flask flask-cors")


# Global event queue for streaming
event_queue: Queue = Queue()


@dataclass
class VisualizationEvent:
    """Event for visualization update."""
    event_type: str  # "layer", "attention", "token", "attack_step", "complete"
    data: Dict[str, Any]
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class LiveVisualizationServer:
    """
    Real-time visualization server for LLM attack monitoring.
    
    Provides:
    - WebSocket-style SSE (Server-Sent Events) for real-time updates
    - Interactive HTML dashboard with D3.js visualizations
    - Attention heatmaps, layer flow, token probabilities
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        self.app = None
        self.server_thread = None
        self.running = False
        
        if FLASK_AVAILABLE:
            self._setup_flask()
    
    def _setup_flask(self):
        """Setup Flask application with routes."""
        self.app = Flask(__name__)
        CORS(self.app)
        
        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route("/api/events")
        def events():
            """SSE endpoint for real-time events."""
            def generate():
                while True:
                    try:
                        event = event_queue.get(timeout=1.0)
                        yield f"data: {event.to_json()}\n\n"
                    except:
                        yield f"data: {json.dumps({'event_type': 'ping'})}\n\n"
            return Response(generate(), mimetype="text/event-stream")
        
        @self.app.route("/api/status")
        def status():
            return jsonify({"status": "running", "events_pending": event_queue.qsize()})
    
    def start(self, open_browser: bool = True):
        """Start the visualization server."""
        if not FLASK_AVAILABLE:
            print("Flask not available. Install: pip install flask flask-cors")
            return
        
        self.running = True
        
        def run_server():
            self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        print(f"\n  🌐 Live Visualization: http://{self.host}:{self.port}")
        
        if open_browser:
            time.sleep(0.5)
            webbrowser.open(f"http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the server."""
        self.running = False
    
    @staticmethod
    def send_event(event: VisualizationEvent):
        """Send event to connected clients."""
        event_queue.put(event)
    
    @staticmethod
    def send_layer_update(
        layer_idx: int,
        refusal_score: float,
        acceptance_score: float,
        direction: str,
        top_tokens: List[tuple] = None,
    ):
        """Send layer processing update."""
        event = VisualizationEvent(
            event_type="layer",
            data={
                "layer": layer_idx,
                "refusal_score": refusal_score,
                "acceptance_score": acceptance_score,
                "direction": direction,
                "top_tokens": top_tokens or [],
            }
        )
        event_queue.put(event)
    
    @staticmethod
    def send_attention_update(
        layer_idx: int,
        head_idx: int,
        attention_weights: List[List[float]],
        tokens: List[str],
    ):
        """Send attention heatmap update."""
        event = VisualizationEvent(
            event_type="attention",
            data={
                "layer": layer_idx,
                "head": head_idx,
                "weights": attention_weights,
                "tokens": tokens,
            }
        )
        event_queue.put(event)
    
    @staticmethod
    def send_attack_step(
        step: int,
        loss: float,
        suffix: str,
        success: bool = False,
    ):
        """Send attack optimization step."""
        event = VisualizationEvent(
            event_type="attack_step",
            data={
                "step": step,
                "loss": loss,
                "suffix": suffix,
                "success": success,
            }
        )
        event_queue.put(event)
    
    @staticmethod
    def send_transformer_trace(
        trace_data: Dict[str, Any],
        trace_type: str = "normal"  # "normal" or "adversarial"
    ):
        """Send complete transformer trace for comparison."""
        event = VisualizationEvent(
            event_type="transformer_trace",
            data={
                "trace_type": trace_type,
                "trace": trace_data,
            }
        )
        event_queue.put(event)
    
    @staticmethod
    def send_embeddings(
        tokens: List[str],
        embeddings: List[List[float]],  # [seq_len, hidden_dim]
    ):
        """Send token embeddings."""
        event = VisualizationEvent(
            event_type="embeddings",
            data={
                "tokens": tokens,
                "embeddings": embeddings,
            }
        )
        event_queue.put(event)
    
    @staticmethod
    def send_qkv_vectors(
        layer_idx: int,
        tokens: List[str],
        query: List[List[float]],  # [seq_len, head_dim]
        key: List[List[float]],
        value: List[List[float]],
    ):
        """Send Q/K/V vectors for a layer."""
        event = VisualizationEvent(
            event_type="qkv",
            data={
                "layer": layer_idx,
                "tokens": tokens,
                "query": query,
                "key": key,
                "value": value,
            }
        )
        event_queue.put(event)
    
    @staticmethod
    def send_mlp_activations(
        layer_idx: int,
        activations: List[List[float]],  # [seq_len, intermediate_dim]
        top_neurons: List[int],  # Indices of most active neurons
    ):
        """Send MLP activation data."""
        event = VisualizationEvent(
            event_type="mlp",
            data={
                "layer": layer_idx,
                "activations": activations,
                "top_neurons": top_neurons,
            }
        )
        event_queue.put(event)
    
    @staticmethod
    def send_residual_update(
        layer_idx: int,
        residual_norm: float,
        delta_norm: float,  # Change from previous layer
    ):
        """Send residual stream update."""
        event = VisualizationEvent(
            event_type="residual",
            data={
                "layer": layer_idx,
                "residual_norm": residual_norm,
                "delta_norm": delta_norm,
            }
        )
        event_queue.put(event)
    
    @staticmethod
    def send_complete(summary: Dict[str, Any]):
        """Send completion event."""
        event = VisualizationEvent(
            event_type="complete",
            data=summary
        )
        event_queue.put(event)


# HTML Dashboard - MIRA Cybersecurity Neural Attack Monitor
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA Neural Attack Monitor</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --primary: #0ff;
            --secondary: #f0f;
            --warning: #ff0;
            --danger: #f44;
            --success: #0f0;
            --bg-dark: #0a0a0f;
            --bg-panel: rgba(15, 25, 35, 0.9);
            --border: rgba(0, 255, 255, 0.2);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Space Mono', monospace;
            background: var(--bg-dark);
            color: #ddd;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated background grid */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                linear-gradient(90deg, rgba(0,255,255,0.03) 1px, transparent 1px),
                linear-gradient(rgba(0,255,255,0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: gridMove 20s linear infinite;
            pointer-events: none;
            z-index: -1;
        }
        
        @keyframes gridMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }
        
        /* Header */
        .header {
            background: linear-gradient(180deg, var(--bg-panel) 0%, transparent 100%);
            padding: 20px 30px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo-icon {
            width: 50px;
            height: 50px;
            background: conic-gradient(from 0deg, var(--primary), var(--secondary), var(--primary));
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: rotateGlow 3s linear infinite;
        }
        
        @keyframes rotateGlow {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }
        
        .logo-icon::after {
            content: '🧠';
            font-size: 24px;
        }
        
        .logo h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8em;
            color: var(--primary);
            text-shadow: 0 0 20px var(--primary);
            letter-spacing: 3px;
        }
        
        .header-stats {
            display: flex;
            gap: 30px;
        }
        
        .header-stat {
            text-align: center;
        }
        
        .header-stat-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5em;
            color: var(--primary);
        }
        
        .header-stat-label {
            font-size: 0.7em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        /* Main Grid */
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            grid-template-rows: auto auto auto;
            gap: 15px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .panel {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }
        
        .panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--primary), transparent);
        }
        
        .panel-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9em;
            color: var(--primary);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .panel-title::before {
            content: '◆';
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        /* Neural Layer Visualization */
        .neural-layers { grid-column: 1; grid-row: 1 / 3; }
        
        .layer-node {
            display: flex;
            align-items: center;
            margin: 12px 0;
            gap: 15px;
        }
        
        .node-circle {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 2px solid var(--primary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            transition: all 0.3s;
            position: relative;
        }
        
        .node-circle.active {
            background: rgba(0, 255, 255, 0.2);
            box-shadow: 0 0 20px var(--primary);
        }
        
        .node-circle.refusal {
            border-color: var(--danger);
            box-shadow: 0 0 15px var(--danger);
        }
        
        .node-circle.acceptance {
            border-color: var(--success);
            box-shadow: 0 0 15px var(--success);
        }
        
        .node-bar {
            flex: 1;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .node-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .node-bar-fill.refusal { background: linear-gradient(90deg, var(--danger), #ff8866); }
        .node-bar-fill.acceptance { background: linear-gradient(90deg, var(--success), #88ff88); }
        .node-bar-fill.neutral { background: linear-gradient(90deg, var(--warning), #ffcc00); }
        
        .node-value {
            width: 50px;
            text-align: right;
            font-size: 0.8em;
            color: #888;
        }
        
        /* Attack Console */
        .attack-console { grid-column: 2; grid-row: 1; }
        
        .console-output {
            background: #000;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 15px;
            font-size: 0.85em;
            height: 200px;
            overflow-y: auto;
        }
        
        .console-line {
            margin: 3px 0;
            opacity: 0.9;
        }
        
        .console-line.success { color: var(--success); }
        .console-line.error { color: var(--danger); }
        .console-line.info { color: var(--primary); }
        
        /* Loss Chart */
        .loss-chart { grid-column: 3; grid-row: 1; }
        
        #chart-svg {
            width: 100%;
            height: 200px;
        }
        
        /* Metrics */
        .metrics { grid-column: 2 / 4; grid-row: 2; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }
        
        .metric-card {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        
        .metric-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 2em;
            color: var(--primary);
            text-shadow: 0 0 10px var(--primary);
        }
        
        .metric-label {
            font-size: 0.7em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 5px;
        }
        
        /* Suffix Display */
        .suffix-panel { grid-column: 1 / 4; grid-row: 3; }
        
        .suffix-display {
            background: #000;
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 20px;
            font-size: 1.1em;
            color: var(--warning);
            word-break: break-all;
            min-height: 60px;
        }
        
        /* Status Bar */
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--bg-panel);
            border-top: 1px solid var(--border);
            padding: 10px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .pulse {
            width: 12px;
            height: 12px;
            background: var(--success);
            border-radius: 50%;
            animation: pulse 1.5s ease infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
        }
        
        .footer-text {
            font-size: 0.8em;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <div class="logo-icon"></div>
            <h1>MIRA NEURAL MONITOR</h1>
        </div>
        <div class="header-stats">
            <div class="header-stat">
                <div class="header-stat-value" id="events-count">0</div>
                <div class="header-stat-label">Events</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-value" id="step-count">0</div>
                <div class="header-stat-label">Steps</div>
            </div>
            <div class="header-stat">
                <div class="header-stat-value" id="best-loss-header">--</div>
                <div class="header-stat-label">Best Loss</div>
            </div>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="panel neural-layers">
            <div class="panel-title">Neural Layer Flow</div>
            <div id="layer-container"></div>
        </div>
        
        <div class="panel attack-console">
            <div class="panel-title">Attack Console</div>
            <div class="console-output" id="console"></div>
        </div>
        
        <div class="panel loss-chart">
            <div class="panel-title">Loss Trajectory</div>
            <svg id="chart-svg"></svg>
        </div>
        
        <div class="panel metrics">
            <div class="panel-title">Attack Metrics</div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="current-step">0</div>
                    <div class="metric-label">Current Step</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="current-loss">--</div>
                    <div class="metric-label">Current Loss</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="best-loss">--</div>
                    <div class="metric-label">Best Loss</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="asr">0%</div>
                    <div class="metric-label">Attack Success</div>
                </div>
            </div>
        </div>
        
        <div class="panel suffix-panel">
            <div class="panel-title">Adversarial Suffix</div>
            <div class="suffix-display" id="suffix">[ Awaiting attack initialization... ]</div>
        </div>
    </div>
    
    <div class="status-bar">
        <div class="status-indicator">
            <div class="pulse"></div>
            <span id="status">Connecting to attack pipeline...</span>
        </div>
        <div class="footer-text">MIRA Framework v1.0 | Neural Attack Monitor</div>
    </div>
    
    <script>
        let eventCount = 0;
        let lossHistory = [];
        let bestLoss = Infinity;
        
        const evtSource = new EventSource('/api/events');
        
        function log(msg, type = 'info') {
            const console = document.getElementById('console');
            const line = document.createElement('div');
            line.className = 'console-line ' + type;
            line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
            console.appendChild(line);
            console.scrollTop = console.scrollHeight;
        }
        
        log('Neural monitor initialized', 'success');
        
        evtSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            if (data.event_type === 'ping') return;
            
            eventCount++;
            document.getElementById('events-count').textContent = eventCount;
            
            switch(data.event_type) {
                case 'layer':
                    updateLayer(data.data);
                    break;
                case 'attack_step':
                    updateAttack(data.data);
                    break;
                case 'complete':
                    showComplete(data.data);
                    break;
            }
        };
        
        function updateLayer(data) {
            const container = document.getElementById('layer-container');
            let node = document.getElementById('node-' + data.layer);
            
            if (!node) {
                node = document.createElement('div');
                node.id = 'node-' + data.layer;
                node.className = 'layer-node';
                node.innerHTML = `
                    <div class="node-circle">L${data.layer}</div>
                    <div class="node-bar"><div class="node-bar-fill"></div></div>
                    <div class="node-value">0%</div>
                `;
                container.appendChild(node);
            }
            
            const score = Math.max(data.refusal_score, data.acceptance_score);
            const pct = Math.min(score * 100, 100);
            const circle = node.querySelector('.node-circle');
            const fill = node.querySelector('.node-bar-fill');
            const value = node.querySelector('.node-value');
            
            circle.className = 'node-circle active ' + data.direction;
            fill.className = 'node-bar-fill ' + data.direction;
            fill.style.width = pct + '%';
            value.textContent = pct.toFixed(0) + '%';
        }
        
        function updateAttack(data) {
            document.getElementById('current-step').textContent = data.step;
            document.getElementById('step-count').textContent = data.step;
            document.getElementById('current-loss').textContent = data.loss.toFixed(4);
            document.getElementById('suffix').textContent = data.suffix || '...';
            
            lossHistory.push(data.loss);
            if (data.loss < bestLoss) {
                bestLoss = data.loss;
                log('New best loss: ' + data.loss.toFixed(4), 'success');
            }
            document.getElementById('best-loss').textContent = bestLoss.toFixed(4);
            document.getElementById('best-loss-header').textContent = bestLoss.toFixed(4);
            document.getElementById('status').textContent = 'Processing step ' + data.step + '...';
            
            drawChart();
        }
        
        function drawChart() {
            const svg = d3.select('#chart-svg');
            svg.selectAll('*').remove();
            
            const rect = svg.node().getBoundingClientRect();
            const w = rect.width, h = rect.height;
            const m = {t: 20, r: 20, b: 30, l: 50};
            
            const x = d3.scaleLinear().domain([0, Math.max(lossHistory.length - 1, 1)]).range([m.l, w - m.r]);
            const y = d3.scaleLinear().domain([0, d3.max(lossHistory) || 1]).nice().range([h - m.b, m.t]);
            
            // Grid
            svg.append('g').attr('transform', `translate(${m.l},0)`)
                .call(d3.axisLeft(y).ticks(4).tickSize(-w + m.l + m.r))
                .attr('color', '#222');
            
            // Area
            const area = d3.area().x((d,i) => x(i)).y0(h - m.b).y1(d => y(d));
            svg.append('path').datum(lossHistory)
                .attr('fill', 'url(#gradient)')
                .attr('d', area);
            
            // Gradient
            const defs = svg.append('defs');
            const grad = defs.append('linearGradient').attr('id', 'gradient').attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%');
            grad.append('stop').attr('offset', '0%').attr('stop-color', '#0ff').attr('stop-opacity', 0.5);
            grad.append('stop').attr('offset', '100%').attr('stop-color', '#0ff').attr('stop-opacity', 0);
            
            // Line
            const line = d3.line().x((d,i) => x(i)).y(d => y(d));
            svg.append('path').datum(lossHistory)
                .attr('fill', 'none')
                .attr('stroke', '#0ff')
                .attr('stroke-width', 2)
                .attr('d', line);
            
            // Axes
            svg.append('g').attr('transform', `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(5)).attr('color', '#555');
            svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4)).attr('color', '#555');
        }
        
        function showComplete(data) {
            document.getElementById('status').textContent = 'Attack complete!';
            document.getElementById('asr').textContent = ((data.asr || 0) * 100).toFixed(0) + '%';
            log('Attack pipeline complete | ASR: ' + ((data.asr || 0) * 100).toFixed(1) + '%', 'success');
            document.querySelector('.pulse').style.background = '#0f0';
        }
        
        drawChart();
    </script>
</body>
</html>
'''


def run_server(port: int = 5000, open_browser: bool = True):
    """Start the visualization server."""
    server = LiveVisualizationServer(port=port)
    server.start(open_browser=open_browser)
    return server


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    
    server = run_server(port=args.port, open_browser=not args.no_browser)
    
    print("\n  Press Ctrl+C to stop the server\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Server stopped.")
