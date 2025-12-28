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
    def send_complete(summary: Dict[str, Any]):
        """Send completion event."""
        event = VisualizationEvent(
            event_type="complete",
            data=summary
        )
        event_queue.put(event)


# HTML Dashboard with D3.js visualization
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA - Live Attack Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 {
            font-size: 2em;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .subtitle { color: #888; margin-top: 5px; }
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .panel h2 {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .panel h2::before {
            content: '';
            width: 4px;
            height: 20px;
            background: #00d4ff;
            border-radius: 2px;
        }
        
        /* Layer Flow */
        .layer-flow { height: 300px; }
        .layer-bar {
            display: flex;
            align-items: center;
            margin: 8px 0;
            gap: 10px;
        }
        .layer-label { width: 60px; font-size: 0.9em; color: #888; }
        .layer-bar-container {
            flex: 1;
            height: 24px;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        .layer-bar-fill {
            height: 100%;
            border-radius: 12px;
            transition: width 0.3s ease;
        }
        .refusal { background: linear-gradient(90deg, #ff4757, #ff6b81); }
        .acceptance { background: linear-gradient(90deg, #2ed573, #7bed9f); }
        .neutral { background: linear-gradient(90deg, #ffa502, #ff7f50); }
        
        /* Attack Progress */
        .attack-progress { height: 200px; }
        #loss-chart { width: 100%; height: 150px; }
        .attack-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }
        .stat-box {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 0.8em; color: #888; margin-top: 5px; }
        
        /* Attention Heatmap */
        .attention-heatmap { height: 350px; overflow: auto; }
        #heatmap-svg { width: 100%; min-height: 300px; }
        .heatmap-cell { stroke: rgba(255,255,255,0.1); stroke-width: 0.5; }
        
        /* Token Probabilities */
        .token-probs { }
        .token-bar {
            display: flex;
            align-items: center;
            margin: 6px 0;
            gap: 10px;
        }
        .token-text { 
            width: 80px; 
            font-family: monospace;
            font-size: 0.9em;
            color: #7bed9f;
            text-overflow: ellipsis;
            overflow: hidden;
        }
        .token-prob-bar {
            flex: 1;
            height: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            overflow: hidden;
        }
        .token-prob-fill {
            height: 100%;
            background: linear-gradient(90deg, #7b2ff7, #00d4ff);
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .token-prob-value { width: 50px; text-align: right; font-size: 0.9em; color: #888; }
        
        /* Status */
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.8);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.9em;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            background: #2ed573;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .full-width { grid-column: 1 / -1; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 MIRA Live Visualization</h1>
        <div class="subtitle">Real-time LLM Attack Monitoring</div>
    </div>
    
    <div class="container">
        <div class="panel layer-flow">
            <h2>Layer Processing Flow</h2>
            <div id="layer-container"></div>
        </div>
        
        <div class="panel attack-progress">
            <h2>Attack Optimization</h2>
            <svg id="loss-chart"></svg>
            <div class="attack-stats">
                <div class="stat-box">
                    <div class="stat-value" id="current-step">0</div>
                    <div class="stat-label">Step</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="current-loss">--</div>
                    <div class="stat-label">Loss</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="best-loss">--</div>
                    <div class="stat-label">Best Loss</div>
                </div>
            </div>
        </div>
        
        <div class="panel attention-heatmap">
            <h2>Attention Weights</h2>
            <svg id="heatmap-svg"></svg>
        </div>
        
        <div class="panel token-probs">
            <h2>Token Probabilities</h2>
            <div id="token-container"></div>
        </div>
        
        <div class="panel full-width">
            <h2>Current Suffix</h2>
            <div id="suffix-display" style="font-family: monospace; font-size: 1.2em; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 8px; word-break: break-all;">
                Waiting for attack...
            </div>
        </div>
    </div>
    
    <div class="status-bar">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div class="status-dot"></div>
            <span id="status-text">Connected - Waiting for events...</span>
        </div>
        <div id="event-count">Events: 0</div>
    </div>
    
    <script>
        let eventCount = 0;
        let lossHistory = [];
        let bestLoss = Infinity;
        
        // Setup SSE connection
        const evtSource = new EventSource('/api/events');
        
        evtSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.event_type === 'ping') return;
            
            eventCount++;
            document.getElementById('event-count').textContent = `Events: ${eventCount}`;
            document.getElementById('status-text').textContent = `Receiving: ${data.event_type}`;
            
            handleEvent(data);
        };
        
        function handleEvent(event) {
            switch(event.event_type) {
                case 'layer':
                    updateLayerFlow(event.data);
                    break;
                case 'attack_step':
                    updateAttackProgress(event.data);
                    break;
                case 'attention':
                    updateHeatmap(event.data);
                    break;
                case 'token':
                    updateTokenProbs(event.data);
                    break;
                case 'complete':
                    showComplete(event.data);
                    break;
            }
        }
        
        function updateLayerFlow(data) {
            const container = document.getElementById('layer-container');
            const layerId = `layer-${data.layer}`;
            let elem = document.getElementById(layerId);
            
            if (!elem) {
                elem = document.createElement('div');
                elem.id = layerId;
                elem.className = 'layer-bar';
                elem.innerHTML = `
                    <div class="layer-label">Layer ${data.layer}</div>
                    <div class="layer-bar-container">
                        <div class="layer-bar-fill ${data.direction}" style="width: 0%"></div>
                    </div>
                `;
                container.appendChild(elem);
            }
            
            const score = Math.max(data.refusal_score, data.acceptance_score);
            const pct = Math.min(score * 100, 100);
            elem.querySelector('.layer-bar-fill').style.width = pct + '%';
            elem.querySelector('.layer-bar-fill').className = `layer-bar-fill ${data.direction}`;
        }
        
        function updateAttackProgress(data) {
            document.getElementById('current-step').textContent = data.step;
            document.getElementById('current-loss').textContent = data.loss.toFixed(4);
            document.getElementById('suffix-display').textContent = data.suffix || '...';
            
            lossHistory.push(data.loss);
            if (data.loss < bestLoss) bestLoss = data.loss;
            document.getElementById('best-loss').textContent = bestLoss.toFixed(4);
            
            drawLossChart();
        }
        
        function drawLossChart() {
            const svg = d3.select('#loss-chart');
            svg.selectAll('*').remove();
            
            const width = svg.node().getBoundingClientRect().width;
            const height = 150;
            const margin = {top: 10, right: 10, bottom: 20, left: 40};
            
            const x = d3.scaleLinear()
                .domain([0, Math.max(lossHistory.length - 1, 1)])
                .range([margin.left, width - margin.right]);
            
            const y = d3.scaleLinear()
                .domain([0, d3.max(lossHistory) || 1])
                .nice()
                .range([height - margin.bottom, margin.top]);
            
            const line = d3.line()
                .x((d, i) => x(i))
                .y(d => y(d));
            
            svg.append('path')
                .datum(lossHistory)
                .attr('fill', 'none')
                .attr('stroke', '#00d4ff')
                .attr('stroke-width', 2)
                .attr('d', line);
            
            svg.append('g')
                .attr('transform', `translate(0,${height - margin.bottom})`)
                .call(d3.axisBottom(x).ticks(5))
                .attr('color', '#888');
            
            svg.append('g')
                .attr('transform', `translate(${margin.left},0)`)
                .call(d3.axisLeft(y).ticks(3))
                .attr('color', '#888');
        }
        
        function updateHeatmap(data) {
            const svg = d3.select('#heatmap-svg');
            svg.selectAll('*').remove();
            
            const tokens = data.tokens;
            const weights = data.weights;
            const n = tokens.length;
            const cellSize = Math.min(25, 300 / n);
            
            const colorScale = d3.scaleSequential(d3.interpolateViridis)
                .domain([0, 1]);
            
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    svg.append('rect')
                        .attr('x', j * cellSize + 60)
                        .attr('y', i * cellSize + 20)
                        .attr('width', cellSize - 1)
                        .attr('height', cellSize - 1)
                        .attr('fill', colorScale(weights[i]?.[j] || 0))
                        .attr('class', 'heatmap-cell');
                }
            }
            
            // Token labels
            tokens.forEach((t, i) => {
                svg.append('text')
                    .attr('x', 55)
                    .attr('y', i * cellSize + cellSize/2 + 20)
                    .attr('text-anchor', 'end')
                    .attr('fill', '#888')
                    .attr('font-size', '10px')
                    .text(t.slice(0, 6));
            });
        }
        
        function updateTokenProbs(data) {
            const container = document.getElementById('token-container');
            container.innerHTML = '';
            
            (data.top_tokens || []).slice(0, 8).forEach(([token, prob]) => {
                const div = document.createElement('div');
                div.className = 'token-bar';
                div.innerHTML = `
                    <div class="token-text">${token}</div>
                    <div class="token-prob-bar">
                        <div class="token-prob-fill" style="width: ${prob * 100}%"></div>
                    </div>
                    <div class="token-prob-value">${(prob * 100).toFixed(1)}%</div>
                `;
                container.appendChild(div);
            });
        }
        
        function showComplete(data) {
            document.getElementById('status-text').textContent = 
                `Complete! ASR: ${(data.asr * 100).toFixed(1)}%`;
            document.querySelector('.status-dot').style.background = '#2ed573';
        }
        
        // Initialize
        drawLossChart();
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
