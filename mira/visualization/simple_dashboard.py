"""
Simplified Transformer Visualization Dashboard

A clean, reliable dashboard that works without complex dependencies.
Shows attack progress and basic transformer information.
"""

SIMPLE_DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA Attack Monitor</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0, 0, 0, 0.8);
            padding: 20px 30px;
            border-bottom: 2px solid #00d4ff;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        
        .status-badge {
            padding: 8px 16px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 20px;
            color: #00ff88;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .main {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .panel {
            background: rgba(20, 30, 50, 0.9);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 10px;
            padding: 20px;
        }
        
        .panel-title {
            color: #00d4ff;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat {
            text-align: center;
            padding: 15px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .stat-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            margin-top: 5px;
        }
        
        .console {
            background: #000;
            border-radius: 8px;
            padding: 15px;
            height: 250px;
            overflow-y: auto;
            font-size: 13px;
            line-height: 1.6;
        }
        
        .log-entry {
            padding: 4px 0;
            border-left: 3px solid #444;
            padding-left: 10px;
            margin-bottom: 4px;
        }
        
        .log-entry.info { border-color: #00d4ff; }
        .log-entry.success { border-color: #00ff88; }
        .log-entry.error { border-color: #ff4466; }
        .log-entry.attack { border-color: #ffaa00; }
        
        .progress-container {
            margin-top: 15px;
        }
        
        .progress-bar {
            height: 8px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #ff00ff);
            transition: width 0.3s;
            border-radius: 4px;
        }
        
        .tokens-display {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 10px 0;
        }
        
        .token {
            padding: 4px 10px;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 4px;
            font-size: 12px;
        }
        
        .attack-info {
            background: rgba(0, 0, 0, 0.4);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .attack-info h4 {
            color: #ffaa00;
            margin-bottom: 10px;
        }
        
        .suffix-display {
            background: #000;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            word-break: break-all;
            color: #ff00ff;
        }
        
        .chart-container {
            height: 200px;
            margin-top: 15px;
        }
        
        #loss-chart {
            width: 100%;
            height: 100%;
        }
        
        .layer-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 8px;
            margin-top: 10px;
        }
        
        .layer-box {
            padding: 8px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 4px;
            text-align: center;
            font-size: 11px;
            transition: all 0.3s;
        }
        
        .layer-box.active {
            background: rgba(0, 212, 255, 0.3);
            border: 1px solid #00d4ff;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #555;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🧠 MIRA Attack Monitor</div>
        <div class="status-badge" id="status">Connecting...</div>
    </div>
    
    <div class="stats">
        <div class="stat">
            <div class="stat-value" id="events">0</div>
            <div class="stat-label">Events</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="step">0</div>
            <div class="stat-label">Step</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="loss">--</div>
            <div class="stat-label">Loss</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="layers">0</div>
            <div class="stat-label">Layers</div>
        </div>
    </div>
    
    <div class="main">
        <div class="panel">
            <div class="panel-title">📋 Attack Console</div>
            <div class="console" id="console"></div>
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-title">🔤 Tokens</div>
            <div class="tokens-display" id="tokens">
                Waiting for data...
            </div>
            <div class="attack-info">
                <h4>⚡ Current Suffix</h4>
                <div class="suffix-display" id="suffix">Waiting for attack...</div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-title">📊 Loss Trajectory</div>
            <div class="chart-container">
                <svg id="loss-chart"></svg>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-title">🔄 Layer Activity</div>
            <div class="layer-grid" id="layer-grid"></div>
        </div>
    </div>
    
    <div class="footer">
        MIRA Framework | Real-time Attack Visualization
    </div>

    <script>
        let eventCount = 0;
        let lossHistory = [];
        let currentStep = 0;
        let numLayers = 0;
        
        const evtSource = new EventSource('/api/events');
        
        function log(msg, type = 'info') {
            const consoleEl = document.getElementById('console');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + type;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            consoleEl.appendChild(entry);
            
            // Keep only last 50 lines for performance
            while (consoleEl.children.length > 50) {
                consoleEl.removeChild(consoleEl.firstChild);
            }
            
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }
        
        log('Dashboard initialized', 'success');
        document.getElementById('status').textContent = 'Connected';
        document.getElementById('status').style.background = 'rgba(0, 255, 136, 0.2)';
        
        evtSource.onmessage = function(e) {
            try {
                const data = JSON.parse(e.data);
                if (data.event_type === 'ping') return;
                
                eventCount++;
                document.getElementById('events').textContent = eventCount;
                
                switch(data.event_type) {
                    case 'attack_step':
                        handleAttackStep(data.data);
                        break;
                    case 'layer':
                        handleLayer(data.data);
                        break;
                    case 'embeddings':
                        handleEmbeddings(data.data);
                        break;
                    case 'transformer_trace':
                        handleTrace(data.data);
                        break;
                    case 'attention_matrix':
                        handleAttention(data.data);
                        break;
                    case 'complete':
                        handleComplete(data.data);
                        break;
                    default:
                        log(`Event: ${data.event_type}`, 'info');
                }
            } catch(err) {
                console.error('Parse error:', err);
            }
        };
        
        evtSource.onerror = function() {
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').style.background = 'rgba(255, 68, 102, 0.2)';
            document.getElementById('status').style.borderColor = '#ff4466';
            document.getElementById('status').style.color = '#ff4466';
        };
        
        function handleAttackStep(data) {
            currentStep = data.step || 0;
            const loss = data.loss || 0;
            
            document.getElementById('step').textContent = currentStep;
            document.getElementById('loss').textContent = loss.toFixed(4);
            
            lossHistory.push(loss);
            if (lossHistory.length > 50) lossHistory.shift();
            
            drawLossChart();
            
            if (data.suffix) {
                document.getElementById('suffix').textContent = data.suffix.substring(0, 100);
            }
            
            const progress = Math.min((currentStep / 30) * 100, 100);
            document.getElementById('progress').style.width = progress + '%';
            
            log(`Step ${currentStep}: loss = ${loss.toFixed(4)}`, 'attack');
            
            if (data.success) {
                log('✓ Attack succeeded!', 'success');
            }
        }
        
        function handleLayer(data) {
            const layer = data.layer || 0;
            if (layer >= numLayers) {
                numLayers = layer + 1;
                document.getElementById('layers').textContent = numLayers;
                renderLayerGrid();
            }
            
            // Highlight active layer
            const box = document.getElementById('layer-' + layer);
            if (box) {
                box.classList.add('active');
                setTimeout(() => box.classList.remove('active'), 500);
            }
        }
        
        function handleEmbeddings(data) {
            if (data.tokens) {
                const container = document.getElementById('tokens');
                container.innerHTML = '';
                data.tokens.forEach(token => {
                    const span = document.createElement('span');
                    span.className = 'token';
                    span.textContent = token.replace('Ġ', ' ').replace('▁', '_');
                    container.appendChild(span);
                });
                log(`Received ${data.tokens.length} tokens`, 'info');
            }
        }
        
        function handleTrace(data) {
            const trace = data.trace || data;
            if (trace && trace.layers) {
                numLayers = trace.layers.length;
                document.getElementById('layers').textContent = numLayers;
                renderLayerGrid();
                log(`Trace: ${numLayers} layers`, 'info');
            }
            if (trace && trace.tokens) {
                handleEmbeddings({tokens: trace.tokens});
            }
        }
        
        function handleAttention(data) {
            log(`Attention L${data.layer} H${data.head}`, 'info');
        }
        
        function handleComplete(data) {
            log('Pipeline complete!', 'success');
            document.getElementById('status').textContent = 'Complete';
            if (data.asr !== undefined) {
                log(`ASR: ${(data.asr * 100).toFixed(1)}%`, 'success');
            }
        }
        
        function renderLayerGrid() {
            const grid = document.getElementById('layer-grid');
            grid.innerHTML = '';
            for (let i = 0; i < numLayers; i++) {
                const box = document.createElement('div');
                box.className = 'layer-box';
                box.id = 'layer-' + i;
                box.textContent = 'L' + i;
                grid.appendChild(box);
            }
        }
        
        function drawLossChart() {
            if (lossHistory.length < 2) return;
            
            const svg = d3.select('#loss-chart');
            const container = svg.node().parentNode;
            const width = container.offsetWidth;
            const height = container.offsetHeight;
            
            svg.selectAll('*').remove();
            svg.attr('width', width).attr('height', height);
            
            const margin = {top: 10, right: 10, bottom: 20, left: 40};
            const w = width - margin.left - margin.right;
            const h = height - margin.top - margin.bottom;
            
            const x = d3.scaleLinear()
                .domain([0, lossHistory.length - 1])
                .range([margin.left, w + margin.left]);
            
            const y = d3.scaleLinear()
                .domain([0, d3.max(lossHistory) * 1.1])
                .range([h + margin.top, margin.top]);
            
            // Area
            const area = d3.area()
                .x((d, i) => x(i))
                .y0(h + margin.top)
                .y1(d => y(d));
            
            svg.append('path')
                .datum(lossHistory)
                .attr('fill', 'rgba(0, 212, 255, 0.2)')
                .attr('d', area);
            
            // Line
            const line = d3.line()
                .x((d, i) => x(i))
                .y(d => y(d));
            
            svg.append('path')
                .datum(lossHistory)
                .attr('fill', 'none')
                .attr('stroke', '#00d4ff')
                .attr('stroke-width', 2)
                .attr('d', line);
            
            // Axes
            svg.append('g')
                .attr('transform', `translate(0,${h + margin.top})`)
                .call(d3.axisBottom(x).ticks(5))
                .attr('color', '#444');
            
            svg.append('g')
                .attr('transform', `translate(${margin.left},0)`)
                .call(d3.axisLeft(y).ticks(4))
                .attr('color', '#444');
        }
        
        // Initialize layer grid
        renderLayerGrid();
    </script>
</body>
</html>
'''


def get_simple_dashboard():
    """Return the simplified dashboard HTML."""
    return SIMPLE_DASHBOARD_HTML
