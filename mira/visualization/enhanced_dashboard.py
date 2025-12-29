"""
Enhanced Live Dashboard with Transformer Internals Visualization

A completely redesigned dashboard that shows:
1. Attack progress and metrics
2. Transformer internal processing in real-time
3. Normal vs Adversarial comparison
4. Layer-by-layer residual stream changes
"""

ENHANCED_DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA Neural Attack Monitor - Transformer Internals</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --primary: #00f5ff;
            --secondary: #ff00ff;
            --accent: #ffff00;
            --danger: #ff4466;
            --success: #00ff88;
            --bg-dark: #0a0a0f;
            --bg-panel: rgba(15, 25, 35, 0.7);
            --border: rgba(0, 255, 255, 0.3);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Space Mono', monospace;
            background: #0a0a0f;
            color: #ddd;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated Background */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: 
                radial-gradient(ellipse at 20% 30%, rgba(0, 245, 255, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 70%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(255, 255, 0, 0.05) 0%, transparent 50%);
            animation: bgPulse 15s ease-in-out infinite alternate;
            z-index: -1;
        }
        
        @keyframes bgPulse {
            0% { opacity: 0.5; transform: scale(1); }
            100% { opacity: 1; transform: scale(1.1); }
        }
        
        /* Grid pattern overlay */
        body::after {
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-image: 
                linear-gradient(rgba(0, 245, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 245, 255, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: -1;
            pointer-events: none;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(15, 25, 45, 0.9) 100%);
            backdrop-filter: blur(20px);
            border-bottom: 2px solid rgba(0, 245, 255, 0.4);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo-icon {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            animation: float 3s ease-in-out infinite;
            box-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-8px); }
        }
        
        .logo h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.6em;
            background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shimmer 3s linear infinite;
        }
        
        @keyframes shimmer {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }
        
        .stats-bar {
            display: flex;
            gap: 30px;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5em;
            color: var(--primary);
            text-shadow: 0 0 10px var(--primary);
        }
        
        .stat-label {
            font-size: 0.7em;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Main Container */
        .main-container {
            display: grid;
            grid-template-columns: 1fr 1.5fr;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        /* Panels */
        .panel {
            background: rgba(15, 25, 35, 0.6);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 16px;
            padding: 20px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .panel:hover {
            border-color: rgba(0, 255, 255, 0.5);
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        }
        
        .panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--primary), transparent);
            animation: scanline 4s linear infinite;
        }
        
        @keyframes scanline {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .panel-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 1em;
            color: var(--primary);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .panel-title::before {
            content: '◆';
            animation: blink 1.5s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        /* Left Column */
        .left-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        /* Right Column - Transformer */
        .right-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        /* Attack Console */
        .console {
            background: #000;
            border-radius: 8px;
            padding: 15px;
            height: 200px;
            overflow-y: auto;
            font-size: 0.85em;
            line-height: 1.6;
        }
        
        .console-line {
            margin-bottom: 5px;
            padding-left: 10px;
            border-left: 2px solid var(--primary);
        }
        
        .console-line.success { border-color: var(--success); color: var(--success); }
        .console-line.error { border-color: var(--danger); color: var(--danger); }
        .console-line.info { border-color: var(--accent); color: var(--accent); }
        
        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }
        
        .metric-card {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(0, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            border-color: var(--primary);
            transform: scale(1.05);
        }
        
        .metric-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8em;
            color: var(--primary);
        }
        
        .metric-label {
            font-size: 0.7em;
            color: #666;
            margin-top: 5px;
        }
        
        /* Token Display */
        .token-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
        }
        
        .token {
            padding: 6px 12px;
            background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(255, 0, 255, 0.1));
            border: 1px solid rgba(0, 245, 255, 0.3);
            border-radius: 6px;
            font-size: 0.9em;
            transition: all 0.3s ease;
        }
        
        .token:hover {
            background: rgba(0, 245, 255, 0.2);
            transform: translateY(-2px);
        }
        
        .token.highlight {
            background: rgba(255, 255, 0, 0.2);
            border-color: var(--accent);
        }
        
        /* Embeddings Heatmap */
        .embeddings-heatmap {
            width: 100%;
            height: 150px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Comparison View */
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .comparison-side {
            background: rgba(0, 0, 0, 0.4);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .comparison-side.normal { border-left: 3px solid var(--primary); }
        .comparison-side.adversarial { border-left: 3px solid var(--danger); }
        
        .comparison-title {
            font-size: 0.9em;
            margin-bottom: 10px;
            color: #aaa;
        }
        
        .comparison-side.normal .comparison-title { color: var(--primary); }
        .comparison-side.adversarial .comparison-title { color: var(--danger); }
        
        /* Residual Stream */
        .residual-container {
            height: 150px;
            position: relative;
        }
        
        .residual-svg {
            width: 100%;
            height: 100%;
        }
        
        .layer-node {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .layer-node:hover {
            transform: scale(1.1);
        }
        
        /* Layer Flow */
        .layer-flow {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 10px 0;
        }
        
        .layer-box {
            min-width: 80px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(0, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .layer-box.active {
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .layer-id {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.2em;
            color: var(--primary);
        }
        
        .layer-delta {
            font-size: 0.8em;
            margin-top: 5px;
        }
        
        .layer-delta.high { color: var(--danger); }
        .layer-delta.medium { color: var(--accent); }
        .layer-delta.low { color: var(--success); }
        
        /* Progress Bar */
        .progress-bar {
            height: 6px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 0.3s ease;
            box-shadow: 0 0 10px var(--primary);
        }
        
        /* Suffix Display */
        .suffix-box {
            background: rgba(0, 0, 0, 0.5);
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 0.9em;
            word-break: break-all;
            border: 1px solid rgba(255, 0, 255, 0.3);
        }
        
        /* Loss Chart */
        .chart-container {
            height: 200px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
        }
        
        /* Status Bar */
        .status-bar {
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid rgba(0, 255, 255, 0.2);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .pulse-dot {
            width: 10px;
            height: 10px;
            background: var(--success);
            border-radius: 50%;
            animation: pulseDot 1.5s infinite;
        }
        
        @keyframes pulseDot {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.7; }
        }
        
        .version {
            font-size: 0.8em;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <div class="logo-icon">🧠</div>
            <h1>MIRA NEURAL MONITOR</h1>
        </div>
        <div class="stats-bar">
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
                <div class="stat-value" id="asr">0%</div>
                <div class="stat-label">ASR</div>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="left-column">
            <div class="panel">
                <div class="panel-title">Attack Console</div>
                <div class="console" id="console"></div>
            </div>
            
            <div class="panel">
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
                        <div class="metric-value" id="success-rate">0%</div>
                        <div class="metric-label">Success</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Layer Flow</div>
                <div class="layer-flow" id="layer-flow"></div>
            </div>
            
            <div class="panel">
                <div class="panel-title">Adversarial Suffix</div>
                <div class="suffix-box" id="suffix">Waiting for attack...</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="right-column">
            <div class="panel">
                <div class="panel-title">🔬 Token Embeddings</div>
                <div class="token-container" id="tokens"></div>
                <canvas id="embeddings-canvas" class="embeddings-heatmap"></canvas>
            </div>
            
            <div class="panel">
                <div class="panel-title">🔄 Transformer Trace Comparison</div>
                <div class="comparison">
                    <div class="comparison-side normal">
                        <div class="comparison-title">◈ Normal Prompt</div>
                        <div id="normal-trace">Waiting for trace data...</div>
                    </div>
                    <div class="comparison-side adversarial">
                        <div class="comparison-title">◈ Adversarial Prompt</div>
                        <div id="adv-trace">Waiting for attack...</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">📊 Residual Stream Changes</div>
                <div class="residual-container">
                    <svg id="residual-svg" class="residual-svg"></svg>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">📈 Loss Trajectory</div>
                <div class="chart-container">
                    <svg id="loss-chart" width="100%" height="100%"></svg>
                </div>
            </div>
        </div>
    </div>
    
    <div class="status-bar">
        <div class="status-indicator">
            <div class="pulse-dot"></div>
            <span id="status">Connecting to attack pipeline...</span>
        </div>
        <div class="version">MIRA Framework v2.0 | Transformer Internals Monitor</div>
    </div>
    
    <script>
        // State
        let eventCount = 0;
        let lossHistory = [];
        let bestLoss = Infinity;
        let currentStep = 0;
        let normalTrace = null;
        let advTrace = null;
        
        // Connect to SSE
        const evtSource = new EventSource('/api/events');
        
        function log(message, type = 'default') {
            const console = document.getElementById('console');
            const line = document.createElement('div');
            line.className = 'console-line ' + type;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            console.appendChild(line);
            console.scrollTop = console.scrollHeight;
        }
        
        log('Neural monitor initialized', 'success');
        document.getElementById('status').textContent = 'Connected - Waiting for data';
        
        evtSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            if (data.event_type === 'ping') return;
            
            eventCount++;
            document.getElementById('events').textContent = eventCount;
            
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
                case 'embeddings':
                    showEmbeddings(data.data);
                    break;
                case 'transformer_trace':
                    showTrace(data.data);
                    break;
                case 'residual':
                    showResidual(data.data);
                    break;
            }
        };
        
        function updateLayer(data) {
            const flow = document.getElementById('layer-flow');
            let box = document.getElementById('layer-' + data.layer);
            
            if (!box) {
                box = document.createElement('div');
                box.id = 'layer-' + data.layer;
                box.className = 'layer-box';
                box.innerHTML = `
                    <div class="layer-id">L${data.layer}</div>
                    <div class="layer-delta" id="delta-${data.layer}">--</div>
                `;
                flow.appendChild(box);
            }
            
            const score = Math.max(data.refusal_score || 0, data.acceptance_score || 0);
            box.classList.toggle('active', score > 0.3);
        }
        
        function updateAttack(data) {
            currentStep = data.step;
            const loss = data.loss;
            
            document.getElementById('step').textContent = currentStep;
            document.getElementById('current-step').textContent = currentStep;
            document.getElementById('loss').textContent = loss.toFixed(4);
            document.getElementById('current-loss').textContent = loss.toFixed(4);
            
            if (loss < bestLoss) {
                bestLoss = loss;
                document.getElementById('best-loss').textContent = loss.toFixed(4);
            }
            
            lossHistory.push(loss);
            drawLossChart();
            
            if (data.suffix) {
                document.getElementById('suffix').textContent = data.suffix;
            }
            
            // Progress bar (assume 30 steps)
            const progress = Math.min((currentStep / 30) * 100, 100);
            document.getElementById('progress').style.width = progress + '%';
            
            document.getElementById('status').textContent = `Attack step ${currentStep}`;
            
            if (data.success) {
                log('✓ Attack succeeded!', 'success');
                document.getElementById('success-rate').textContent = '100%';
            }
        }
        
        function showComplete(data) {
            log('Pipeline complete', 'success');
            document.getElementById('status').textContent = 'Complete!';
            if (data.asr !== undefined) {
                document.getElementById('asr').textContent = (data.asr * 100).toFixed(0) + '%';
            }
        }
        
        function showEmbeddings(data) {
            // Display tokens
            const tokenContainer = document.getElementById('tokens');
            tokenContainer.innerHTML = '';
            
            if (data.tokens) {
                data.tokens.forEach((token, i) => {
                    const span = document.createElement('span');
                    span.className = 'token';
                    span.textContent = token;
                    span.title = `Token ${i}`;
                    tokenContainer.appendChild(span);
                });
            }
            
            // Draw embeddings heatmap
            const canvas = document.getElementById('embeddings-canvas');
            const ctx = canvas.getContext('2d');
            
            if (data.embeddings && data.embeddings.length > 0) {
                const embeddings = data.embeddings;
                const seqLen = embeddings.length;
                const hiddenDim = embeddings[0].length;
                
                canvas.width = canvas.offsetWidth;
                canvas.height = canvas.offsetHeight;
                
                const cellWidth = canvas.width / Math.min(hiddenDim, 128);
                const cellHeight = canvas.height / seqLen;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                for (let i = 0; i < seqLen; i++) {
                    for (let j = 0; j < Math.min(hiddenDim, 128); j++) {
                        const val = embeddings[i][j];
                        const intensity = Math.min(Math.abs(val) * 3, 1);
                        const color = val > 0 
                            ? `rgba(0, 245, 255, ${intensity})` 
                            : `rgba(255, 68, 102, ${intensity})`;
                        ctx.fillStyle = color;
                        ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
                    }
                }
                
                log('📊 Embeddings visualized: ' + seqLen + ' tokens', 'info');
            }
        }
        
        function showTrace(data) {
            const traceType = data.trace_type;
            const trace = data.trace;
            
            const targetId = traceType === 'normal' ? 'normal-trace' : 'adv-trace';
            const container = document.getElementById(targetId);
            
            if (traceType === 'normal') normalTrace = trace;
            else advTrace = trace;
            
            if (!trace || !trace.layers) {
                container.innerHTML = '<p style="color: #666;">No data available</p>';
                return;
            }
            
            let html = '<div style="font-size: 0.85em; line-height: 1.8;">';
            html += `<div><strong>Tokens:</strong> ${trace.tokens ? trace.tokens.length : 0}</div>`;
            html += `<div><strong>Layers:</strong> ${trace.layers.length}</div>`;
            
            if (trace.layers.length > 0) {
                const lastLayer = trace.layers[trace.layers.length - 1];
                const norm = lastLayer.residual_norm || 0;
                html += `<div><strong>Final Residual:</strong> ${norm.toFixed(4)}</div>`;
            }
            
            html += '</div>';
            container.innerHTML = html;
            
            log(`🔄 ${traceType} trace received`, 'info');
            
            // Compare if both available
            if (normalTrace && advTrace && traceType === 'adversarial') {
                compareTraces();
            }
        }
        
        function showResidual(data) {
            const svg = d3.select('#residual-svg');
            const width = svg.node().getBoundingClientRect().width;
            const height = 140;
            
            const layerIdx = data.layer_idx;
            const residualNorm = data.residual_norm || 1;
            const deltaNorm = data.delta_norm || 0;
            
            const x = 40 + (layerIdx * 70);
            const y = height / 2;
            const radius = Math.min(residualNorm * 15, 30);
            
            // Draw circle
            svg.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', radius)
                .attr('fill', 'rgba(0, 245, 255, 0.2)')
                .attr('stroke', '#00f5ff')
                .attr('stroke-width', 2);
            
            // Layer label
            svg.append('text')
                .attr('x', x)
                .attr('y', y + 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#00f5ff')
                .attr('font-size', '12px')
                .attr('font-family', 'Orbitron')
                .text(`L${layerIdx}`);
            
            // Delta label
            const deltaColor = deltaNorm > 1 ? '#ff4466' : (deltaNorm > 0.5 ? '#ffff00' : '#00ff88');
            svg.append('text')
                .attr('x', x)
                .attr('y', y + radius + 18)
                .attr('text-anchor', 'middle')
                .attr('fill', deltaColor)
                .attr('font-size', '10px')
                .text(`Δ${deltaNorm.toFixed(2)}`);
            
            // Update layer box delta
            const deltaEl = document.getElementById('delta-' + layerIdx);
            if (deltaEl) {
                deltaEl.textContent = `Δ ${deltaNorm.toFixed(3)}`;
                deltaEl.className = 'layer-delta ' + (deltaNorm > 1 ? 'high' : (deltaNorm > 0.5 ? 'medium' : 'low'));
            }
        }
        
        function compareTraces() {
            if (!normalTrace || !advTrace) return;
            
            log('📊 Comparing normal vs adversarial traces', 'info');
            
            // Find most affected layer
            let maxDiff = 0;
            let maxLayer = 0;
            
            for (let i = 0; i < Math.min(normalTrace.layers.length, advTrace.layers.length); i++) {
                const nLayer = normalTrace.layers[i];
                const aLayer = advTrace.layers[i];
                
                if (nLayer.residual_norm && aLayer.residual_norm) {
                    const diff = Math.abs(nLayer.residual_norm - aLayer.residual_norm);
                    if (diff > maxDiff) {
                        maxDiff = diff;
                        maxLayer = i;
                    }
                }
            }
            
            log(`🎯 Most affected layer: L${maxLayer} (Δ = ${maxDiff.toFixed(4)})`, 'success');
        }
        
        function drawLossChart() {
            const svg = d3.select('#loss-chart');
            const container = svg.node().parentNode;
            const width = container.offsetWidth;
            const height = container.offsetHeight;
            
            svg.selectAll('*').remove();
            svg.attr('width', width).attr('height', height);
            
            if (lossHistory.length < 2) return;
            
            const margin = {top: 20, right: 20, bottom: 30, left: 50};
            const w = width - margin.left - margin.right;
            const h = height - margin.top - margin.bottom;
            
            const x = d3.scaleLinear()
                .domain([0, lossHistory.length - 1])
                .range([margin.left, w + margin.left]);
            
            const y = d3.scaleLinear()
                .domain([0, d3.max(lossHistory) * 1.1])
                .range([h + margin.top, margin.top]);
            
            // Grid
            svg.append('g')
                .attr('transform', `translate(0,${h + margin.top})`)
                .call(d3.axisBottom(x).ticks(5))
                .attr('color', '#333');
            
            svg.append('g')
                .attr('transform', `translate(${margin.left},0)`)
                .call(d3.axisLeft(y).ticks(4))
                .attr('color', '#333');
            
            // Area
            const area = d3.area()
                .x((d, i) => x(i))
                .y0(h + margin.top)
                .y1(d => y(d));
            
            const gradient = svg.append('defs')
                .append('linearGradient')
                .attr('id', 'lossGradient')
                .attr('x1', '0%').attr('y1', '0%')
                .attr('x2', '0%').attr('y2', '100%');
            
            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', '#00f5ff')
                .attr('stop-opacity', 0.4);
            
            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', '#00f5ff')
                .attr('stop-opacity', 0);
            
            svg.append('path')
                .datum(lossHistory)
                .attr('fill', 'url(#lossGradient)')
                .attr('d', area);
            
            // Line
            const line = d3.line()
                .x((d, i) => x(i))
                .y(d => y(d));
            
            svg.append('path')
                .datum(lossHistory)
                .attr('fill', 'none')
                .attr('stroke', '#00f5ff')
                .attr('stroke-width', 2)
                .attr('d', line);
        }
        
        // Initialize
        window.addEventListener('resize', drawLossChart);
    </script>
</body>
</html>
'''


def get_enhanced_dashboard():
    """Return the enhanced dashboard HTML."""
    return ENHANCED_DASHBOARD_HTML
