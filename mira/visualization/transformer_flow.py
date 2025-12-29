"""
Transformer Flow Visualization Dashboard

Shows the complete transformer processing flow:
- Input Tokens → Embedding
- Embedding → Q/K/V Projections  
- Q·K^T → Attention Weights
- Attention × V → Attention Output
- MLP Processing
- Output Logits → Next Token Prediction

Inspired by transformer-explainer but adapted for MIRA attack analysis.
"""

TRANSFORMER_FLOW_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA - Transformer Flow</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --bg: #0f0f1a;
            --panel: #1a1a2e;
            --border: #2d2d44;
            --text: #e0e0e0;
            --primary: #00d4ff;
            --secondary: #ff6b9d;
            --accent: #ffd93d;
            --success: #00ff88;
            --embedding: #4facfe;
            --query: #00c6fb;
            --key: #ff6b9d;
            --value: #c471ed;
            --attention: #48c6ef;
            --mlp: #f093fb;
            --output: #ffd93d;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .header {
            background: linear-gradient(90deg, var(--panel) 0%, rgba(26, 26, 46, 0.8) 100%);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }
        
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: var(--primary);
        }
        
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 20px;
            font-size: 13px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .main {
            display: flex;
            height: calc(100vh - 60px);
        }
        
        /* Left Panel - Transformer Flow */
        .flow-panel {
            flex: 2;
            padding: 20px;
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border);
            overflow-y: auto;
        }
        
        .flow-title {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #888;
            margin-bottom: 20px;
        }
        
        /* Flow Container */
        .flow-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            position: relative;
            min-height: 500px;
        }
        
        /* Flow Stage */
        .stage {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            padding: 15px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            min-width: 100px;
            transition: all 0.3s;
        }
        
        .stage:hover {
            border-color: var(--primary);
            transform: translateY(-5px);
        }
        
        .stage.active {
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        
        .stage-name {
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stage-tokens {
            display: flex;
            flex-direction: column;
            gap: 4px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .token-box {
            padding: 4px 8px;
            font-size: 11px;
            border-radius: 4px;
            text-align: center;
            min-width: 60px;
        }
        
        .token-box.embedding { background: rgba(79, 172, 254, 0.3); border: 1px solid var(--embedding); }
        .token-box.query { background: rgba(0, 198, 251, 0.3); border: 1px solid var(--query); }
        .token-box.key { background: rgba(255, 107, 157, 0.3); border: 1px solid var(--key); }
        .token-box.value { background: rgba(196, 113, 237, 0.3); border: 1px solid var(--value); }
        .token-box.attention { background: rgba(72, 198, 239, 0.3); border: 1px solid var(--attention); }
        .token-box.mlp { background: rgba(240, 147, 251, 0.3); border: 1px solid var(--mlp); }
        .token-box.output { background: rgba(255, 217, 61, 0.3); border: 1px solid var(--output); }
        
        /* Flow Arrow */
        .flow-arrow {
            font-size: 24px;
            color: #444;
        }
        
        /* Attention Matrix */
        .attention-section {
            margin-top: 20px;
            padding: 15px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        
        .attention-title {
            font-size: 13px;
            color: var(--attention);
            margin-bottom: 10px;
        }
        
        #attention-matrix {
            width: 100%;
            height: 200px;
        }
        
        /* Right Panel - Attack Info */
        .info-panel {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }
        
        .info-section {
            margin-bottom: 20px;
            padding: 15px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
        }
        
        .info-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            color: #888;
        }
        
        .metric-value {
            font-weight: bold;
            color: var(--primary);
        }
        
        /* Console */
        .console {
            background: #000;
            border-radius: 8px;
            padding: 10px;
            height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        
        .console-line {
            padding: 2px 0;
            color: #888;
        }
        
        .console-line.info { color: var(--primary); }
        .console-line.success { color: var(--success); }
        .console-line.attack { color: var(--accent); }
        
        /* Loss Chart */
        #loss-chart {
            width: 100%;
            height: 150px;
        }
        
        /* Layer Selector */
        .layer-nav {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .layer-btn {
            padding: 6px 12px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text);
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .layer-btn:hover, .layer-btn.active {
            background: rgba(0, 212, 255, 0.2);
            border-color: var(--primary);
        }
        
        /* Processing indicator */
        .processing {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            background: rgba(255, 217, 61, 0.1);
            border: 1px solid var(--accent);
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .processing-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🧠 MIRA Transformer Flow</div>
        <div class="status">
            <div class="status-item">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">Connecting...</span>
            </div>
            <div class="status-item">
                Layer: <span id="current-layer">0</span> / <span id="total-layers">?</span>
            </div>
            <div class="status-item">
                Step: <span id="current-step">0</span>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="flow-panel">
            <div class="flow-title">Transformer Processing Flow</div>
            
            <div class="layer-nav" id="layer-nav"></div>
            
            <div class="processing" id="processing" style="display: none;">
                <div class="processing-spinner"></div>
                <span id="processing-text">Processing...</span>
            </div>
            
            <div class="flow-container">
                <!-- Input -->
                <div class="stage" id="stage-input">
                    <div class="stage-name">Input</div>
                    <div class="stage-tokens" id="input-tokens"></div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- Embedding -->
                <div class="stage" id="stage-embedding">
                    <div class="stage-name">Embedding</div>
                    <div class="stage-tokens" id="embedding-tokens"></div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- Q/K/V -->
                <div class="stage" id="stage-qkv">
                    <div class="stage-name">Q · K · V</div>
                    <div style="display: flex; gap: 5px;">
                        <div class="stage-tokens">
                            <div class="token-box query">Q</div>
                        </div>
                        <div class="stage-tokens">
                            <div class="token-box key">K</div>
                        </div>
                        <div class="stage-tokens">
                            <div class="token-box value">V</div>
                        </div>
                    </div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- Attention -->
                <div class="stage" id="stage-attention">
                    <div class="stage-name">Attention</div>
                    <div class="stage-tokens" id="attention-out"></div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- MLP -->
                <div class="stage" id="stage-mlp">
                    <div class="stage-name">MLP</div>
                    <div class="stage-tokens" id="mlp-out"></div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- Output -->
                <div class="stage" id="stage-output">
                    <div class="stage-name">Output</div>
                    <div class="stage-tokens" id="output-tokens"></div>
                </div>
            </div>
            
            <!-- Attention Matrix -->
            <div class="attention-section">
                <div class="attention-title">Attention Matrix (Layer <span id="attn-layer">0</span>, Head <span id="attn-head">0</span>)</div>
                <svg id="attention-matrix"></svg>
            </div>
        </div>
        
        <div class="info-panel">
            <div class="info-section">
                <div class="info-title">Attack Status</div>
                <div class="metric">
                    <span class="metric-label">Step</span>
                    <span class="metric-value" id="attack-step">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Loss</span>
                    <span class="metric-value" id="attack-loss">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Best Loss</span>
                    <span class="metric-value" id="best-loss">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Suffix</span>
                    <span class="metric-value" id="suffix" style="font-size: 10px;">--</span>
                </div>
            </div>
            
            <div class="info-section">
                <div class="info-title">Loss Trajectory</div>
                <svg id="loss-chart"></svg>
            </div>
            
            <div class="info-section">
                <div class="info-title">Event Log</div>
                <div class="console" id="console"></div>
            </div>
        </div>
    </div>

    <script>
        // State
        let tokens = [];
        let currentLayer = 0;
        let totalLayers = 0;
        let lossHistory = [];
        let bestLoss = Infinity;
        let attentionData = null;
        
        // Connect to SSE
        const evtSource = new EventSource('/api/events');
        
        function log(msg, type = '') {
            const console = document.getElementById('console');
            const line = document.createElement('div');
            line.className = 'console-line ' + type;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            console.appendChild(line);
            console.scrollTop = console.scrollHeight;
        }
        
        evtSource.onopen = function() {
            document.getElementById('status-text').textContent = 'Connected';
            document.getElementById('status-dot').style.background = '#00ff88';
            log('Connected to server', 'success');
        };
        
        evtSource.onerror = function() {
            document.getElementById('status-text').textContent = 'Disconnected';
            document.getElementById('status-dot').style.background = '#ff4466';
        };
        
        evtSource.onmessage = function(e) {
            try {
                const data = JSON.parse(e.data);
                if (data.event_type === 'ping') return;
                
                handleEvent(data);
            } catch(err) {
                console.error('Parse error:', err);
            }
        };
        
        function handleEvent(data) {
            switch(data.event_type) {
                case 'layer':
                    handleLayer(data.data);
                    break;
                case 'attack_step':
                    handleAttackStep(data.data);
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
        }
        
        function handleLayer(data) {
            currentLayer = data.layer || 0;
            document.getElementById('current-layer').textContent = currentLayer;
            
            // Highlight active stage based on direction
            highlightStage('stage-attention');
            
            log(`Processing layer ${currentLayer}`, 'info');
        }
        
        function handleAttackStep(data) {
            const step = data.step || 0;
            const loss = data.loss || 0;
            
            document.getElementById('attack-step').textContent = step;
            document.getElementById('current-step').textContent = step;
            document.getElementById('attack-loss').textContent = loss.toFixed(4);
            
            if (loss < bestLoss) {
                bestLoss = loss;
                document.getElementById('best-loss').textContent = loss.toFixed(4);
            }
            
            if (data.suffix) {
                document.getElementById('suffix').textContent = data.suffix.substring(0, 30) + '...';
            }
            
            lossHistory.push(loss);
            if (lossHistory.length > 50) lossHistory.shift();
            drawLossChart();
            
            // Show processing indicator
            document.getElementById('processing').style.display = 'flex';
            document.getElementById('processing-text').textContent = `Attack Step ${step}: loss=${loss.toFixed(3)}`;
            
            log(`Step ${step}: loss=${loss.toFixed(4)}`, 'attack');
            
            // Animate flow
            animateFlow();
        }
        
        function handleEmbeddings(data) {
            tokens = data.tokens || [];
            
            // Update input tokens
            const inputContainer = document.getElementById('input-tokens');
            inputContainer.innerHTML = '';
            tokens.forEach(token => {
                const box = document.createElement('div');
                box.className = 'token-box';
                box.textContent = token.replace('Ġ', ' ').slice(0, 8);
                box.style.background = 'rgba(255, 255, 255, 0.1)';
                box.style.border = '1px solid #444';
                inputContainer.appendChild(box);
            });
            
            // Update embedding tokens
            const embContainer = document.getElementById('embedding-tokens');
            embContainer.innerHTML = '';
            tokens.forEach((token, i) => {
                const box = document.createElement('div');
                box.className = 'token-box embedding';
                box.textContent = `E${i}`;
                embContainer.appendChild(box);
            });
            
            log(`Received ${tokens.length} tokens`, 'info');
            highlightStage('stage-input');
            setTimeout(() => highlightStage('stage-embedding'), 500);
        }
        
        function handleTrace(data) {
            const trace = data.trace || data;
            if (trace && trace.layers) {
                totalLayers = trace.layers.length;
                document.getElementById('total-layers').textContent = totalLayers;
                
                // Create layer buttons
                const nav = document.getElementById('layer-nav');
                nav.innerHTML = '';
                for (let i = 0; i < totalLayers; i++) {
                    const btn = document.createElement('button');
                    btn.className = 'layer-btn';
                    btn.textContent = `L${i}`;
                    btn.onclick = () => selectLayer(i);
                    nav.appendChild(btn);
                }
                
                log(`Trace: ${totalLayers} layers loaded`, 'success');
            }
            
            if (trace && trace.tokens) {
                handleEmbeddings({tokens: trace.tokens});
            }
        }
        
        function handleAttention(data) {
            attentionData = data;
            document.getElementById('attn-layer').textContent = data.layer;
            document.getElementById('attn-head').textContent = data.head;
            
            if (data.weights && data.tokens) {
                drawAttentionMatrix(data.weights, data.tokens);
            }
            
            highlightStage('stage-attention');
        }
        
        function handleComplete(data) {
            document.getElementById('processing').style.display = 'none';
            log('Processing complete!', 'success');
            highlightStage('stage-output');
        }
        
        function selectLayer(layer) {
            currentLayer = layer;
            document.getElementById('current-layer').textContent = layer;
            document.querySelectorAll('.layer-btn').forEach((btn, i) => {
                btn.classList.toggle('active', i === layer);
            });
        }
        
        function highlightStage(stageId) {
            document.querySelectorAll('.stage').forEach(s => s.classList.remove('active'));
            const stage = document.getElementById(stageId);
            if (stage) stage.classList.add('active');
        }
        
        function animateFlow() {
            const stages = ['stage-input', 'stage-embedding', 'stage-qkv', 'stage-attention', 'stage-mlp', 'stage-output'];
            let i = 0;
            
            const interval = setInterval(() => {
                if (i >= stages.length) {
                    clearInterval(interval);
                    return;
                }
                highlightStage(stages[i]);
                i++;
            }, 200);
        }
        
        function drawLossChart() {
            if (lossHistory.length < 2) return;
            
            const svg = d3.select('#loss-chart');
            const container = svg.node().parentNode;
            const width = container.offsetWidth - 20;
            const height = 140;
            
            svg.selectAll('*').remove();
            svg.attr('width', width).attr('height', height);
            
            const margin = {top: 10, right: 10, bottom: 20, left: 35};
            const w = width - margin.left - margin.right;
            const h = height - margin.top - margin.bottom;
            
            const x = d3.scaleLinear()
                .domain([0, lossHistory.length - 1])
                .range([margin.left, w + margin.left]);
            
            const y = d3.scaleLinear()
                .domain([0, d3.max(lossHistory) * 1.1])
                .range([h + margin.top, margin.top]);
            
            // Gradient
            const gradient = svg.append('defs')
                .append('linearGradient')
                .attr('id', 'lossGradient')
                .attr('x1', '0%').attr('y1', '0%')
                .attr('x2', '0%').attr('y2', '100%');
            
            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', 'rgba(0, 212, 255, 0.5)');
            
            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', 'rgba(0, 212, 255, 0)');
            
            // Area
            const area = d3.area()
                .x((d, i) => x(i))
                .y0(h + margin.top)
                .y1(d => y(d));
            
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
                .call(d3.axisLeft(y).ticks(4).tickFormat(d => d.toFixed(1)))
                .attr('color', '#444');
        }
        
        function drawAttentionMatrix(weights, tokenLabels) {
            const svg = d3.select('#attention-matrix');
            const container = svg.node().parentNode;
            const width = container.offsetWidth - 30;
            const height = 180;
            
            svg.selectAll('*').remove();
            svg.attr('width', width).attr('height', height);
            
            const n = weights.length;
            const cellSize = Math.min(20, (width - 60) / n, (height - 40) / n);
            
            const margin = {top: 30, left: 50};
            
            // Color scale
            const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);
            
            // Draw cells
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    svg.append('rect')
                        .attr('x', margin.left + j * cellSize)
                        .attr('y', margin.top + i * cellSize)
                        .attr('width', cellSize - 1)
                        .attr('height', cellSize - 1)
                        .attr('fill', colorScale(weights[i][j]))
                        .attr('rx', 2)
                        .style('cursor', 'pointer')
                        .on('mouseover', function() {
                            d3.select(this).attr('stroke', '#fff').attr('stroke-width', 1);
                        })
                        .on('mouseout', function() {
                            d3.select(this).attr('stroke', 'none');
                        });
                }
            }
            
            // Labels
            tokenLabels.slice(0, n).forEach((token, i) => {
                svg.append('text')
                    .attr('x', margin.left - 5)
                    .attr('y', margin.top + i * cellSize + cellSize / 2 + 4)
                    .attr('text-anchor', 'end')
                    .attr('font-size', 9)
                    .attr('fill', '#888')
                    .text(token.slice(0, 4));
                
                svg.append('text')
                    .attr('x', margin.left + i * cellSize + cellSize / 2)
                    .attr('y', margin.top - 5)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', 9)
                    .attr('fill', '#888')
                    .text(token.slice(0, 4));
            });
        }
        
        // Initial log
        log('Waiting for data...', 'info');
    </script>
</body>
</html>
'''


def get_transformer_flow_html():
    """Return the transformer flow visualization HTML."""
    return TRANSFORMER_FLOW_HTML
