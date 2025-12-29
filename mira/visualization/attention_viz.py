"""
Rich Attention Visualization Dashboard

Features:
1. Interactive Attention Heatmap
2. Token-to-Token Connection Lines
3. Layer/Head Selectors
4. Real-time Updates During Attacks
"""

ATTENTION_VIZ_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA - Attention Visualizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --primary: #00f5ff;
            --secondary: #ff00ff;
            --accent: #ffff00;
            --bg-dark: #0a0a0f;
            --panel-bg: rgba(15, 25, 35, 0.8);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Space Mono', monospace;
            background: var(--bg-dark);
            color: #ddd;
            min-height: 100vh;
        }
        
        /* Animated background */
        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background: 
                radial-gradient(ellipse at 20% 30%, rgba(0, 245, 255, 0.12) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 70%, rgba(255, 0, 255, 0.08) 0%, transparent 50%);
            animation: pulse 10s ease-in-out infinite alternate;
            z-index: -1;
        }
        
        @keyframes pulse {
            0% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        /* Header */
        .header {
            background: rgba(0, 0, 0, 0.9);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid rgba(0, 245, 255, 0.3);
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .logo-icon {
            font-size: 32px;
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
        
        .logo h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5em;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .controls {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .control-group label {
            font-size: 0.9em;
            color: #888;
        }
        
        .control-group select {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid var(--primary);
            color: var(--primary);
            padding: 8px 15px;
            border-radius: 6px;
            font-family: inherit;
            cursor: pointer;
        }
        
        /* Main Container */
        .main {
            display: grid;
            grid-template-columns: 1fr 1.5fr;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        /* Panel */
        .panel {
            background: var(--panel-bg);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        .panel-title {
            font-family: 'Orbitron', sans-serif;
            color: var(--primary);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Tokens */
        .tokens-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }
        
        .token {
            padding: 8px 12px;
            background: rgba(0, 245, 255, 0.1);
            border: 1px solid rgba(0, 245, 255, 0.3);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .token:hover, .token.active {
            background: rgba(0, 245, 255, 0.3);
            border-color: var(--primary);
        }
        
        .token.source {
            background: rgba(0, 100, 255, 0.3);
            border-color: #0066ff;
        }
        
        .token.target {
            background: rgba(255, 0, 100, 0.3);
            border-color: #ff0066;
        }
        
        /* Attention Heatmap */
        .heatmap-container {
            position: relative;
            overflow: auto;
        }
        
        #heatmap-svg {
            display: block;
        }
        
        .heatmap-cell {
            cursor: pointer;
            transition: opacity 0.2s;
        }
        
        .heatmap-cell:hover {
            stroke: #fff;
            stroke-width: 2px;
        }
        
        .axis-label {
            font-size: 11px;
            fill: #888;
        }
        
        .axis-label.source { fill: #0088ff; }
        .axis-label.target { fill: #ff0088; }
        
        /* Connection Lines */
        .connections-container {
            position: relative;
            height: 200px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            margin-top: 15px;
        }
        
        #connections-svg {
            width: 100%;
            height: 100%;
        }
        
        .connection-line {
            fill: none;
            stroke-linecap: round;
        }
        
        .token-node {
            cursor: pointer;
        }
        
        .token-node text {
            font-size: 12px;
            fill: #ddd;
        }
        
        /* Legend */
        .legend {
            display: flex;
            gap: 20px;
            margin-top: 15px;
            font-size: 0.85em;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 12px;
            border-radius: 2px;
        }
        
        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: rgba(0, 0, 0, 0.4);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(0, 255, 255, 0.1);
        }
        
        .stat-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5em;
            color: var(--primary);
        }
        
        .stat-label {
            font-size: 0.75em;
            color: #666;
            margin-top: 5px;
        }
        
        /* Info Box */
        .info-box {
            background: rgba(0, 0, 0, 0.5);
            padding: 15px;
            border-radius: 8px;
            border-left: 3px solid var(--primary);
            font-size: 0.9em;
            line-height: 1.6;
        }
        
        /* Console */
        .console {
            background: #000;
            padding: 15px;
            border-radius: 8px;
            height: 150px;
            overflow-y: auto;
            font-size: 0.85em;
        }
        
        .console-line {
            margin-bottom: 5px;
            padding-left: 10px;
            border-left: 2px solid var(--primary);
        }
        
        .console-line.success { border-color: #00ff88; color: #00ff88; }
        .console-line.warn { border-color: #ffaa00; color: #ffaa00; }
        
        /* Status */
        .status-bar {
            background: rgba(0, 0, 0, 0.9);
            padding: 10px 30px;
            display: flex;
            justify-content: space-between;
            border-top: 1px solid rgba(0, 255, 255, 0.2);
        }
        
        .pulse-dot {
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulseDot 1.5s infinite;
            display: inline-block;
            margin-right: 10px;
        }
        
        @keyframes pulseDot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <span class="logo-icon">🧠</span>
            <h1>MIRA ATTENTION VISUALIZER</h1>
        </div>
        <div class="controls">
            <div class="control-group">
                <label>Layer:</label>
                <select id="layer-select"></select>
            </div>
            <div class="control-group">
                <label>Head:</label>
                <select id="head-select"></select>
            </div>
            <div class="control-group">
                <label>Mode:</label>
                <select id="mode-select">
                    <option value="heatmap">Heatmap</option>
                    <option value="connections">Connections</option>
                    <option value="both">Both</option>
                </select>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="left-column">
            <div class="panel">
                <div class="panel-title">📊 Attention Stats</div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="num-layers">0</div>
                        <div class="stat-label">Layers</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="num-heads">0</div>
                        <div class="stat-label">Heads</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="seq-len">0</div>
                        <div class="stat-label">Tokens</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="max-attn">0.00</div>
                        <div class="stat-label">Max Attn</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">🔤 Tokens</div>
                <div class="tokens-row" id="tokens-container"></div>
            </div>
            
            <div class="panel">
                <div class="panel-title">ℹ️ Attention Analysis</div>
                <div class="info-box" id="analysis-box">
                    Select a cell in the heatmap or hover over tokens to see attention details.
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">📋 Console</div>
                <div class="console" id="console"></div>
            </div>
        </div>
        
        <div class="right-column">
            <div class="panel">
                <div class="panel-title">🔥 Attention Heatmap</div>
                <div class="heatmap-container">
                    <svg id="heatmap-svg"></svg>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #1a1a2e;"></div>
                        <span>Low (0.0)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4a0080;"></div>
                        <span>Medium</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ff0066;"></div>
                        <span>High (1.0)</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">🔗 Token Connections</div>
                <div class="connections-container">
                    <svg id="connections-svg"></svg>
                </div>
            </div>
        </div>
    </div>
    
    <div class="status-bar">
        <div>
            <span class="pulse-dot"></span>
            <span id="status">Waiting for attention data...</span>
        </div>
        <div style="color: #555;">MIRA Framework | Real-time Attention Analysis</div>
    </div>

    <script>
        // State
        let attentionData = null;
        let currentLayer = 0;
        let currentHead = 0;
        let numLayers = 0;
        let numHeads = 0;
        
        // Color scale
        const colorScale = d3.scaleSequential(d3.interpolateViridis);
        
        // Connect to SSE
        const eventSource = new EventSource('/api/events');
        
        function log(msg, type = '') {
            const console = document.getElementById('console');
            const line = document.createElement('div');
            line.className = 'console-line ' + type;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            console.appendChild(line);
            console.scrollTop = console.scrollHeight;
        }
        
        log('Visualizer initialized', 'success');
        
        eventSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            if (data.event_type === 'ping') return;
            
            switch(data.event_type) {
                case 'attention_matrix':
                    handleAttentionMatrix(data.data);
                    break;
                case 'transformer_trace':
                    handleTransformerTrace(data.data);
                    break;
                case 'embeddings':
                    handleEmbeddings(data.data);
                    break;
            }
        };
        
        function handleAttentionMatrix(data) {
            const {layer, head, weights, tokens} = data;
            
            log(`Received attention: Layer ${layer}, Head ${head}`, 'success');
            
            if (!attentionData) {
                attentionData = {
                    tokens: tokens,
                    layers: {}
                };
            }
            
            if (!attentionData.layers[layer]) {
                attentionData.layers[layer] = {};
            }
            
            attentionData.layers[layer][head] = weights;
            attentionData.tokens = tokens;
            
            // Update selectors
            updateSelectors();
            
            // Render
            renderHeatmap();
            renderConnections();
            updateStats();
            
            document.getElementById('status').textContent = 
                `Showing Layer ${currentLayer}, Head ${currentHead}`;
        }
        
        function handleTransformerTrace(data) {
            const trace = data.trace;
            if (!trace || !trace.layers) return;
            
            numLayers = trace.layers.length;
            
            // Extract all attention data
            attentionData = {
                tokens: trace.tokens || [],
                layers: {}
            };
            
            trace.layers.forEach((layer, idx) => {
                if (layer.attention_weights) {
                    attentionData.layers[idx] = {};
                    // attention_weights shape: [num_heads, seq, seq]
                    const attn = layer.attention_weights;
                    if (Array.isArray(attn) && attn.length > 0) {
                        numHeads = attn.length;
                        attn.forEach((headWeights, headIdx) => {
                            attentionData.layers[idx][headIdx] = headWeights;
                        });
                    }
                }
            });
            
            log(`Trace received: ${numLayers} layers, ${numHeads} heads`, 'success');
            
            updateSelectors();
            renderHeatmap();
            renderConnections();
            updateStats();
            renderTokens();
        }
        
        function handleEmbeddings(data) {
            if (data.tokens) {
                renderTokens(data.tokens);
            }
        }
        
        function updateSelectors() {
            const layerSelect = document.getElementById('layer-select');
            const headSelect = document.getElementById('head-select');
            
            // Get actual layer/head counts
            const layers = Object.keys(attentionData.layers).map(Number);
            numLayers = layers.length;
            
            if (numLayers > 0) {
                const firstLayer = attentionData.layers[layers[0]];
                numHeads = Object.keys(firstLayer).length;
            }
            
            // Update layer selector
            if (layerSelect.options.length !== numLayers) {
                layerSelect.innerHTML = '';
                for (let i = 0; i < numLayers; i++) {
                    const opt = document.createElement('option');
                    opt.value = i;
                    opt.textContent = `Layer ${i}`;
                    layerSelect.appendChild(opt);
                }
            }
            
            // Update head selector
            if (headSelect.options.length !== numHeads) {
                headSelect.innerHTML = '';
                for (let i = 0; i < numHeads; i++) {
                    const opt = document.createElement('option');
                    opt.value = i;
                    opt.textContent = `Head ${i}`;
                    headSelect.appendChild(opt);
                }
            }
            
            document.getElementById('num-layers').textContent = numLayers;
            document.getElementById('num-heads').textContent = numHeads;
        }
        
        function renderTokens(tokens) {
            const container = document.getElementById('tokens-container');
            container.innerHTML = '';
            
            const tokenList = tokens || (attentionData ? attentionData.tokens : []);
            
            tokenList.forEach((token, idx) => {
                const span = document.createElement('span');
                span.className = 'token';
                span.textContent = token.replace('Ġ', ' ').replace('▁', ' ');
                span.dataset.idx = idx;
                span.addEventListener('mouseenter', () => highlightToken(idx));
                span.addEventListener('mouseleave', () => clearHighlight());
                container.appendChild(span);
            });
            
            document.getElementById('seq-len').textContent = tokenList.length;
        }
        
        function renderHeatmap() {
            if (!attentionData || !attentionData.layers[currentLayer]) return;
            
            const weights = attentionData.layers[currentLayer][currentHead];
            if (!weights) return;
            
            const tokens = attentionData.tokens;
            const n = tokens.length;
            
            const svg = d3.select('#heatmap-svg');
            svg.selectAll('*').remove();
            
            const margin = {top: 60, right: 30, bottom: 30, left: 80};
            const cellSize = Math.min(30, 400 / n);
            const width = margin.left + n * cellSize + margin.right;
            const height = margin.top + n * cellSize + margin.bottom;
            
            svg.attr('width', width).attr('height', height);
            
            // Color scale
            const maxVal = d3.max(weights.flat());
            colorScale.domain([0, maxVal || 1]);
            
            document.getElementById('max-attn').textContent = maxVal ? maxVal.toFixed(3) : '0.000';
            
            // Draw cells
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const val = weights[i][j];
                    svg.append('rect')
                        .attr('class', 'heatmap-cell')
                        .attr('x', margin.left + j * cellSize)
                        .attr('y', margin.top + i * cellSize)
                        .attr('width', cellSize - 1)
                        .attr('height', cellSize - 1)
                        .attr('fill', colorScale(val))
                        .attr('rx', 2)
                        .on('mouseover', function() {
                            showAttnInfo(i, j, val);
                        })
                        .on('click', function() {
                            selectCell(i, j, val);
                        });
                }
            }
            
            // Row labels (source tokens)
            tokens.forEach((token, i) => {
                svg.append('text')
                    .attr('class', 'axis-label source')
                    .attr('x', margin.left - 5)
                    .attr('y', margin.top + i * cellSize + cellSize / 2 + 4)
                    .attr('text-anchor', 'end')
                    .text(token.slice(0, 6));
            });
            
            // Column labels (target tokens)
            tokens.forEach((token, i) => {
                svg.append('text')
                    .attr('class', 'axis-label target')
                    .attr('x', margin.left + i * cellSize + cellSize / 2)
                    .attr('y', margin.top - 10)
                    .attr('text-anchor', 'middle')
                    .attr('transform', `rotate(-45, ${margin.left + i * cellSize + cellSize / 2}, ${margin.top - 10})`)
                    .text(token.slice(0, 6));
            });
            
            // Title
            svg.append('text')
                .attr('x', width / 2)
                .attr('y', 20)
                .attr('text-anchor', 'middle')
                .attr('fill', '#00f5ff')
                .attr('font-family', 'Orbitron')
                .text(`Layer ${currentLayer}, Head ${currentHead}`);
        }
        
        function renderConnections() {
            if (!attentionData || !attentionData.layers[currentLayer]) return;
            
            const weights = attentionData.layers[currentLayer][currentHead];
            if (!weights) return;
            
            const tokens = attentionData.tokens;
            const n = tokens.length;
            
            const svg = d3.select('#connections-svg');
            const container = svg.node().parentNode;
            const width = container.offsetWidth;
            const height = container.offsetHeight;
            
            svg.selectAll('*').remove();
            svg.attr('width', width).attr('height', height);
            
            const margin = 40;
            const tokenSpacing = (width - 2 * margin) / (n - 1 || 1);
            
            // Top row (source) and bottom row (target)
            const yTop = 40;
            const yBottom = height - 30;
            
            // Draw tokens - top row
            tokens.forEach((token, i) => {
                const x = margin + i * tokenSpacing;
                
                svg.append('text')
                    .attr('x', x)
                    .attr('y', yTop)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#0088ff')
                    .attr('font-size', '11px')
                    .text(token.slice(0, 5));
                    
                svg.append('circle')
                    .attr('cx', x)
                    .attr('cy', yTop + 15)
                    .attr('r', 4)
                    .attr('fill', '#0088ff');
            });
            
            // Draw tokens - bottom row
            tokens.forEach((token, i) => {
                const x = margin + i * tokenSpacing;
                
                svg.append('circle')
                    .attr('cx', x)
                    .attr('cy', yBottom - 10)
                    .attr('r', 4)
                    .attr('fill', '#ff0088');
                    
                svg.append('text')
                    .attr('x', x)
                    .attr('y', yBottom + 5)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#ff0088')
                    .attr('font-size', '11px')
                    .text(token.slice(0, 5));
            });
            
            // Draw connection lines
            const threshold = 0.1; // Only show significant attention
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const val = weights[i][j];
                    if (val < threshold) continue;
                    
                    const x1 = margin + i * tokenSpacing;
                    const x2 = margin + j * tokenSpacing;
                    
                    const gradient = svg.append('defs')
                        .append('linearGradient')
                        .attr('id', `grad-${i}-${j}`)
                        .attr('x1', '0%')
                        .attr('y1', '0%')
                        .attr('x2', '0%')
                        .attr('y2', '100%');
                    
                    gradient.append('stop')
                        .attr('offset', '0%')
                        .attr('stop-color', '#0088ff')
                        .attr('stop-opacity', val);
                    
                    gradient.append('stop')
                        .attr('offset', '100%')
                        .attr('stop-color', '#ff0088')
                        .attr('stop-opacity', val);
                    
                    svg.append('line')
                        .attr('class', 'connection-line')
                        .attr('x1', x1)
                        .attr('y1', yTop + 20)
                        .attr('x2', x2)
                        .attr('y2', yBottom - 15)
                        .attr('stroke', `url(#grad-${i}-${j})`)
                        .attr('stroke-width', Math.max(1, val * 5))
                        .attr('opacity', Math.max(0.2, val));
                }
            }
        }
        
        function showAttnInfo(i, j, val) {
            const tokens = attentionData.tokens;
            const box = document.getElementById('analysis-box');
            box.innerHTML = `
                <strong>Attention Value:</strong> ${val.toFixed(4)}<br>
                <strong>From:</strong> "${tokens[i]}" (position ${i})<br>
                <strong>To:</strong> "${tokens[j]}" (position ${j})<br>
                <strong>Layer:</strong> ${currentLayer}, <strong>Head:</strong> ${currentHead}
            `;
        }
        
        function selectCell(i, j, val) {
            // Highlight corresponding tokens
            const tokenEls = document.querySelectorAll('.token');
            tokenEls.forEach(el => el.classList.remove('source', 'target'));
            tokenEls[i]?.classList.add('source');
            tokenEls[j]?.classList.add('target');
        }
        
        function highlightToken(idx) {
            // Could highlight row/column in heatmap
        }
        
        function clearHighlight() {
            // Clear highlights
        }
        
        function updateStats() {
            if (!attentionData) return;
            document.getElementById('seq-len').textContent = attentionData.tokens.length;
        }
        
        // Event listeners
        document.getElementById('layer-select').addEventListener('change', function(e) {
            currentLayer = parseInt(e.target.value);
            renderHeatmap();
            renderConnections();
        });
        
        document.getElementById('head-select').addEventListener('change', function(e) {
            currentHead = parseInt(e.target.value);
            renderHeatmap();
            renderConnections();
        });
        
        // Handle resize
        window.addEventListener('resize', function() {
            renderHeatmap();
            renderConnections();
        });
    </script>
</body>
</html>
'''


def get_attention_viz_html():
    """Return the attention visualization HTML."""
    return ATTENTION_VIZ_HTML
