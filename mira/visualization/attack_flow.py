"""
Transformer Attack Flow Visualization

Full-page visualization inspired by transformer-explainer.
Shows the complete processing flow:
Input → Embedding → Layer Processing → Attention → MLP → Output

Focus on the FLOW, not metrics.
"""

ATTACK_FLOW_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA - Attack Flow</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --bg: #0a0a12;
            --surface: #12121a;
            --border: #2a2a3a;
            --text: #e8e8f0;
            --primary: #00d4ff;
            --attention: #ff6b9d;
            --mlp: #c471ed;
            --embedding: #4facfe;
            --output: #ffd93d;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 24px;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
        }
        
        .logo {
            font-size: 20px;
            font-weight: bold;
            color: var(--primary);
        }
        
        .status {
            display: flex;
            gap: 16px;
            align-items: center;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            padding: 4px 12px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 16px;
        }
        
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        /* Main Container */
        .main {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 50px);
            padding: 16px;
            gap: 16px;
        }
        
        /* Flow Section - MAIN FOCUS */
        .flow-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            min-height: 0;
        }
        
        .section-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #666;
            margin-bottom: 12px;
        }
        
        /* Transformer Flow */
        .flow-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            overflow-x: auto;
            padding: 20px 0;
        }
        
        .stage {
            display: flex;
            flex-direction: column;
            align-items: center;
            min-width: 120px;
            padding: 16px;
            background: rgba(0, 0, 0, 0.4);
            border: 2px solid var(--border);
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .stage:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }
        
        .stage.active {
            border-color: var(--primary);
            box-shadow: 0 0 24px rgba(0, 212, 255, 0.4);
        }
        
        .stage.input { border-color: #888; }
        .stage.embedding { border-color: var(--embedding); }
        .stage.attention { border-color: var(--attention); }
        .stage.mlp { border-color: var(--mlp); }
        .stage.output { border-color: var(--output); }
        
        .stage-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }
        
        .stage-label {
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        
        .stage-content {
            display: flex;
            flex-direction: column;
            gap: 4px;
            max-height: 200px;
            overflow-y: auto;
            width: 100%;
        }
        
        .token-item {
            font-size: 11px;
            padding: 4px 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .arrow {
            font-size: 28px;
            color: #444;
            flex-shrink: 0;
        }
        
        /* Layer Navigation */
        .layer-nav {
            display: flex;
            gap: 4px;
            margin-bottom: 12px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .layer-btn {
            padding: 6px 10px;
            font-size: 11px;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text);
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .layer-btn:hover, .layer-btn.active {
            background: rgba(0, 212, 255, 0.2);
            border-color: var(--primary);
        }
        
        /* Bottom Bar - Compact Info */
        .bottom-bar {
            display: flex;
            gap: 16px;
            padding: 12px 16px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        
        .info-box {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .info-label {
            font-size: 10px;
            text-transform: uppercase;
            color: #666;
        }
        
        .info-value {
            font-size: 14px;
            font-weight: bold;
            color: var(--primary);
        }
        
        /* Console - Compact */
        .console-box {
            flex: 2;
            background: #000;
            border-radius: 8px;
            padding: 8px 12px;
            max-height: 80px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 11px;
        }
        
        .log-line {
            color: #888;
            padding: 2px 0;
        }
        .log-line.info { color: var(--primary); }
        .log-line.atk { color: var(--output); }
        .log-line.ok { color: #00ff88; }
        
        /* Attention Visualization */
        .attention-box {
            position: fixed;
            bottom: 120px;
            right: 20px;
            width: 280px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
            display: none;
        }
        
        .attention-box.show { display: block; }
        
        #attention-canvas {
            width: 100%;
            height: 200px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🧠 MIRA Attack Flow</div>
        <div class="status">
            <div class="status-item">
                <div class="dot" id="status-dot"></div>
                <span id="status-text">Waiting...</span>
            </div>
            <div class="status-item">
                Layer <span id="layer-num">0</span>/<span id="total-layers">?</span>
            </div>
            <div class="status-item">
                Step <span id="step-num">0</span>
            </div>
        </div>
    </div>
    
    <div class="main">
        <div class="flow-section">
            <div class="section-title">Transformer Processing Flow</div>
            
            <div class="layer-nav" id="layer-nav">
                <!-- Layer buttons will be added dynamically -->
            </div>
            
            <div class="flow-container">
                <div class="stage input" id="stage-input">
                    <div class="stage-icon">📝</div>
                    <div class="stage-label">Input</div>
                    <div class="stage-content" id="input-content">
                        <div class="token-item">Waiting...</div>
                    </div>
                </div>
                
                <div class="arrow">→</div>
                
                <div class="stage embedding" id="stage-embedding">
                    <div class="stage-icon">🔢</div>
                    <div class="stage-label">Embedding</div>
                    <div class="stage-content" id="embedding-content">
                        <div class="token-item">E[0...n]</div>
                    </div>
                </div>
                
                <div class="arrow">→</div>
                
                <div class="stage attention" id="stage-attention">
                    <div class="stage-icon">👁️</div>
                    <div class="stage-label">Attention</div>
                    <div class="stage-content" id="attention-content">
                        <div class="token-item">Q × K^T × V</div>
                    </div>
                </div>
                
                <div class="arrow">→</div>
                
                <div class="stage mlp" id="stage-mlp">
                    <div class="stage-icon">⚡</div>
                    <div class="stage-label">MLP</div>
                    <div class="stage-content" id="mlp-content">
                        <div class="token-item">FFN</div>
                    </div>
                </div>
                
                <div class="arrow">→</div>
                
                <div class="stage output" id="stage-output">
                    <div class="stage-icon">🎯</div>
                    <div class="stage-label">Output</div>
                    <div class="stage-content" id="output-content">
                        <div class="token-item">Next Token</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="bottom-bar">
            <div class="info-box">
                <div class="info-label">Attack Step</div>
                <div class="info-value" id="attack-step">0</div>
            </div>
            <div class="info-box">
                <div class="info-label">Loss</div>
                <div class="info-value" id="attack-loss">--</div>
            </div>
            <div class="info-box">
                <div class="info-label">Suffix</div>
                <div class="info-value" id="suffix" style="font-size: 10px;">--</div>
            </div>
            <div class="console-box" id="console">
                <div class="log-line info">[Ready] Waiting for events...</div>
            </div>
        </div>
    </div>
    
    <div class="attention-box" id="attention-box">
        <div class="section-title">Attention Weights</div>
        <svg id="attention-canvas"></svg>
    </div>

    <script>
        // State
        let tokens = [];
        let currentLayer = 0;
        let totalLayers = 0;
        let eventCount = 0;
        
        // SSE Connection
        const evtSource = new EventSource('/api/events');
        
        function log(msg, type = '') {
            const c = document.getElementById('console');
            const line = document.createElement('div');
            line.className = 'log-line ' + type;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            c.appendChild(line);
            c.scrollTop = c.scrollHeight;
            if (c.children.length > 50) c.removeChild(c.firstChild);
        }
        
        evtSource.onopen = function() {
            document.getElementById('status-text').textContent = 'Connected';
            log('Connected to server', 'ok');
        };
        
        evtSource.onerror = function() {
            document.getElementById('status-text').textContent = 'Disconnected';
            document.getElementById('status-dot').style.background = '#ff4466';
        };
        
        evtSource.onmessage = function(e) {
            try {
                const data = JSON.parse(e.data);
                if (data.event_type === 'ping') return;
                
                eventCount++;
                processEvent(data);
            } catch(err) {
                console.error(err);
            }
        };
        
        function processEvent(data) {
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
                    log('✓ Pipeline complete!', 'ok');
                    break;
                default:
                    log(`Event: ${data.event_type}`, 'info');
            }
        }
        
        function handleLayer(data) {
            currentLayer = data.layer || 0;
            document.getElementById('layer-num').textContent = currentLayer;
            
            // Animate stages
            animateStages();
            log(`Processing layer ${currentLayer}`, 'info');
        }
        
        function handleAttackStep(data) {
            const step = data.step || 0;
            const loss = data.loss || 0;
            
            document.getElementById('step-num').textContent = step;
            document.getElementById('attack-step').textContent = step;
            document.getElementById('attack-loss').textContent = loss.toFixed(4);
            
            if (data.suffix) {
                document.getElementById('suffix').textContent = data.suffix.slice(0, 40);
            }
            
            log(`Step ${step}: loss=${loss.toFixed(3)}`, 'atk');
            animateStages();
        }
        
        function handleEmbeddings(data) {
            tokens = data.tokens || [];
            
            // Update input
            const inputContent = document.getElementById('input-content');
            inputContent.innerHTML = '';
            tokens.slice(0, 8).forEach(t => {
                const el = document.createElement('div');
                el.className = 'token-item';
                el.textContent = t.replace('Ġ', ' ').slice(0, 10);
                inputContent.appendChild(el);
            });
            if (tokens.length > 8) {
                const el = document.createElement('div');
                el.className = 'token-item';
                el.textContent = `+${tokens.length - 8} more`;
                inputContent.appendChild(el);
            }
            
            // Update embedding
            const embContent = document.getElementById('embedding-content');
            embContent.innerHTML = '';
            tokens.slice(0, 6).forEach((t, i) => {
                const el = document.createElement('div');
                el.className = 'token-item';
                el.textContent = `E[${i}]`;
                el.style.background = `rgba(79, 172, 254, ${0.3 + i * 0.1})`;
                embContent.appendChild(el);
            });
            
            log(`Received ${tokens.length} tokens`, 'info');
            highlightStage('stage-input');
            setTimeout(() => highlightStage('stage-embedding'), 300);
        }
        
        function handleTrace(data) {
            const trace = data.trace || data;
            if (trace && trace.layers) {
                totalLayers = trace.layers.length;
                document.getElementById('total-layers').textContent = totalLayers;
                createLayerNav();
            }
            if (trace && trace.tokens) {
                handleEmbeddings({tokens: trace.tokens});
            }
            log(`Trace: ${totalLayers} layers`, 'ok');
        }
        
        function handleAttention(data) {
            document.getElementById('attention-box').classList.add('show');
            if (data.weights) {
                drawAttention(data.weights);
            }
            highlightStage('stage-attention');
        }
        
        function createLayerNav() {
            const nav = document.getElementById('layer-nav');
            nav.innerHTML = '';
            for (let i = 0; i < totalLayers; i++) {
                const btn = document.createElement('button');
                btn.className = 'layer-btn';
                btn.textContent = `L${i}`;
                btn.onclick = () => selectLayer(i);
                nav.appendChild(btn);
            }
        }
        
        function selectLayer(i) {
            currentLayer = i;
            document.getElementById('layer-num').textContent = i;
            document.querySelectorAll('.layer-btn').forEach((b, idx) => {
                b.classList.toggle('active', idx === i);
            });
        }
        
        function highlightStage(id) {
            document.querySelectorAll('.stage').forEach(s => s.classList.remove('active'));
            const el = document.getElementById(id);
            if (el) el.classList.add('active');
        }
        
        function animateStages() {
            const stages = ['stage-input', 'stage-embedding', 'stage-attention', 'stage-mlp', 'stage-output'];
            let i = 0;
            const interval = setInterval(() => {
                if (i >= stages.length) {
                    clearInterval(interval);
                    return;
                }
                highlightStage(stages[i]);
                i++;
            }, 150);
        }
        
        function drawAttention(weights) {
            const svg = d3.select('#attention-canvas');
            const w = 250, h = 180;
            svg.attr('width', w).attr('height', h);
            svg.selectAll('*').remove();
            
            const n = Math.min(weights.length, 8);
            const cellSize = Math.min(20, (w - 40) / n);
            
            const color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);
            
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    svg.append('rect')
                        .attr('x', 30 + j * cellSize)
                        .attr('y', 20 + i * cellSize)
                        .attr('width', cellSize - 1)
                        .attr('height', cellSize - 1)
                        .attr('fill', color(weights[i][j] || 0))
                        .attr('rx', 2);
                }
            }
        }
        
        // Initialize
        log('Dashboard ready', 'ok');
        animateStages();
    </script>
</body>
</html>
'''


def get_attack_flow_html():
    """Return the attack flow visualization HTML."""
    return ATTACK_FLOW_HTML
