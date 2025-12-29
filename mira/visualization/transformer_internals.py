"""
Transformer Internals Visualization.

Real-time visualization of Transformer processing during attack execution.
Shows Q/K/V flow, attention matrices, MLP activations, and before/after comparison.
"""


def get_transformer_internals_html():
    """
    Returns the HTML for the transformer internals visualization.
    
    Features:
    - Sankey-style flow diagram: Input -> Embedding -> Q/K/V -> Attention -> MLP -> Output
    - Attention matrix heatmap with head selection
    - MLP activation visualization
    - Toggle between before/after attack states
    - Real-time SSE updates during attack execution
    """
    return TRANSFORMER_INTERNALS_HTML


TRANSFORMER_INTERNALS_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA - Transformer Attack Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --bg: #0f0f1a;
            --surface: #1a1a2e;
            --surface-2: #16213e;
            --border: #2a2a4a;
            --text: #e8e8f0;
            --text-muted: #888;
            --primary: #00d4ff;
            --query: #60a5fa;
            --key: #f87171;
            --value: #4ade80;
            --attention: #c084fc;
            --mlp: #fb923c;
            --output: #fbbf24;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
            font-size: 18px;
            font-weight: 600;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .status-bar {
            display: flex;
            gap: 16px;
            align-items: center;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            padding: 6px 12px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 20px;
            border: 1px solid rgba(0, 212, 255, 0.2);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        .toggle-btn {
            padding: 6px 14px;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        
        .toggle-btn:hover, .toggle-btn.active {
            background: rgba(0, 212, 255, 0.2);
            border-color: var(--primary);
        }
        
        /* Main Layout */
        .main-container {
            display: grid;
            grid-template-columns: 1fr;
            grid-template-rows: 1fr auto;
            height: calc(100vh - 52px);
            padding: 16px;
            gap: 16px;
        }
        
        /* Transformer Flow Section */
        .flow-section {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .section-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-muted);
        }
        
        .layer-selector {
            display: flex;
            gap: 4px;
        }
        
        .layer-btn {
            padding: 4px 10px;
            font-size: 11px;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .layer-btn:hover, .layer-btn.active {
            background: rgba(0, 212, 255, 0.15);
            border-color: var(--primary);
            color: var(--primary);
        }
        
        .head-selector {
            display: flex;
            gap: 4px;
            margin-left: 16px;
        }
        
        .head-btn {
            padding: 4px 8px;
            font-size: 10px;
            background: rgba(192, 132, 252, 0.1);
            border: 1px solid rgba(192, 132, 252, 0.3);
            border-radius: 4px;
            color: var(--attention);
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .head-btn:hover, .head-btn.active {
            background: rgba(192, 132, 252, 0.3);
            border-color: var(--attention);
        }
        
        /* Flow Container */
        .flow-container {
            flex: 1;
            display: flex;
            align-items: stretch;
            gap: 8px;
            overflow-x: auto;
            padding: 8px 0;
        }
        
        .flow-arrow {
            display: flex;
            align-items: center;
            color: var(--border);
            font-size: 20px;
            padding: 0 4px;
        }
        
        /* Stage Panels */
        .stage {
            display: flex;
            flex-direction: column;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            min-width: 120px;
            transition: all 0.3s;
        }
        
        .stage.active {
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
        }
        
        .stage-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 10px;
        }
        
        .stage-icon {
            font-size: 18px;
        }
        
        .stage-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stage.input .stage-label { color: var(--text-muted); }
        .stage.embedding .stage-label { color: var(--primary); }
        .stage.qkv .stage-label { color: var(--query); }
        .stage.attention .stage-label { color: var(--attention); }
        .stage.mlp .stage-label { color: var(--mlp); }
        .stage.output .stage-label { color: var(--output); }
        
        .stage-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        /* Token items */
        .token-item {
            padding: 4px 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            font-size: 11px;
            font-family: monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .token-item.highlight {
            background: rgba(0, 212, 255, 0.2);
            border: 1px solid rgba(0, 212, 255, 0.4);
        }
        
        /* QKV Panel */
        .qkv-columns {
            display: flex;
            gap: 8px;
        }
        
        .qkv-column {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .qkv-column-header {
            font-size: 10px;
            font-weight: 600;
            text-align: center;
            padding: 4px;
            border-radius: 4px;
        }
        
        .qkv-column.query .qkv-column-header { 
            background: rgba(96, 165, 250, 0.2);
            color: var(--query);
        }
        .qkv-column.key .qkv-column-header { 
            background: rgba(248, 113, 113, 0.2);
            color: var(--key);
        }
        .qkv-column.value .qkv-column-header { 
            background: rgba(74, 222, 128, 0.2);
            color: var(--value);
        }
        
        .vector-bar {
            height: 16px;
            border-radius: 3px;
            position: relative;
            overflow: hidden;
        }
        
        .vector-bar.query { background: linear-gradient(90deg, var(--query), rgba(96, 165, 250, 0.3)); }
        .vector-bar.key { background: linear-gradient(90deg, var(--key), rgba(248, 113, 113, 0.3)); }
        .vector-bar.value { background: linear-gradient(90deg, var(--value), rgba(74, 222, 128, 0.3)); }
        
        /* Attention Matrix */
        .attention-panel {
            min-width: 280px;
            flex: 2;
        }
        
        .attention-matrix-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        #attention-svg {
            max-width: 100%;
            max-height: 250px;
        }
        
        .matrix-label {
            font-size: 10px;
            color: var(--text-muted);
            margin-top: 8px;
        }
        
        /* MLP Panel */
        .mlp-bars {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .mlp-bar-row {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .mlp-bar-label {
            font-size: 9px;
            color: var(--text-muted);
            width: 24px;
            text-align: right;
        }
        
        .mlp-bar {
            flex: 1;
            height: 12px;
            background: rgba(251, 146, 60, 0.2);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .mlp-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--mlp), rgba(251, 146, 60, 0.5));
            border-radius: 3px;
            transition: width 0.3s;
        }
        
        /* Output Panel */
        .prob-bars {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        
        .prob-bar-row {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .prob-token {
            font-size: 11px;
            font-family: monospace;
            width: 50px;
            text-align: right;
            color: var(--output);
        }
        
        .prob-bar {
            flex: 1;
            height: 14px;
            background: rgba(251, 191, 36, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .prob-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--output), rgba(251, 191, 36, 0.4));
            border-radius: 3px;
            transition: width 0.3s;
        }
        
        .prob-value {
            font-size: 10px;
            color: var(--text-muted);
            width: 40px;
        }
        
        /* Bottom Info Bar */
        .info-bar {
            display: flex;
            gap: 16px;
            padding: 12px 16px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        
        .info-item {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .info-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        
        .info-value {
            font-size: 14px;
            font-weight: 600;
            color: var(--primary);
        }
        
        .info-value.success { color: #4ade80; }
        .info-value.error { color: #f87171; }
        
        .console-box {
            flex: 1;
            background: #000;
            border-radius: 6px;
            padding: 8px 12px;
            font-family: monospace;
            font-size: 11px;
            max-height: 60px;
            overflow-y: auto;
        }
        
        .log-line {
            color: #666;
            padding: 1px 0;
        }
        .log-line.info { color: var(--primary); }
        .log-line.success { color: #4ade80; }
        .log-line.attack { color: var(--output); }
        
        /* Comparison Toggle */
        .comparison-toggle {
            display: flex;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 6px;
            padding: 2px;
        }
        
        .comparison-toggle button {
            padding: 6px 14px;
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            border-radius: 4px;
            font-size: 11px;
            transition: all 0.2s;
        }
        
        .comparison-toggle button.active {
            background: rgba(0, 212, 255, 0.2);
            color: var(--primary);
        }
        
        /* Animations */
        @keyframes flowPulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
        
        .processing .stage-icon {
            animation: flowPulse 0.5s ease-in-out;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <span>🧠</span>
            <span>MIRA Transformer Visualization</span>
        </div>
        <div class="status-bar">
            <div class="comparison-toggle">
                <button id="btn-before" class="active">Before Attack</button>
                <button id="btn-after">After Attack</button>
            </div>
            <div class="status-item">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">Connecting...</span>
            </div>
            <div class="status-item">
                Layer <span id="current-layer">0</span>/<span id="total-layers">?</span>
            </div>
            <div class="status-item">
                Head <span id="current-head">1</span>/<span id="total-heads">?</span>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="flow-section">
            <div class="section-header">
                <div class="section-title">Transformer Processing Flow</div>
                <div style="display: flex; align-items: center;">
                    <div class="layer-selector" id="layer-selector">
                        <!-- Layer buttons added dynamically -->
                    </div>
                    <div class="head-selector" id="head-selector">
                        <!-- Head buttons added dynamically -->
                    </div>
                </div>
            </div>
            
            <div class="flow-container">
                <!-- Input Stage -->
                <div class="stage input" id="stage-input">
                    <div class="stage-header">
                        <span class="stage-icon">📝</span>
                        <span class="stage-label">Input</span>
                    </div>
                    <div class="stage-content" id="input-tokens">
                        <div class="token-item">Waiting...</div>
                    </div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- Embedding Stage -->
                <div class="stage embedding" id="stage-embedding">
                    <div class="stage-header">
                        <span class="stage-icon">🔢</span>
                        <span class="stage-label">Embedding</span>
                    </div>
                    <div class="stage-content" id="embedding-vectors">
                        <div class="token-item">E[0..n]</div>
                    </div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- Q/K/V Stage -->
                <div class="stage qkv" id="stage-qkv">
                    <div class="stage-header">
                        <span class="stage-icon">🔀</span>
                        <span class="stage-label">Q / K / V</span>
                    </div>
                    <div class="stage-content">
                        <div class="qkv-columns">
                            <div class="qkv-column query">
                                <div class="qkv-column-header">Query</div>
                                <div id="query-vectors"></div>
                            </div>
                            <div class="qkv-column key">
                                <div class="qkv-column-header">Key</div>
                                <div id="key-vectors"></div>
                            </div>
                            <div class="qkv-column value">
                                <div class="qkv-column-header">Value</div>
                                <div id="value-vectors"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- Attention Stage -->
                <div class="stage attention attention-panel" id="stage-attention">
                    <div class="stage-header">
                        <span class="stage-icon">👁️</span>
                        <span class="stage-label">Attention Matrix</span>
                    </div>
                    <div class="attention-matrix-container">
                        <svg id="attention-svg"></svg>
                        <div class="matrix-label">Q × K<sup>T</sup> / √d<sub>k</sub></div>
                    </div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- MLP Stage -->
                <div class="stage mlp" id="stage-mlp">
                    <div class="stage-header">
                        <span class="stage-icon">⚡</span>
                        <span class="stage-label">MLP</span>
                    </div>
                    <div class="stage-content">
                        <div class="mlp-bars" id="mlp-activations">
                            <!-- MLP bars added dynamically -->
                        </div>
                    </div>
                </div>
                
                <div class="flow-arrow">→</div>
                
                <!-- Output Stage -->
                <div class="stage output" id="stage-output">
                    <div class="stage-header">
                        <span class="stage-icon">🎯</span>
                        <span class="stage-label">Output</span>
                    </div>
                    <div class="stage-content">
                        <div class="prob-bars" id="output-probs">
                            <!-- Probability bars added dynamically -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="info-bar">
            <div class="info-item">
                <div class="info-label">Attack Step</div>
                <div class="info-value" id="attack-step">0</div>
            </div>
            <div class="info-item">
                <div class="info-label">Loss</div>
                <div class="info-value" id="attack-loss">--</div>
            </div>
            <div class="info-item">
                <div class="info-label">Status</div>
                <div class="info-value" id="attack-status">Waiting</div>
            </div>
            <div class="info-item">
                <div class="info-label">Suffix</div>
                <div class="info-value" id="attack-suffix" style="font-size: 10px; max-width: 200px; overflow: hidden; text-overflow: ellipsis;">--</div>
            </div>
            <div class="console-box" id="console">
                <div class="log-line info">[Ready] Waiting for events...</div>
            </div>
        </div>
    </div>

    <script>
        // State
        let state = {
            tokens: [],
            currentLayer: 0,
            currentHead: 0,
            totalLayers: 6,
            totalHeads: 8,
            viewMode: 'before', // 'before' or 'after'
            beforeState: null,
            afterState: null
        };
        
        // SSE Connection
        const evtSource = new EventSource('/api/events');
        
        evtSource.onopen = function() {
            document.getElementById('status-text').textContent = 'Connected';
            document.getElementById('status-dot').style.background = '#4ade80';
            log('Connected to server', 'success');
        };
        
        evtSource.onerror = function() {
            document.getElementById('status-text').textContent = 'Disconnected';
            document.getElementById('status-dot').style.background = '#f87171';
        };
        
        evtSource.onmessage = function(e) {
            try {
                const data = JSON.parse(e.data);
                if (data.event_type === 'ping') return;
                processEvent(data);
            } catch(err) {
                console.error('Event parse error:', err);
            }
        };
        
        function processEvent(data) {
            switch(data.event_type) {
                case 'embeddings':
                    handleEmbeddings(data.data);
                    break;
                case 'layer':
                    handleLayer(data.data);
                    break;
                case 'attention_matrix':
                    handleAttentionMatrix(data.data);
                    break;
                case 'qkv':
                    handleQKV(data.data);
                    break;
                case 'mlp':
                    handleMLP(data.data);
                    break;
                case 'attack_step':
                    handleAttackStep(data.data);
                    break;
                case 'transformer_trace':
                    handleTrace(data.data);
                    break;
                case 'output_probs':
                    handleOutputProbs(data.data);
                    break;
                case 'residual_predictions':
                    handleResidualPredictions(data.data);
                    break;
                case 'complete':
                    handleComplete(data.data);
                    break;
                default:
                    log(`Event: ${data.event_type}`, 'info');
            }
        }
        
        function handleOutputProbs(data) {
            const probs = data.probs || [];
            updateOutputProbs(probs);
        }
        
        function handleResidualPredictions(data) {
            // Log layer predictions for theory-to-practice view
            const layer = data.layer || 0;
            const before = data.before_ffn || [];
            const after = data.after_ffn || [];
            
            if (before.length > 0 && after.length > 0) {
                const beforeTop = before[0]?.token || '?';
                const afterTop = after[0]?.token || '?';
                log(`L${layer}: ${beforeTop} → ${afterTop}`, 'info');
            }
            
            // Update output predictions with after_ffn data from the last layer
            if (after.length > 0) {
                updateOutputProbs(after);
            }
        }
        
        function handleEmbeddings(data) {
            state.tokens = data.tokens || [];
            
            // Update input tokens
            const inputEl = document.getElementById('input-tokens');
            inputEl.innerHTML = '';
            state.tokens.slice(0, 10).forEach((t, i) => {
                const el = document.createElement('div');
                el.className = 'token-item';
                el.textContent = t.replace('Ġ', ' ').slice(0, 12);
                if (i === state.tokens.length - 1) el.classList.add('highlight');
                inputEl.appendChild(el);
            });
            if (state.tokens.length > 10) {
                const el = document.createElement('div');
                el.className = 'token-item';
                el.textContent = `+${state.tokens.length - 10} more`;
                inputEl.appendChild(el);
            }
            
            // Update embedding representation
            const embEl = document.getElementById('embedding-vectors');
            embEl.innerHTML = '';
            state.tokens.slice(0, 8).forEach((t, i) => {
                const el = document.createElement('div');
                el.className = 'token-item';
                el.style.background = `rgba(0, 212, 255, ${0.1 + i * 0.05})`;
                el.textContent = `E[${i}]`;
                embEl.appendChild(el);
            });
            
            highlightStage('stage-input');
            setTimeout(() => highlightStage('stage-embedding'), 200);
            log(`Received ${state.tokens.length} tokens`, 'info');
        }
        
        function handleLayer(data) {
            state.currentLayer = data.layer || 0;
            document.getElementById('current-layer').textContent = state.currentLayer;
            
            // Update layer selector
            updateLayerButtons();
            
            // Animate flow
            highlightStage('stage-qkv');
            setTimeout(() => highlightStage('stage-attention'), 150);
            setTimeout(() => highlightStage('stage-mlp'), 300);
        }
        
        function handleQKV(data) {
            const n = Math.min(6, (data.tokens || []).length);
            
            // Query vectors - only show if we have actual data
            const queryEl = document.getElementById('query-vectors');
            queryEl.innerHTML = '';
            const qData = data.query_vectors || [];
            for (let i = 0; i < n; i++) {
                const bar = document.createElement('div');
                bar.className = 'vector-bar query';
                // Use actual norm if available, otherwise don't show
                const norm = qData[i] ? Math.min(100, Math.abs(qData[i]) * 100) : 0;
                bar.style.width = norm > 0 ? `${norm}%` : '5%';
                queryEl.appendChild(bar);
            }
            
            // Key vectors
            const keyEl = document.getElementById('key-vectors');
            keyEl.innerHTML = '';
            const kData = data.key_vectors || [];
            for (let i = 0; i < n; i++) {
                const bar = document.createElement('div');
                bar.className = 'vector-bar key';
                const norm = kData[i] ? Math.min(100, Math.abs(kData[i]) * 100) : 0;
                bar.style.width = norm > 0 ? `${norm}%` : '5%';
                keyEl.appendChild(bar);
            }
            
            // Value vectors
            const valueEl = document.getElementById('value-vectors');
            valueEl.innerHTML = '';
            const vData = data.value_vectors || [];
            for (let i = 0; i < n; i++) {
                const bar = document.createElement('div');
                bar.className = 'vector-bar value';
                const norm = vData[i] ? Math.min(100, Math.abs(vData[i]) * 100) : 0;
                bar.style.width = norm > 0 ? `${norm}%` : '5%';
                valueEl.appendChild(bar);
            }
            
            highlightStage('stage-qkv');
        }
        
        function handleAttentionMatrix(data) {
            const weights = data.weights || [];
            const tokens = data.tokens || state.tokens;
            const n = Math.min(weights.length, 10);
            
            if (n === 0) return;
            
            const svg = d3.select('#attention-svg');
            svg.selectAll('*').remove();
            
            const size = Math.min(240, n * 24);
            const cellSize = size / n;
            const margin = 30;
            
            svg.attr('width', size + margin)
               .attr('height', size + margin);
            
            const color = d3.scaleSequential(d3.interpolatePurples).domain([0, 1]);
            
            // Draw cells
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const val = weights[i] && weights[i][j] !== undefined ? weights[i][j] : 0;
                    svg.append('rect')
                        .attr('x', margin + j * cellSize)
                        .attr('y', i * cellSize)
                        .attr('width', cellSize - 1)
                        .attr('height', cellSize - 1)
                        .attr('fill', color(val))
                        .attr('rx', 2)
                        .style('cursor', 'pointer')
                        .on('mouseover', function() {
                            d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2);
                        })
                        .on('mouseout', function() {
                            d3.select(this).attr('stroke', 'none');
                        });
                }
            }
            
            // Row labels (queries)
            const shortTokens = tokens.slice(0, n).map(t => 
                (t || '').replace('Ġ', '').slice(0, 4)
            );
            
            shortTokens.forEach((t, i) => {
                svg.append('text')
                    .attr('x', margin - 4)
                    .attr('y', i * cellSize + cellSize / 2 + 3)
                    .attr('text-anchor', 'end')
                    .attr('fill', '#888')
                    .attr('font-size', 9)
                    .text(t);
            });
            
            // Column labels (keys)
            shortTokens.forEach((t, i) => {
                svg.append('text')
                    .attr('x', margin + i * cellSize + cellSize / 2)
                    .attr('y', size + 12)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#888')
                    .attr('font-size', 9)
                    .text(t);
            });
            
            highlightStage('stage-attention');
            log(`Attention matrix L${data.layer} H${data.head}`, 'info');
        }
        
        function handleMLP(data) {
            const activations = data.activations || [];
            const topNeurons = data.top_neurons || [];
            
            const mlpEl = document.getElementById('mlp-activations');
            mlpEl.innerHTML = '';
            
            // Show top 8 neuron activations
            const n = Math.min(8, topNeurons.length || 8);
            for (let i = 0; i < n; i++) {
                const row = document.createElement('div');
                row.className = 'mlp-bar-row';
                
                const label = document.createElement('div');
                label.className = 'mlp-bar-label';
                label.textContent = topNeurons[i] !== undefined ? `#${topNeurons[i]}` : `n${i}`;
                
                const bar = document.createElement('div');
                bar.className = 'mlp-bar';
                
                const fill = document.createElement('div');
                fill.className = 'mlp-bar-fill';
                // Use actual activation value if available
                const activation = activations[i] || 0;
                fill.style.width = activation > 0 ? `${Math.min(100, activation * 100)}%` : '5%';
                
                bar.appendChild(fill);
                row.appendChild(label);
                row.appendChild(bar);
                mlpEl.appendChild(row);
            }
            
            highlightStage('stage-mlp');
        }
        
        function handleAttackStep(data) {
            document.getElementById('attack-step').textContent = data.step || 0;
            document.getElementById('attack-loss').textContent = 
                data.loss !== undefined ? data.loss.toFixed(4) : '--';
            
            if (data.suffix) {
                document.getElementById('attack-suffix').textContent = 
                    data.suffix.slice(0, 30) + (data.suffix.length > 30 ? '...' : '');
            }
            
            if (data.success) {
                document.getElementById('attack-status').textContent = 'Success!';
                document.getElementById('attack-status').className = 'info-value success';
            } else {
                document.getElementById('attack-status').textContent = 'Running...';
                document.getElementById('attack-status').className = 'info-value';
            }
            
            log(`Step ${data.step}: loss=${(data.loss || 0).toFixed(3)}`, 'attack');
            
            // Trigger flow animation
            animateFlow();
        }
        
        function handleTrace(data) {
            const trace = data.trace || {};
            
            if (trace.layers) {
                state.totalLayers = trace.layers.length;
                document.getElementById('total-layers').textContent = state.totalLayers;
                createLayerButtons();
            }
            
            if (trace.tokens) {
                handleEmbeddings({ tokens: trace.tokens });
            }
            
            // Store state for comparison
            if (data.trace_type === 'adversarial') {
                state.afterState = trace;
                log('Captured adversarial state', 'success');
            } else {
                state.beforeState = trace;
                log('Captured baseline state', 'info');
            }
        }
        
        function handleComplete(data) {
            document.getElementById('attack-status').textContent = 'Complete';
            document.getElementById('attack-status').className = 'info-value success';
            log(`Pipeline complete! ASR: ${((data.asr || 0) * 100).toFixed(1)}%`, 'success');
        }
        
        function updateOutputProbs(probs) {
            // Only update if we have real probability data
            if (!probs || probs.length === 0) return;
            
            const probsEl = document.getElementById('output-probs');
            probsEl.innerHTML = '';
            
            // Show top 5 predictions
            probs.slice(0, 5).forEach(p => {
                const row = document.createElement('div');
                row.className = 'prob-bar-row';
                
                const token = document.createElement('div');
                token.className = 'prob-token';
                token.textContent = p.token || p.text || '';
                
                const bar = document.createElement('div');
                bar.className = 'prob-bar';
                
                const fill = document.createElement('div');
                fill.className = 'prob-bar-fill';
                const prob = p.prob || p.probability || 0;
                fill.style.width = `${prob * 100}%`;
                
                const value = document.createElement('div');
                value.className = 'prob-value';
                value.textContent = (prob * 100).toFixed(1) + '%';
                
                bar.appendChild(fill);
                row.appendChild(token);
                row.appendChild(bar);
                row.appendChild(value);
                probsEl.appendChild(row);
            });
            
            highlightStage('stage-output');
        }
        
        function createLayerButtons() {
            const container = document.getElementById('layer-selector');
            container.innerHTML = '';
            
            for (let i = 0; i < state.totalLayers; i++) {
                const btn = document.createElement('button');
                btn.className = 'layer-btn' + (i === state.currentLayer ? ' active' : '');
                btn.textContent = `L${i}`;
                btn.onclick = () => selectLayer(i);
                container.appendChild(btn);
            }
        }
        
        function createHeadButtons() {
            const container = document.getElementById('head-selector');
            container.innerHTML = '';
            
            for (let i = 0; i < state.totalHeads; i++) {
                const btn = document.createElement('button');
                btn.className = 'head-btn' + (i === state.currentHead ? ' active' : '');
                btn.textContent = `H${i + 1}`;
                btn.onclick = () => selectHead(i);
                container.appendChild(btn);
            }
        }
        
        function selectLayer(idx) {
            state.currentLayer = idx;
            document.getElementById('current-layer').textContent = idx;
            updateLayerButtons();
            log(`Selected layer ${idx}`, 'info');
        }
        
        function selectHead(idx) {
            state.currentHead = idx;
            document.getElementById('current-head').textContent = idx + 1;
            document.querySelectorAll('.head-btn').forEach((b, i) => {
                b.classList.toggle('active', i === idx);
            });
            log(`Selected head ${idx + 1}`, 'info');
        }
        
        function updateLayerButtons() {
            document.querySelectorAll('.layer-btn').forEach((b, i) => {
                b.classList.toggle('active', i === state.currentLayer);
            });
        }
        
        function highlightStage(id) {
            document.querySelectorAll('.stage').forEach(s => s.classList.remove('active'));
            const el = document.getElementById(id);
            if (el) el.classList.add('active');
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
            }, 100);
        }
        
        function log(msg, type = '') {
            const c = document.getElementById('console');
            const line = document.createElement('div');
            line.className = 'log-line ' + type;
            line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            c.appendChild(line);
            c.scrollTop = c.scrollHeight;
            if (c.children.length > 100) c.removeChild(c.firstChild);
        }
        
        // Comparison toggle
        document.getElementById('btn-before').onclick = function() {
            state.viewMode = 'before';
            this.classList.add('active');
            document.getElementById('btn-after').classList.remove('active');
            log('Viewing: Before Attack', 'info');
        };
        
        document.getElementById('btn-after').onclick = function() {
            state.viewMode = 'after';
            this.classList.add('active');
            document.getElementById('btn-before').classList.remove('active');
            log('Viewing: After Attack', 'info');
        };
        
        // Initialize
        createLayerButtons();
        createHeadButtons();
        log('Dashboard initialized - waiting for data...', 'success');
    </script>
</body>
</html>
'''
