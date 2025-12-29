"""
Detailed Transformer Visualization for Attack Analysis.

Provides comprehensive visualization of internal transformer states during
adversarial attacks, including:
- Token-level embedding vectors
- Q/K/V vector decomposition per attention head
- Attention weight matrices with heatmap
- MLP neuron activations
- Residual stream flow
- Output probability distribution
"""


def get_detailed_transformer_html() -> str:
    """Return the complete HTML for detailed transformer visualization."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA - Transformer Internals Analyzer</title>
    
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    
    <style>
        :root {
            --bg-dark: #0a0e14;
            --bg-panel: #0d1117;
            --bg-card: #161b22;
            --bg-highlight: #21262d;
            
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #484f58;
            
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-orange: #d29922;
            --accent-purple: #a371f7;
            --accent-cyan: #39c5cf;
            --accent-pink: #db61a2;
            
            --query-color: #58a6ff;
            --key-color: #f85149;
            --value-color: #3fb950;
            --output-color: #a371f7;
            
            --border-color: #30363d;
            --glow-blue: rgba(88, 166, 255, 0.4);
            --glow-green: rgba(63, 185, 80, 0.4);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .app-container {
            display: grid;
            grid-template-rows: auto 1fr auto;
            min-height: 100vh;
        }
        
        /* Header */
        .header {
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .logo {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.75rem;
            font-weight: 500;
            background: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
        }
        
        .status-badge.attacking {
            background: rgba(248, 81, 73, 0.15);
            color: var(--accent-red);
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }
        
        .header-metrics {
            display: flex;
            gap: 24px;
        }
        
        .metric {
            text-align: center;
        }
        
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent-cyan);
        }
        
        .metric-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Main Content */
        .main-content {
            display: grid;
            grid-template-columns: 320px 1fr 300px;
            gap: 1px;
            background: var(--border-color);
            overflow: hidden;
        }
        
        .panel {
            background: var(--bg-panel);
            overflow-y: auto;
        }
        
        .panel-header {
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 8px;
            position: sticky;
            top: 0;
            background: var(--bg-panel);
            z-index: 10;
        }
        
        .panel-icon {
            font-size: 1rem;
        }
        
        /* Left Panel - Token List */
        .token-list {
            padding: 12px;
        }
        
        .token-item {
            display: grid;
            grid-template-columns: 32px 1fr auto;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            margin-bottom: 4px;
            background: var(--bg-card);
            border-radius: 6px;
            border: 1px solid transparent;
            cursor: pointer;
            transition: all 0.15s;
        }
        
        .token-item:hover {
            background: var(--bg-highlight);
            border-color: var(--border-color);
        }
        
        .token-item.selected {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 1px var(--accent-blue);
        }
        
        .token-item.adversarial {
            border-color: var(--accent-red);
            background: rgba(248, 81, 73, 0.1);
        }
        
        .token-idx {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-muted);
        }
        
        .token-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .token-embedding-mini {
            width: 60px;
            height: 16px;
            background: var(--bg-highlight);
            border-radius: 2px;
            overflow: hidden;
        }
        
        /* Center Panel - Transformer Layers */
        .transformer-view {
            display: flex;
            flex-direction: column;
        }
        
        .layer-tabs {
            display: flex;
            gap: 4px;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            overflow-x: auto;
            background: var(--bg-card);
        }
        
        .layer-tab {
            padding: 6px 16px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-secondary);
            transition: all 0.15s;
            white-space: nowrap;
        }
        
        .layer-tab:hover {
            background: var(--bg-highlight);
            color: var(--text-primary);
        }
        
        .layer-tab.active {
            background: var(--accent-blue);
            color: white;
            border-color: var(--accent-blue);
        }
        
        .layer-content {
            flex: 1;
            padding: 16px;
            display: grid;
            grid-template-rows: auto 1fr auto;
            gap: 16px;
            overflow-y: auto;
        }
        
        /* QKV Section */
        .qkv-section {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }
        
        .qkv-card {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }
        
        .qkv-header {
            padding: 10px 14px;
            font-weight: 600;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .qkv-header.query { 
            background: rgba(88, 166, 255, 0.1); 
            color: var(--query-color);
        }
        .qkv-header.key { 
            background: rgba(248, 81, 73, 0.1); 
            color: var(--key-color);
        }
        .qkv-header.value { 
            background: rgba(63, 185, 80, 0.1); 
            color: var(--value-color);
        }
        
        .qkv-canvas {
            width: 100%;
            height: 120px;
            display: block;
        }
        
        .qkv-stats {
            padding: 8px 14px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
        }
        
        /* Attention Matrix */
        .attention-section {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }
        
        .attention-header {
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
        }
        
        .attention-title {
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .head-selector {
            display: flex;
            gap: 4px;
        }
        
        .head-btn {
            width: 28px;
            height: 28px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background: var(--bg-highlight);
            color: var(--text-secondary);
            font-size: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.15s;
        }
        
        .head-btn:hover {
            background: var(--bg-panel);
            color: var(--text-primary);
        }
        
        .head-btn.active {
            background: var(--accent-purple);
            color: white;
            border-color: var(--accent-purple);
        }
        
        .attention-matrix-container {
            padding: 16px;
            display: flex;
            justify-content: center;
        }
        
        .attention-matrix {
            display: grid;
            gap: 2px;
        }
        
        .attention-cell {
            width: 32px;
            height: 32px;
            border-radius: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.6rem;
            font-family: 'JetBrains Mono', monospace;
            color: rgba(255, 255, 255, 0.8);
            cursor: pointer;
            transition: transform 0.1s;
        }
        
        .attention-cell:hover {
            transform: scale(1.2);
            z-index: 10;
        }
        
        .attention-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            text-align: center;
            padding: 4px;
        }
        
        .attention-label.row-label {
            writing-mode: vertical-rl;
            text-orientation: mixed;
        }
        
        /* MLP Section */
        .mlp-section {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }
        
        .mlp-header {
            padding: 12px 16px;
            font-weight: 600;
            font-size: 0.9rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .mlp-viz {
            padding: 16px;
            height: 140px;
        }
        
        .neuron-bar {
            fill: var(--accent-orange);
            transition: fill 0.2s;
        }
        
        .neuron-bar.negative {
            fill: var(--accent-purple);
        }
        
        .neuron-bar:hover {
            fill: var(--accent-cyan);
        }
        
        /* Right Panel - Analysis */
        .analysis-panel {
            display: flex;
            flex-direction: column;
        }
        
        .analysis-section {
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
        }
        
        .analysis-title {
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        /* Output Probabilities */
        .prob-bar-container {
            margin-bottom: 8px;
        }
        
        .prob-bar-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        
        .prob-token {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }
        
        .prob-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        .prob-bar {
            height: 6px;
            background: var(--bg-highlight);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .prob-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        
        /* Attack Progress */
        .attack-progress {
            padding: 16px;
        }
        
        .attack-step-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 12px;
        }
        
        .step-card {
            background: var(--bg-card);
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }
        
        .step-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--accent-cyan);
        }
        
        .step-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 2px;
        }
        
        /* Residual Stream */
        .residual-flow {
            height: 80px;
            position: relative;
        }
        
        .residual-svg {
            width: 100%;
            height: 100%;
        }
        
        /* Adversarial Suffix Display */
        .suffix-display {
            background: var(--bg-card);
            border-radius: 6px;
            padding: 12px;
            margin-top: 12px;
        }
        
        .suffix-label {
            font-size: 0.7rem;
            color: var(--accent-red);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .suffix-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--accent-orange);
            word-break: break-all;
            line-height: 1.5;
        }
        
        /* Loss Chart */
        .loss-chart {
            height: 100px;
            margin-top: 12px;
        }
        
        .loss-line {
            fill: none;
            stroke: var(--accent-cyan);
            stroke-width: 2;
        }
        
        .loss-area {
            fill: url(#lossGradient);
        }
        
        /* Footer */
        .footer {
            background: var(--bg-panel);
            border-top: 1px solid var(--border-color);
            padding: 8px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .connection-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
        }
        
        .connection-dot.disconnected {
            background: var(--accent-red);
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header class="header">
            <div class="header-left">
                <div class="logo">MIRA</div>
                <div class="status-badge" id="statusBadge">
                    <span class="status-dot"></span>
                    <span id="statusText">Idle</span>
                </div>
            </div>
            <div class="header-metrics">
                <div class="metric">
                    <div class="metric-value" id="stepCount">0</div>
                    <div class="metric-label">Step</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="lossValue">--</div>
                    <div class="metric-label">Loss</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="asrValue">0%</div>
                    <div class="metric-label">ASR</div>
                </div>
            </div>
        </header>
        
        <main class="main-content">
            <!-- Left Panel: Token List -->
            <div class="panel left-panel">
                <div class="panel-header">
                    <span class="panel-icon">📝</span>
                    Input Tokens
                    <span style="margin-left: auto; font-size: 0.75rem; color: var(--text-muted);" id="tokenCount">0 tokens</span>
                </div>
                <div class="token-list" id="tokenList">
                    <div style="color: var(--text-muted); font-size: 0.85rem; padding: 16px; text-align: center;">
                        Waiting for input...
                    </div>
                </div>
            </div>
            
            <!-- Center Panel: Transformer Layers -->
            <div class="panel transformer-view">
                <div class="layer-tabs" id="layerTabs">
                    <!-- Layer tabs will be generated dynamically -->
                </div>
                <div class="layer-content">
                    <!-- QKV Vectors -->
                    <div class="qkv-section">
                        <div class="qkv-card">
                            <div class="qkv-header query">
                                <span>Query (Q)</span>
                                <span style="font-size: 0.7rem; opacity: 0.7;">d=64</span>
                            </div>
                            <canvas class="qkv-canvas" id="queryCanvas"></canvas>
                            <div class="qkv-stats">
                                <span>μ: <span id="queryMean">--</span></span>
                                <span>σ: <span id="queryStd">--</span></span>
                                <span>‖·‖: <span id="queryNorm">--</span></span>
                            </div>
                        </div>
                        <div class="qkv-card">
                            <div class="qkv-header key">
                                <span>Key (K)</span>
                                <span style="font-size: 0.7rem; opacity: 0.7;">d=64</span>
                            </div>
                            <canvas class="qkv-canvas" id="keyCanvas"></canvas>
                            <div class="qkv-stats">
                                <span>μ: <span id="keyMean">--</span></span>
                                <span>σ: <span id="keyStd">--</span></span>
                                <span>‖·‖: <span id="keyNorm">--</span></span>
                            </div>
                        </div>
                        <div class="qkv-card">
                            <div class="qkv-header value">
                                <span>Value (V)</span>
                                <span style="font-size: 0.7rem; opacity: 0.7;">d=64</span>
                            </div>
                            <canvas class="qkv-canvas" id="valueCanvas"></canvas>
                            <div class="qkv-stats">
                                <span>μ: <span id="valueMean">--</span></span>
                                <span>σ: <span id="valueStd">--</span></span>
                                <span>‖·‖: <span id="valueNorm">--</span></span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Attention Matrix -->
                    <div class="attention-section">
                        <div class="attention-header">
                            <span class="attention-title">Attention Weights</span>
                            <div class="head-selector" id="headSelector">
                                <!-- Head buttons generated dynamically -->
                            </div>
                        </div>
                        <div class="attention-matrix-container" id="attentionContainer">
                            <div style="color: var(--text-muted); font-size: 0.85rem;">
                                Waiting for attention data...
                            </div>
                        </div>
                    </div>
                    
                    <!-- MLP Activations -->
                    <div class="mlp-section">
                        <div class="mlp-header">
                            <span>MLP Neuron Activations</span>
                            <span style="font-size: 0.75rem; color: var(--text-secondary);">Top 32 neurons</span>
                        </div>
                        <div class="mlp-viz" id="mlpViz">
                            <svg width="100%" height="100%" id="mlpSvg"></svg>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel: Analysis -->
            <div class="panel analysis-panel">
                <div class="panel-header">
                    <span class="panel-icon">📊</span>
                    Attack Analysis
                </div>
                
                <div class="analysis-section">
                    <div class="analysis-title">
                        <span>📈</span> Output Probabilities
                    </div>
                    <div id="outputProbs">
                        <div style="color: var(--text-muted); font-size: 0.8rem;">
                            Waiting for output...
                        </div>
                    </div>
                </div>
                
                <div class="analysis-section">
                    <div class="analysis-title">
                        <span>🎯</span> Attack Progress
                    </div>
                    <div class="attack-step-info">
                        <div class="step-card">
                            <div class="step-value" id="currentStep">0</div>
                            <div class="step-label">Current Step</div>
                        </div>
                        <div class="step-card">
                            <div class="step-value" id="bestLoss">--</div>
                            <div class="step-label">Best Loss</div>
                        </div>
                    </div>
                    <div class="loss-chart" id="lossChart">
                        <svg width="100%" height="100%" id="lossChartSvg"></svg>
                    </div>
                    <div class="suffix-display">
                        <div class="suffix-label">Adversarial Suffix</div>
                        <div class="suffix-text" id="suffixText">--</div>
                    </div>
                </div>
                
                <div class="analysis-section">
                    <div class="analysis-title">
                        <span>🌊</span> Residual Stream
                    </div>
                    <div class="residual-flow">
                        <svg class="residual-svg" id="residualSvg"></svg>
                    </div>
                </div>
            </div>
        </main>
        
        <footer class="footer">
            <div class="connection-status">
                <span class="connection-dot" id="connectionDot"></span>
                <span id="connectionText">Connecting...</span>
            </div>
            <div>
                Events: <span id="eventCount">0</span> | MIRA Framework v1.0
            </div>
        </footer>
    </div>

    <script>
        // Application State
        const state = {
            currentLayer: 0,
            currentHead: 0,
            numLayers: 12,
            numHeads: 8,
            tokens: [],
            selectedToken: 0,
            lossHistory: [],
            eventCount: 0,
            isConnected: false,
            bestLoss: Infinity,
            attackStep: 0
        };
        
        // Initialize the application
        function init() {
            initLayerTabs();
            initHeadSelector();
            initCanvases();
            initLossChart();
            initResidualViz();
            connectEventSource();
        }
        
        // Layer tab initialization
        function initLayerTabs() {
            const container = document.getElementById('layerTabs');
            container.innerHTML = '';
            
            for (let i = 0; i < state.numLayers; i++) {
                const tab = document.createElement('div');
                tab.className = 'layer-tab' + (i === 0 ? ' active' : '');
                tab.textContent = 'L' + i;
                tab.onclick = () => selectLayer(i);
                container.appendChild(tab);
            }
        }
        
        function selectLayer(idx) {
            state.currentLayer = idx;
            document.querySelectorAll('.layer-tab').forEach((tab, i) => {
                tab.classList.toggle('active', i === idx);
            });
        }
        
        // Head selector initialization
        function initHeadSelector() {
            const container = document.getElementById('headSelector');
            container.innerHTML = '';
            
            for (let i = 0; i < state.numHeads; i++) {
                const btn = document.createElement('button');
                btn.className = 'head-btn' + (i === 0 ? ' active' : '');
                btn.textContent = i;
                btn.onclick = () => selectHead(i);
                container.appendChild(btn);
            }
        }
        
        function selectHead(idx) {
            state.currentHead = idx;
            document.querySelectorAll('.head-btn').forEach((btn, i) => {
                btn.classList.toggle('active', i === idx);
            });
        }
        
        // Canvas initialization for QKV vectors
        function initCanvases() {
            ['queryCanvas', 'keyCanvas', 'valueCanvas'].forEach(id => {
                const canvas = document.getElementById(id);
                const ctx = canvas.getContext('2d');
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width * 2;
                canvas.height = rect.height * 2;
                ctx.scale(2, 2);
                drawVectorPlaceholder(ctx, rect.width, rect.height);
            });
        }
        
        function drawVectorPlaceholder(ctx, width, height) {
            ctx.fillStyle = '#21262d';
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = '#484f58';
            ctx.font = '11px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for data...', width/2, height/2);
        }
        
        function drawVectorVisualization(canvasId, data, colorScale) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const width = canvas.width / 2;
            const height = canvas.height / 2;
            
            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = '#161b22';
            ctx.fillRect(0, 0, width, height);
            
            if (!data || data.length === 0) return;
            
            const barWidth = width / data.length;
            const maxVal = Math.max(...data.map(Math.abs));
            
            data.forEach((val, i) => {
                const normalized = val / (maxVal || 1);
                const barHeight = Math.abs(normalized) * (height / 2 - 4);
                const x = i * barWidth;
                const y = normalized >= 0 ? height/2 - barHeight : height/2;
                
                const intensity = Math.abs(normalized);
                ctx.fillStyle = colorScale(intensity);
                ctx.fillRect(x, y, barWidth - 1, barHeight);
            });
        }
        
        // Color scales for QKV
        const colorScales = {
            query: (t) => `rgba(88, 166, 255, ${0.3 + t * 0.7})`,
            key: (t) => `rgba(248, 81, 73, ${0.3 + t * 0.7})`,
            value: (t) => `rgba(63, 185, 80, ${0.3 + t * 0.7})`
        };
        
        // Loss chart initialization
        function initLossChart() {
            const svg = d3.select('#lossChartSvg');
            svg.selectAll('*').remove();
            
            const defs = svg.append('defs');
            const gradient = defs.append('linearGradient')
                .attr('id', 'lossGradient')
                .attr('x1', '0%').attr('y1', '0%')
                .attr('x2', '0%').attr('y2', '100%');
            gradient.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(57, 197, 207, 0.3)');
            gradient.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(57, 197, 207, 0)');
        }
        
        function updateLossChart() {
            const svg = d3.select('#lossChartSvg');
            const container = document.getElementById('lossChart');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            if (state.lossHistory.length < 2) return;
            
            const xScale = d3.scaleLinear()
                .domain([0, state.lossHistory.length - 1])
                .range([0, width]);
            
            const yScale = d3.scaleLinear()
                .domain([0, d3.max(state.lossHistory) * 1.1])
                .range([height - 4, 4]);
            
            const line = d3.line()
                .x((d, i) => xScale(i))
                .y(d => yScale(d))
                .curve(d3.curveMonotoneX);
            
            const area = d3.area()
                .x((d, i) => xScale(i))
                .y0(height)
                .y1(d => yScale(d))
                .curve(d3.curveMonotoneX);
            
            svg.selectAll('.loss-area').remove();
            svg.selectAll('.loss-line').remove();
            
            svg.append('path')
                .datum(state.lossHistory)
                .attr('class', 'loss-area')
                .attr('d', area);
            
            svg.append('path')
                .datum(state.lossHistory)
                .attr('class', 'loss-line')
                .attr('d', line);
        }
        
        // Residual stream visualization
        function initResidualViz() {
            const svg = d3.select('#residualSvg');
            svg.selectAll('*').remove();
        }
        
        function updateResidualViz(layerNorms) {
            const svg = d3.select('#residualSvg');
            const container = document.getElementById('residualSvg');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            svg.selectAll('*').remove();
            
            if (!layerNorms || layerNorms.length === 0) return;
            
            const xScale = d3.scaleLinear()
                .domain([0, layerNorms.length - 1])
                .range([20, width - 20]);
            
            const maxNorm = Math.max(...layerNorms);
            const rScale = d3.scaleLinear()
                .domain([0, maxNorm])
                .range([4, 16]);
            
            // Draw connecting line
            const line = d3.line()
                .x((d, i) => xScale(i))
                .y(height / 2)
                .curve(d3.curveMonotoneX);
            
            svg.append('path')
                .datum(layerNorms)
                .attr('fill', 'none')
                .attr('stroke', '#30363d')
                .attr('stroke-width', 2)
                .attr('d', line);
            
            // Draw nodes
            layerNorms.forEach((norm, i) => {
                const x = xScale(i);
                const r = rScale(norm);
                
                svg.append('circle')
                    .attr('cx', x)
                    .attr('cy', height / 2)
                    .attr('r', r)
                    .attr('fill', `rgba(57, 197, 207, ${0.3 + (norm/maxNorm) * 0.7})`)
                    .attr('stroke', '#39c5cf')
                    .attr('stroke-width', 1);
                
                svg.append('text')
                    .attr('x', x)
                    .attr('y', height / 2 + r + 12)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#8b949e')
                    .attr('font-size', '9px')
                    .attr('font-family', 'JetBrains Mono')
                    .text('L' + i);
            });
        }
        
        // Token list update
        function updateTokenList(tokens, adversarialStart = -1) {
            state.tokens = tokens;
            const container = document.getElementById('tokenList');
            container.innerHTML = '';
            
            tokens.forEach((token, i) => {
                const item = document.createElement('div');
                item.className = 'token-item';
                if (i === state.selectedToken) item.classList.add('selected');
                if (adversarialStart >= 0 && i >= adversarialStart) item.classList.add('adversarial');
                
                item.innerHTML = `
                    <span class="token-idx">${i}</span>
                    <span class="token-text">${escapeHtml(token)}</span>
                    <div class="token-embedding-mini" id="emb-mini-${i}"></div>
                `;
                item.onclick = () => selectToken(i);
                container.appendChild(item);
            });
            
            document.getElementById('tokenCount').textContent = tokens.length + ' tokens';
        }
        
        function selectToken(idx) {
            state.selectedToken = idx;
            document.querySelectorAll('.token-item').forEach((item, i) => {
                item.classList.toggle('selected', i === idx);
            });
        }
        
        // Attention matrix rendering
        function renderAttentionMatrix(weights, tokens) {
            const container = document.getElementById('attentionContainer');
            const n = Math.min(weights.length, tokens.length, 10);
            
            let html = '<div class="attention-matrix" style="grid-template-columns: auto repeat(' + n + ', 1fr);">';
            
            // Header row
            html += '<div></div>';
            for (let j = 0; j < n; j++) {
                html += '<div class="attention-label">' + escapeHtml(tokens[j].slice(0, 4)) + '</div>';
            }
            
            // Data rows
            for (let i = 0; i < n; i++) {
                html += '<div class="attention-label row-label">' + escapeHtml(tokens[i].slice(0, 4)) + '</div>';
                for (let j = 0; j < n; j++) {
                    const val = weights[i][j];
                    const color = getAttentionColor(val);
                    html += `<div class="attention-cell" style="background:${color}" title="${tokens[i]} → ${tokens[j]}: ${val.toFixed(3)}">${val.toFixed(2)}</div>`;
                }
            }
            
            html += '</div>';
            container.innerHTML = html;
        }
        
        function getAttentionColor(val) {
            const r = Math.round(163 + (255 - 163) * val);
            const g = Math.round(113 + (200 - 113) * val);
            const b = Math.round(247 - 100 * val);
            return `rgba(${r}, ${g}, ${b}, ${0.3 + val * 0.7})`;
        }
        
        // MLP visualization
        function updateMLPViz(activations, topNeurons) {
            const svg = d3.select('#mlpSvg');
            const container = document.getElementById('mlpViz');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            svg.selectAll('*').remove();
            
            if (!activations || activations.length === 0) return;
            
            const barWidth = (width - 40) / activations.length;
            const maxAct = Math.max(...activations.map(Math.abs));
            
            activations.forEach((act, i) => {
                const normalized = act / (maxAct || 1);
                const barHeight = Math.abs(normalized) * (height / 2 - 10);
                const x = 20 + i * barWidth;
                const y = normalized >= 0 ? height/2 - barHeight : height/2;
                
                svg.append('rect')
                    .attr('class', 'neuron-bar' + (normalized < 0 ? ' negative' : ''))
                    .attr('x', x)
                    .attr('y', y)
                    .attr('width', barWidth - 2)
                    .attr('height', barHeight)
                    .attr('rx', 2);
            });
            
            // Add axis line
            svg.append('line')
                .attr('x1', 20).attr('y1', height/2)
                .attr('x2', width - 20).attr('y2', height/2)
                .attr('stroke', '#30363d')
                .attr('stroke-width', 1);
        }
        
        // Output probabilities
        function updateOutputProbs(tokens, probs) {
            const container = document.getElementById('outputProbs');
            container.innerHTML = '';
            
            tokens.forEach((token, i) => {
                const prob = probs[i] || 0;
                const div = document.createElement('div');
                div.className = 'prob-bar-container';
                div.innerHTML = `
                    <div class="prob-bar-header">
                        <span class="prob-token">${escapeHtml(token)}</span>
                        <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div class="prob-bar">
                        <div class="prob-bar-fill" style="width: ${prob * 100}%"></div>
                    </div>
                `;
                container.appendChild(div);
            });
        }
        
        // Event Source connection
        function connectEventSource() {
            const eventSource = new EventSource('/api/events');
            
            eventSource.onopen = () => {
                state.isConnected = true;
                document.getElementById('connectionDot').classList.remove('disconnected');
                document.getElementById('connectionText').textContent = 'Connected';
            };
            
            eventSource.onerror = () => {
                state.isConnected = false;
                document.getElementById('connectionDot').classList.add('disconnected');
                document.getElementById('connectionText').textContent = 'Disconnected - Reconnecting...';
            };
            
            eventSource.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    handleEvent(data);
                } catch (err) {
                    console.error('Failed to parse event:', err);
                }
            };
        }
        
        function handleEvent(event) {
            state.eventCount++;
            document.getElementById('eventCount').textContent = state.eventCount;
            
            const type = event.event_type || event.type;
            const data = event.data || event;
            
            switch (type) {
                case 'attack_step':
                    handleAttackStep(data);
                    break;
                case 'embeddings':
                    handleEmbeddings(data);
                    break;
                case 'qkv':
                    handleQKV(data);
                    break;
                case 'attention_matrix':
                    handleAttentionMatrix(data);
                    break;
                case 'mlp':
                    handleMLP(data);
                    break;
                case 'output_probs':
                    handleOutputProbs(data);
                    break;
                case 'layer_update':
                    handleLayerUpdate(data);
                    break;
                case 'residual':
                    handleResidual(data);
                    break;
            }
        }
        
        function handleAttackStep(data) {
            state.attackStep = data.step || 0;
            const loss = data.loss || 0;
            
            document.getElementById('stepCount').textContent = state.attackStep;
            document.getElementById('currentStep').textContent = state.attackStep;
            document.getElementById('lossValue').textContent = loss.toFixed(4);
            
            if (loss < state.bestLoss) {
                state.bestLoss = loss;
                document.getElementById('bestLoss').textContent = loss.toFixed(4);
            }
            
            if (data.suffix) {
                document.getElementById('suffixText').textContent = data.suffix;
            }
            
            state.lossHistory.push(loss);
            if (state.lossHistory.length > 100) state.lossHistory.shift();
            updateLossChart();
            
            // Update status
            const badge = document.getElementById('statusBadge');
            const statusText = document.getElementById('statusText');
            if (data.success) {
                badge.classList.remove('attacking');
                badge.style.background = 'rgba(63, 185, 80, 0.15)';
                badge.style.color = '#3fb950';
                statusText.textContent = 'Success';
            } else if (state.attackStep > 0) {
                badge.classList.add('attacking');
                statusText.textContent = 'Attacking';
            }
        }
        
        function handleEmbeddings(data) {
            if (data.tokens) {
                updateTokenList(data.tokens, data.adversarial_start || -1);
            }
        }
        
        function handleQKV(data) {
            if (data.query_vectors) {
                drawVectorVisualization('queryCanvas', data.query_vectors, colorScales.query);
                updateStats('query', data.query_vectors);
            }
            if (data.key_vectors) {
                drawVectorVisualization('keyCanvas', data.key_vectors, colorScales.key);
                updateStats('key', data.key_vectors);
            }
            if (data.value_vectors) {
                drawVectorVisualization('valueCanvas', data.value_vectors, colorScales.value);
                updateStats('value', data.value_vectors);
            }
        }
        
        function updateStats(type, values) {
            if (!values || values.length === 0) return;
            const mean = values.reduce((a, b) => a + b, 0) / values.length;
            const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
            const std = Math.sqrt(variance);
            const norm = Math.sqrt(values.reduce((a, b) => a + b * b, 0));
            
            document.getElementById(type + 'Mean').textContent = mean.toFixed(3);
            document.getElementById(type + 'Std').textContent = std.toFixed(3);
            document.getElementById(type + 'Norm').textContent = norm.toFixed(3);
        }
        
        function handleAttentionMatrix(data) {
            if (data.attention_weights && data.tokens) {
                renderAttentionMatrix(data.attention_weights, data.tokens);
            }
        }
        
        function handleMLP(data) {
            if (data.activations) {
                updateMLPViz(data.activations, data.top_neurons || []);
            }
        }
        
        function handleOutputProbs(data) {
            if (data.tokens && data.probabilities) {
                updateOutputProbs(data.tokens, data.probabilities);
            }
        }
        
        function handleLayerUpdate(data) {
            // Could update layer-specific visualizations
        }
        
        function handleResidual(data) {
            if (data.layer_norms) {
                updateResidualViz(data.layer_norms);
            }
        }
        
        // Utility functions
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Initialize on load
        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>'''


# Export function for backward compatibility
def get_transformer_attack_html() -> str:
    """Alias for get_detailed_transformer_html for compatibility."""
    return get_detailed_transformer_html()

