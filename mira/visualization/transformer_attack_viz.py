"""
Transformer Attack Visualization - Professional Real-time Dashboard.

This module provides a comprehensive real-time visualization of the 
Transformer processing during adversarial attacks.

Features:
- Complete Transformer flow: Embedding -> QKV -> Attention -> MLP -> Output
- Terminal-style attack console
- Real-time attention heatmaps
- Layer-by-layer activation flow
- Attack progress with animated effects

Usage:
    from mira.visualization.transformer_attack_viz import get_transformer_attack_html
"""


def get_transformer_attack_html() -> str:
    """
    Returns the complete HTML for the Transformer Attack Visualization dashboard.
    Provides detailed visualization of internal model states during attacks.
    """
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA - Transformer Attack Visualizer</title>
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- D3.js for visualizations -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    
    <style>
        :root {
            /* Color Palette - Cyberpunk/Tech theme */
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --bg-accent: #1c2128;
            
            /* Text Colors */
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #484f58;
            
            /* Accent Colors */
            --accent-cyan: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-orange: #db6d28;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
            --accent-pink: #db61a2;
            
            /* Transformer Components */
            --embedding-color: #58a6ff;
            --query-color: #3fb950;
            --key-color: #f85149;
            --value-color: #a371f7;
            --attention-color: #d29922;
            --mlp-color: #db6d28;
            --output-color: #db61a2;
            
            /* Borders */
            --border-default: #30363d;
            --border-muted: #21262d;
            
            /* Shadows */
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.2);
            --shadow-md: 0 4px 8px rgba(0,0,0,0.3);
            --shadow-lg: 0 8px 24px rgba(0,0,0,0.4);
            
            /* Sizes */
            --header-height: 60px;
            --sidebar-width: 320px;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* ========== HEADER ========== */
        .header {
            height: var(--header-height);
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-default);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            animation: logoGlow 3s ease-in-out infinite;
        }
        
        @keyframes logoGlow {
            0%, 100% { box-shadow: 0 0 10px rgba(88, 166, 255, 0.3); }
            50% { box-shadow: 0 0 20px rgba(88, 166, 255, 0.6); }
        }
        
        .logo-text {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 18px;
            letter-spacing: 2px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header-stats {
            display: flex;
            gap: 24px;
        }
        
        .stat-item {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }
        
        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 20px;
            font-weight: 600;
            color: var(--accent-cyan);
        }
        
        .stat-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* ========== MAIN LAYOUT ========== */
        .main-container {
            display: flex;
            margin-top: var(--header-height);
            height: calc(100vh - var(--header-height));
        }
        
        /* ========== SIDEBAR (Attack Console) ========== */
        .sidebar {
            width: var(--sidebar-width);
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-default);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }
        
        .sidebar-section {
            padding: 16px;
            border-bottom: 1px solid var(--border-default);
        }
        
        .section-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .section-title::before {
            content: '';
            width: 8px;
            height: 8px;
            background: var(--accent-cyan);
            border-radius: 2px;
        }
        
        /* Attack Console */
        .attack-console {
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            min-height: 150px;
            max-height: 300px;
        }
        
        .console-output {
            flex: 1;
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            line-height: 1.6;
            overflow-y: auto;
            overflow-x: hidden;
            border: 1px solid var(--border-muted);
            max-height: 250px;
        }
        
        .console-line {
            margin-bottom: 4px;
            display: flex;
            gap: 8px;
        }
        
        .console-timestamp {
            color: var(--text-muted);
            flex-shrink: 0;
        }
        
        .console-message {
            word-break: break-word;
        }
        
        .console-line.success .console-message { color: var(--accent-green); }
        .console-line.error .console-message { color: var(--accent-red); }
        .console-line.warning .console-message { color: var(--accent-yellow); }
        .console-line.info .console-message { color: var(--accent-cyan); }
        .console-line.attack .console-message { color: var(--accent-purple); }
        
        /* Metrics Cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        
        .metric-card {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
            border: 1px solid var(--border-muted);
            transition: all 0.2s ease;
        }
        
        .metric-card:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
        }
        
        .metric-card .value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 24px;
            font-weight: 600;
            color: var(--accent-cyan);
        }
        
        .metric-card .label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 4px;
        }
        
        /* Suffix Display */
        .suffix-display {
            background: var(--bg-primary);
            border: 1px solid var(--accent-purple);
            border-radius: 8px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--accent-purple);
            word-break: break-all;
            min-height: 60px;
            position: relative;
            overflow: hidden;
        }
        
        .suffix-display::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-purple), var(--accent-pink));
            animation: suffixScan 2s linear infinite;
        }
        
        @keyframes suffixScan {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        /* ========== MAIN CONTENT (Transformer Visualization) ========== */
        .content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            padding: 24px;
            gap: 20px;
        }
        
        /* Transformer Flow Diagram */
        .transformer-flow {
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-default);
            padding: 20px;
            flex-shrink: 0;
        }
        
        .flow-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .flow-title .icon { font-size: 16px; }
        
        .flow-diagram {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 10px 0;
        }
        
        .flow-stage {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            flex: 1;
            max-width: 140px;
        }
        
        .stage-box {
            width: 100%;
            padding: 16px 12px;
            border-radius: 8px;
            text-align: center;
            font-size: 12px;
            font-weight: 500;
            position: relative;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .stage-box.embedding { background: rgba(88, 166, 255, 0.15); color: var(--embedding-color); border-color: rgba(88, 166, 255, 0.3); }
        .stage-box.qkv { background: rgba(63, 185, 80, 0.15); color: var(--query-color); border-color: rgba(63, 185, 80, 0.3); }
        .stage-box.attention { background: rgba(210, 153, 34, 0.15); color: var(--attention-color); border-color: rgba(210, 153, 34, 0.3); }
        .stage-box.mlp { background: rgba(219, 109, 40, 0.15); color: var(--mlp-color); border-color: rgba(219, 109, 40, 0.3); }
        .stage-box.output { background: rgba(219, 97, 162, 0.15); color: var(--output-color); border-color: rgba(219, 97, 162, 0.3); }
        
        .stage-box.active {
            animation: stagePulse 1s ease-in-out infinite;
        }
        
        @keyframes stagePulse {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 currentColor; }
            50% { transform: scale(1.05); box-shadow: 0 0 20px currentColor; }
        }
        
        .stage-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            margin-top: 4px;
            opacity: 0.8;
        }
        
        .flow-arrow {
            color: var(--text-muted);
            font-size: 20px;
            flex-shrink: 0;
        }
        
        /* Tokens Display */
        .tokens-section {
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-default);
            padding: 20px;
        }
        
        .tokens-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .token {
            padding: 6px 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-muted);
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            transition: all 0.2s ease;
            position: relative;
        }
        
        .token:hover {
            border-color: var(--accent-cyan);
            background: rgba(88, 166, 255, 0.1);
        }
        
        .token.highlighted {
            border-color: var(--accent-yellow);
            background: rgba(210, 153, 34, 0.2);
        }
        
        .token.adversarial {
            border-color: var(--accent-red);
            background: rgba(248, 81, 73, 0.2);
            animation: tokenGlow 1.5s ease-in-out infinite;
        }
        
        @keyframes tokenGlow {
            0%, 100% { box-shadow: 0 0 5px rgba(248, 81, 73, 0.3); }
            50% { box-shadow: 0 0 15px rgba(248, 81, 73, 0.6); }
        }
        
        /* Main Visualization Area */
        .viz-area {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            min-height: 0;
        }
        
        .viz-panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-default);
            padding: 20px;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        
        .viz-panel-title {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .viz-panel-title .badge {
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 10px;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }
        
        .viz-content {
            flex: 1;
            position: relative;
            min-height: 0;
        }
        
        /* Attention Heatmap */
        .attention-heatmap {
            width: 100%;
            height: 100%;
        }
        
        .attention-heatmap svg {
            width: 100%;
            height: 100%;
        }
        
        .attention-cell {
            transition: all 0.3s ease;
        }
        
        .attention-cell:hover {
            stroke: white;
            stroke-width: 1;
        }
        
        /* Layer Flow Visualization */
        .layer-flow {
            display: flex;
            flex-direction: column;
            gap: 8px;
            overflow-y: auto;
        }
        
        .layer-row {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        
        .layer-row:hover {
            background: var(--bg-accent);
        }
        
        .layer-row.active {
            border-left: 3px solid var(--accent-cyan);
        }
        
        .layer-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
            width: 40px;
            flex-shrink: 0;
        }
        
        .layer-bar-container {
            flex: 1;
            display: flex;
            gap: 4px;
        }
        
        .layer-bar {
            height: 20px;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }
        
        .layer-bar.refusal {
            flex: 1;
            background: rgba(248, 81, 73, 0.2);
        }
        
        .layer-bar.acceptance {
            flex: 1;
            background: rgba(63, 185, 80, 0.2);
        }
        
        .layer-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .layer-bar.refusal .layer-bar-fill {
            background: linear-gradient(90deg, var(--accent-red), rgba(248, 81, 73, 0.5));
        }
        
        .layer-bar.acceptance .layer-bar-fill {
            background: linear-gradient(90deg, var(--accent-green), rgba(63, 185, 80, 0.5));
        }
        
        .layer-score {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
            width: 45px;
            text-align: right;
        }
        
        /* Output Probabilities */
        .prob-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .prob-item {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .prob-token {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            background: var(--bg-tertiary);
            padding: 4px 8px;
            border-radius: 4px;
            min-width: 60px;
            text-align: center;
        }
        
        .prob-bar-bg {
            flex: 1;
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .prob-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
            width: 50px;
            text-align: right;
        }
        
        /* Loss Chart */
        .loss-chart {
            width: 100%;
            height: 100%;
        }
        
        .loss-chart svg {
            width: 100%;
            height: 100%;
        }
        
        .loss-line {
            fill: none;
            stroke: var(--accent-cyan);
            stroke-width: 2;
        }
        
        .loss-area {
            fill: url(#lossGradient);
        }
        
        .axis-line {
            stroke: var(--border-default);
        }
        
        .axis-text {
            fill: var(--text-muted);
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Status Bar */
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 32px;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-default);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            font-size: 11px;
            z-index: 1000;
        }
        
        .status-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: statusPulse 2s ease-in-out infinite;
        }
        
        @keyframes statusPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-dot.warning { background: var(--accent-yellow); }
        .status-dot.error { background: var(--accent-red); }
        .status-dot.idle { background: var(--text-muted); animation: none; }
        
        .status-text {
            color: var(--text-secondary);
        }
        
        .status-right {
            display: flex;
            align-items: center;
            gap: 16px;
            color: var(--text-muted);
        }
        
        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-default);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-in {
            animation: fadeIn 0.3s ease forwards;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="logo">
            <div class="logo-icon">🧠</div>
            <span class="logo-text">MIRA ATTACK VISUALIZER</span>
        </div>
        <div class="header-stats">
            <div class="stat-item">
                <span class="stat-value" id="header-step">0</span>
                <span class="stat-label">Step</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="header-loss">--</span>
                <span class="stat-label">Loss</span>
            </div>
            <div class="stat-item">
                <span class="stat-value" id="header-asr">0%</span>
                <span class="stat-label">ASR</span>
            </div>
        </div>
    </header>
    
    <!-- Main Container -->
    <div class="main-container">
        <!-- Sidebar: Attack Console -->
        <aside class="sidebar">
            <!-- Metrics -->
            <div class="sidebar-section">
                <div class="section-title">Attack Metrics</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="value" id="metric-step">0</div>
                        <div class="label">Current Step</div>
                    </div>
                    <div class="metric-card">
                        <div class="value" id="metric-loss">--</div>
                        <div class="label">Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="value" id="metric-best">--</div>
                        <div class="label">Best Loss</div>
                    </div>
                    <div class="metric-card">
                        <div class="value" id="metric-asr">0%</div>
                        <div class="label">Success Rate</div>
                    </div>
                </div>
            </div>
            
            <!-- Adversarial Suffix -->
            <div class="sidebar-section">
                <div class="section-title">Adversarial Suffix</div>
                <div class="suffix-display" id="suffix-display">
                    Waiting for attack to start...
                </div>
            </div>
            
            <!-- Attack Console -->
            <div class="sidebar-section attack-console">
                <div class="section-title">Attack Console</div>
                <div class="console-output" id="console-output">
                    <div class="console-line info">
                        <span class="console-timestamp">[00:00:00]</span>
                        <span class="console-message">MIRA Attack Visualizer initialized</span>
                    </div>
                    <div class="console-line info">
                        <span class="console-timestamp">[00:00:00]</span>
                        <span class="console-message">Waiting for connection...</span>
                    </div>
                </div>
            </div>
        </aside>
        
        <!-- Main Content -->
        <main class="content">
            <!-- Transformer Flow Diagram -->
            <div class="transformer-flow">
                <div class="flow-title">
                    <span class="icon">🔄</span>
                    <span>Transformer Processing Flow</span>
                </div>
                <div class="flow-diagram">
                    <div class="flow-stage">
                        <div class="stage-box embedding" id="stage-embedding">
                            <div>Embedding</div>
                            <div class="stage-value" id="embedding-val">--</div>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-stage">
                        <div class="stage-box qkv" id="stage-qkv">
                            <div>Q / K / V</div>
                            <div class="stage-value" id="qkv-val">--</div>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-stage">
                        <div class="stage-box attention" id="stage-attention">
                            <div>Attention</div>
                            <div class="stage-value" id="attention-val">--</div>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-stage">
                        <div class="stage-box mlp" id="stage-mlp">
                            <div>MLP</div>
                            <div class="stage-value" id="mlp-val">--</div>
                        </div>
                    </div>
                    <div class="flow-arrow">→</div>
                    <div class="flow-stage">
                        <div class="stage-box output" id="stage-output">
                            <div>Output</div>
                            <div class="stage-value" id="output-val">--</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Tokens Display -->
            <div class="tokens-section">
                <div class="flow-title">
                    <span class="icon">📝</span>
                    <span>Input Tokens</span>
                    <span id="token-count" style="margin-left: auto; font-size: 12px; color: var(--text-muted);">0 tokens</span>
                </div>
                <div class="tokens-list" id="tokens-list">
                    <span class="token">Waiting for input...</span>
                </div>
            </div>
            
            <!-- Visualization Panels -->
            <div class="viz-area">
                <!-- Attention Heatmap -->
                <div class="viz-panel">
                    <div class="viz-panel-title">
                        <span>🎯 Attention Weights</span>
                        <span class="badge" id="attention-layer">Layer 0 Head 0</span>
                    </div>
                    <div class="viz-content attention-heatmap" id="attention-heatmap">
                        <svg></svg>
                    </div>
                </div>
                
                <!-- Layer Flow -->
                <div class="viz-panel">
                    <div class="viz-panel-title">
                        <span>📊 Layer Activations</span>
                        <span class="badge">Refusal ↔ Acceptance</span>
                    </div>
                    <div class="viz-content layer-flow" id="layer-flow">
                        <!-- Layers will be populated dynamically -->
                    </div>
                </div>
                
                <!-- Loss Chart -->
                <div class="viz-panel">
                    <div class="viz-panel-title">
                        <span>📉 Loss Trajectory</span>
                        <span class="badge" id="loss-count">0 points</span>
                    </div>
                    <div class="viz-content loss-chart" id="loss-chart">
                        <svg></svg>
                    </div>
                </div>
                
                <!-- Output Probabilities -->
                <div class="viz-panel">
                    <div class="viz-panel-title">
                        <span>🎲 Output Probabilities</span>
                        <span class="badge">Top 5</span>
                    </div>
                    <div class="viz-content">
                        <div class="prob-list" id="prob-list">
                            <!-- Probabilities will be populated dynamically -->
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>
    
    <!-- Status Bar -->
    <div class="status-bar">
        <div class="status-left">
            <div class="status-indicator">
                <div class="status-dot" id="status-dot"></div>
                <span class="status-text" id="status-text">Connecting...</span>
            </div>
        </div>
        <div class="status-right">
            <span id="event-count">Events: 0</span>
            <span>MIRA Framework v1.0</span>
        </div>
    </div>
    
    <script>
        // ========== STATE ==========
        const state = {
            eventCount: 0,
            step: 0,
            lossHistory: [],
            bestLoss: Infinity,
            tokens: [],
            layers: [],
            attentionMatrix: null,
            outputProbs: [],
            connected: false,
            startTime: Date.now()
        };
        
        // ========== CONSOLE LOGGING ==========
        function log(message, type = 'info') {
            const console = document.getElementById('console-output');
            const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
            const time = new Date().toLocaleTimeString();
            
            const line = document.createElement('div');
            line.className = `console-line ${type} animate-in`;
            line.innerHTML = `
                <span class="console-timestamp">[${time}]</span>
                <span class="console-message">${message}</span>
            `;
            console.appendChild(line);
            
            // Keep only last 50 lines for better performance
            while (console.children.length > 50) {
                console.removeChild(console.firstChild);
            }
            
            console.scrollTop = console.scrollHeight;
        }
        
        // ========== EVENT SOURCE ==========
        const evtSource = new EventSource('/api/events');
        
        evtSource.onopen = function() {
            state.connected = true;
            updateStatus('connected', 'Connected to attack pipeline');
            log('Connected to MIRA attack pipeline', 'success');
        };
        
        evtSource.onerror = function() {
            state.connected = false;
            updateStatus('error', 'Connection lost - reconnecting...');
            log('Connection error - attempting to reconnect...', 'error');
        };
        
        evtSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            if (data.event_type === 'ping') return;
            
            state.eventCount++;
            document.getElementById('event-count').textContent = `Events: ${state.eventCount}`;
            
            handleEvent(data);
        };
        
        // ========== EVENT HANDLERS ==========
        function handleEvent(event) {
            const { event_type, data } = event;
            
            switch (event_type) {
                case 'layer':
                    handleLayerUpdate(data);
                    break;
                case 'attack_step':
                    handleAttackStep(data);
                    break;
                case 'attention_matrix':
                    handleAttentionMatrix(data);
                    break;
                case 'embeddings':
                    handleEmbeddings(data);
                    break;
                case 'qkv':
                    handleQKV(data);
                    break;
                case 'mlp':
                    handleMLP(data);
                    break;
                case 'output_probs':
                    handleOutputProbs(data);
                    break;
                case 'transformer_trace':
                    handleTransformerTrace(data);
                    break;
                case 'complete':
                    handleComplete(data);
                    break;
            }
        }
        
        function handleLayerUpdate(data) {
            const { layer, refusal_score, acceptance_score, direction } = data;
            
            // Update or create layer row
            let row = document.getElementById(`layer-${layer}`);
            if (!row) {
                row = createLayerRow(layer);
            }
            
            // Update bars
            const refusalBar = row.querySelector('.refusal .layer-bar-fill');
            const acceptanceBar = row.querySelector('.acceptance .layer-bar-fill');
            const scoreEl = row.querySelector('.layer-score');
            
            refusalBar.style.width = `${refusal_score * 100}%`;
            acceptanceBar.style.width = `${acceptance_score * 100}%`;
            scoreEl.textContent = `${(Math.max(refusal_score, acceptance_score) * 100).toFixed(0)}%`;
            
            // Highlight active layer
            document.querySelectorAll('.layer-row').forEach(r => r.classList.remove('active'));
            row.classList.add('active');
        }
        
        function createLayerRow(layer) {
            const container = document.getElementById('layer-flow');
            const row = document.createElement('div');
            row.id = `layer-${layer}`;
            row.className = 'layer-row animate-in';
            row.innerHTML = `
                <span class="layer-label">L${layer}</span>
                <div class="layer-bar-container">
                    <div class="layer-bar refusal">
                        <div class="layer-bar-fill" style="width: 0%"></div>
                    </div>
                    <div class="layer-bar acceptance">
                        <div class="layer-bar-fill" style="width: 0%"></div>
                    </div>
                </div>
                <span class="layer-score">0%</span>
            `;
            container.appendChild(row);
            return row;
        }
        
        function handleAttackStep(data) {
            const { step, loss, suffix, success } = data;
            
            state.step = step;
            state.lossHistory.push(loss);
            
            if (loss < state.bestLoss) {
                state.bestLoss = loss;
                log(`New best loss: ${loss.toFixed(4)}`, 'success');
            }
            
            // Update metrics
            document.getElementById('header-step').textContent = step;
            document.getElementById('header-loss').textContent = loss.toFixed(4);
            document.getElementById('metric-step').textContent = step;
            document.getElementById('metric-loss').textContent = loss.toFixed(4);
            document.getElementById('metric-best').textContent = state.bestLoss.toFixed(4);
            
            // Update suffix
            document.getElementById('suffix-display').textContent = suffix || 'Optimizing...';
            
            // Update status
            updateStatus('connected', `Processing step ${step}...`);
            
            // Draw loss chart
            drawLossChart();
            
            // Log
            if (step % 5 === 0) {
                log(`Step ${step}: loss=${loss.toFixed(4)}`, success ? 'success' : 'attack');
            }
            
            // Animate flow stages based on step
            animateFlowStages(step);
        }
        
        function handleAttentionMatrix(data) {
            const { layer, head, weights, tokens } = data;
            
            state.attentionMatrix = { weights, tokens };
            document.getElementById('attention-layer').textContent = `Layer ${layer} Head ${head}`;
            
            drawAttentionHeatmap(weights, tokens);
        }
        
        function handleEmbeddings(data) {
            const { tokens } = data;
            state.tokens = tokens;
            
            // Update tokens display
            const container = document.getElementById('tokens-list');
            container.innerHTML = tokens.map((t, i) => 
                `<span class="token" data-index="${i}">${escapeHtml(t)}</span>`
            ).join('');
            
            document.getElementById('token-count').textContent = `${tokens.length} tokens`;
            
            // Update embedding stage
            document.getElementById('embedding-val').textContent = `${tokens.length} tokens`;
            activateStage('embedding');
        }
        
        function handleQKV(data) {
            const { layer, tokens, query_vectors, key_vectors, value_vectors } = data;
            
            document.getElementById('qkv-val').textContent = `L${layer}`;
            activateStage('qkv');
        }
        
        function handleMLP(data) {
            const { layer, activations, top_neurons } = data;
            
            document.getElementById('mlp-val').textContent = `L${layer}`;
            activateStage('mlp');
        }
        
        function handleOutputProbs(data) {
            const { probs } = data;
            
            state.outputProbs = probs;
            document.getElementById('output-val').textContent = probs.length > 0 ? probs[0].token : '--';
            activateStage('output');
            
            // Update probability list
            const container = document.getElementById('prob-list');
            container.innerHTML = probs.slice(0, 5).map(p => `
                <div class="prob-item animate-in">
                    <span class="prob-token">${escapeHtml(p.token)}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar" style="width: ${p.prob * 100}%"></div>
                    </div>
                    <span class="prob-value">${(p.prob * 100).toFixed(1)}%</span>
                </div>
            `).join('');
        }
        
        function handleTransformerTrace(data) {
            const { trace_type, trace } = data;
            
            if (trace_type === 'adversarial') {
                log(`Adversarial trace captured (${trace.tokens?.length || 0} tokens)`, 'attack');
            }
        }
        
        function handleComplete(data) {
            const { asr, probe_bypass, duration } = data;
            
            document.getElementById('header-asr').textContent = `${(asr * 100).toFixed(0)}%`;
            document.getElementById('metric-asr').textContent = `${(asr * 100).toFixed(0)}%`;
            
            updateStatus('success', 'Attack complete!');
            log(`Attack pipeline complete! ASR: ${(asr * 100).toFixed(1)}%`, 'success');
            log(`Duration: ${duration?.toFixed(1) || '--'}s`, 'info');
        }
        
        // ========== VISUALIZATION FUNCTIONS ==========
        function drawLossChart() {
            const container = document.getElementById('loss-chart');
            const svg = d3.select('#loss-chart svg');
            svg.selectAll('*').remove();
            
            const rect = container.getBoundingClientRect();
            const width = rect.width;
            const height = rect.height;
            const margin = { top: 20, right: 20, bottom: 30, left: 50 };
            
            if (width <= 0 || height <= 0) return;
            
            const data = state.lossHistory;
            if (data.length === 0) return;
            
            const x = d3.scaleLinear()
                .domain([0, Math.max(data.length - 1, 1)])
                .range([margin.left, width - margin.right]);
            
            const y = d3.scaleLinear()
                .domain([0, d3.max(data) || 1])
                .nice()
                .range([height - margin.bottom, margin.top]);
            
            // Gradient
            const defs = svg.append('defs');
            const gradient = defs.append('linearGradient')
                .attr('id', 'lossGradient')
                .attr('x1', '0%').attr('y1', '0%')
                .attr('x2', '0%').attr('y2', '100%');
            gradient.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(88, 166, 255, 0.3)');
            gradient.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(88, 166, 255, 0)');
            
            // Area
            const area = d3.area()
                .x((d, i) => x(i))
                .y0(height - margin.bottom)
                .y1(d => y(d));
            
            svg.append('path')
                .datum(data)
                .attr('class', 'loss-area')
                .attr('d', area);
            
            // Line
            const line = d3.line()
                .x((d, i) => x(i))
                .y(d => y(d));
            
            svg.append('path')
                .datum(data)
                .attr('class', 'loss-line')
                .attr('d', line);
            
            // Axes
            svg.append('g')
                .attr('transform', `translate(0,${height - margin.bottom})`)
                .call(d3.axisBottom(x).ticks(5))
                .attr('color', '#30363d');
            
            svg.append('g')
                .attr('transform', `translate(${margin.left},0)`)
                .call(d3.axisLeft(y).ticks(4))
                .attr('color', '#30363d');
            
            document.getElementById('loss-count').textContent = `${data.length} points`;
        }
        
        function drawAttentionHeatmap(weights, tokens) {
            const container = document.getElementById('attention-heatmap');
            const svg = d3.select('#attention-heatmap svg');
            svg.selectAll('*').remove();
            
            const rect = container.getBoundingClientRect();
            const width = rect.width;
            const height = rect.height;
            const margin = { top: 30, right: 10, bottom: 10, left: 60 };
            
            if (width <= 0 || height <= 0 || !weights || weights.length === 0) return;
            
            const n = weights.length;
            const cellSize = Math.min(
                (width - margin.left - margin.right) / n,
                (height - margin.top - margin.bottom) / n
            );
            
            // Color scale
            const colorScale = d3.scaleSequential(d3.interpolateViridis)
                .domain([0, d3.max(weights.flat()) || 1]);
            
            // Draw cells
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    svg.append('rect')
                        .attr('class', 'attention-cell')
                        .attr('x', margin.left + j * cellSize)
                        .attr('y', margin.top + i * cellSize)
                        .attr('width', cellSize - 1)
                        .attr('height', cellSize - 1)
                        .attr('fill', colorScale(weights[i][j] || 0))
                        .attr('rx', 2);
                }
            }
            
            // Labels
            const labelTokens = tokens.slice(0, n);
            labelTokens.forEach((t, i) => {
                // Row labels
                svg.append('text')
                    .attr('x', margin.left - 5)
                    .attr('y', margin.top + i * cellSize + cellSize / 2)
                    .attr('text-anchor', 'end')
                    .attr('dominant-baseline', 'middle')
                    .attr('fill', '#8b949e')
                    .attr('font-size', '9px')
                    .text(t.slice(0, 6));
                
                // Column labels
                svg.append('text')
                    .attr('x', margin.left + i * cellSize + cellSize / 2)
                    .attr('y', margin.top - 5)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#8b949e')
                    .attr('font-size', '9px')
                    .attr('transform', `rotate(-45, ${margin.left + i * cellSize + cellSize / 2}, ${margin.top - 5})`)
                    .text(t.slice(0, 6));
            });
        }
        
        // ========== UI HELPERS ==========
        function activateStage(stageName) {
            // Remove active from all
            document.querySelectorAll('.stage-box').forEach(s => s.classList.remove('active'));
            // Add to current
            const stage = document.getElementById(`stage-${stageName}`);
            if (stage) stage.classList.add('active');
        }
        
        function animateFlowStages(step) {
            const stages = ['embedding', 'qkv', 'attention', 'mlp', 'output'];
            const stageIndex = step % stages.length;
            activateStage(stages[stageIndex]);
        }
        
        function updateStatus(type, message) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            
            dot.className = 'status-dot';
            if (type === 'connected') dot.classList.add('connected');
            else if (type === 'warning') dot.classList.add('warning');
            else if (type === 'error') dot.classList.add('error');
            else if (type === 'success') { dot.style.background = '#3fb950'; }
            
            text.textContent = message;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // ========== INITIALIZATION ==========
        function init() {
            log('Initializing visualization...', 'info');
            
            // Initialize charts
            drawLossChart();
            
            // Initialize layer flow with placeholder
            const layerContainer = document.getElementById('layer-flow');
            for (let i = 0; i < 6; i++) {
                createLayerRow(i);
            }
            
            // Initialize output probs placeholder
            const probContainer = document.getElementById('prob-list');
            probContainer.innerHTML = `
                <div class="prob-item">
                    <span class="prob-token">--</span>
                    <div class="prob-bar-bg"><div class="prob-bar" style="width: 0%"></div></div>
                    <span class="prob-value">--%</span>
                </div>
            `;
            
            // Window resize handler
            window.addEventListener('resize', () => {
                drawLossChart();
                if (state.attentionMatrix) {
                    drawAttentionHeatmap(state.attentionMatrix.weights, state.attentionMatrix.tokens);
                }
            });
            
            log('Ready for attack data', 'success');
        }
        
        // Start
        init();
    </script>
</body>
</html>'''

