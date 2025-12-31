"""
Provides real-time Sankey diagram visualization showing information flow
through transformer layers during adversarial attacks, including:
- Block input/output at each layer
- Attention input/output with Q/K/V decomposition
- MLP feed-forward network flow
- Residual connections
- Token probability predictions at each stage
"""


def get_flow_graph_html() -> str:
    """Return complete HTML for interactive flow graph visualization."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MIRA - Transformer Flow Analysis</title>
    
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a24;
            --accent-cyan: #00d4ff;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
            --accent-orange: #f97316;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #2d2d3a;
            --glow-cyan: 0 0 20px rgba(0, 212, 255, 0.3);
            --glow-purple: 0 0 20px rgba(139, 92, 246, 0.3);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 100%;
            margin: 0;
            padding: 20px 30px;
        }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 30px;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            box-shadow: var(--glow-cyan);
        }

        .logo-text {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .logo-subtitle {
            font-size: 12px;
            color: var(--text-secondary);
            letter-spacing: 2px;
            text-transform: uppercase;
        }

        /* Status Bar */
        .status-bar {
            display: flex;
            gap: 30px;
            align-items: center;
        }

        .status-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 10px 20px;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .status-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .status-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 20px;
            font-weight: 600;
            color: var(--accent-cyan);
        }

        .status-value.loss {
            color: var(--accent-yellow);
        }

        .status-value.success {
            color: var(--accent-green);
        }

        .status-value.events {
            color: var(--accent-purple);
        }

        /* Phase Progress Display */
        .phase-progress {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 12px 20px;
            background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(139,92,246,0.1));
            border-radius: 12px;
            border: 1px solid rgba(0,212,255,0.3);
            min-width: 280px;
        }

        .phase-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .phase-number {
            font-family: 'JetBrains Mono', monospace;
            font-size: 18px;
            font-weight: 700;
            color: var(--accent-cyan);
            background: var(--bg-tertiary);
            padding: 4px 10px;
            border-radius: 6px;
        }

        .phase-name {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .progress-bar-container {
            width: 100%;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            border-radius: 3px;
            transition: width 0.3s ease;
        }

        .phase-detail {
            font-size: 11px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Loading Hint Styles */
        .loading-hint {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            color: var(--text-muted);
            text-align: center;
            gap: 12px;
        }

        .loading-hint .spinner {
            width: 24px;
            height: 24px;
            border: 2px solid var(--border-color);
            border-top: 2px solid var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .loading-hint .hint-text {
            font-size: 12px;
            line-height: 1.5;
            max-width: 200px;
        }

        .loading-hint .hint-icon {
            font-size: 24px;
            opacity: 0.7;
            animation: pulse 2s infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 1; }
        }

        .loading-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(139,92,246,0.1));
            border: 1px solid rgba(0,212,255,0.3);
            border-radius: 20px;
            font-size: 10px;
            color: var(--accent-cyan);
            animation: pulse 2s infinite;
        }

        .loading-badge::before {
            content: '';
            width: 6px;
            height: 6px;
            background: var(--accent-cyan);
            border-radius: 50%;
            animation: pulse 1s infinite;
        }

        /* Main Layout - Full width responsive */
        .main-layout {
            display: grid;
            grid-template-columns: minmax(350px, 1fr) minmax(500px, 2fr) minmax(350px, 1fr);
            gap: 20px;
            min-height: calc(100vh - 200px);
            width: 100%;
        }

        /* Panel Base Styles */
        .panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .panel-header {
            padding: 15px 20px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .panel-icon {
            font-size: 18px;
        }

        .panel-title {
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .panel-description {
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 400;
            margin-left: 8px;
            opacity: 0.8;
            font-style: italic;
            text-transform: none;
            letter-spacing: 0;
        }

        .panel-content {
            padding: 20px;
            height: calc(100% - 60px);
            overflow-y: auto;
        }

        /* Left Panel - Token List & Attack Console */
        .left-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .token-list {
            flex: 0 0 auto;
            max-height: 250px;
        }

        .token-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            margin-bottom: 4px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
        }

        .token-item:hover {
            border-left-color: var(--accent-cyan);
            background: rgba(0, 212, 255, 0.1);
        }

        .token-item.adversarial {
            border-left-color: var(--accent-red);
            background: rgba(239, 68, 68, 0.1);
        }

        .token-index {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
            width: 24px;
        }

        .token-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--text-primary);
        }

        .token-prob {
            margin-left: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--accent-green);
        }

        /* Attack Console */
        .attack-console {
            flex: 0 0 auto;
            min-height: 200px;
            max-height: 350px;
            display: flex;
            flex-direction: column;
        }

        .console-output {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            line-height: 1.8;
            background: #000;
            padding: 15px;
            border-radius: 8px;
            flex: 1;
            max-height: 280px;
            overflow-y: auto;
            overflow-x: hidden;
        }

        .console-line {
            display: flex;
            gap: 10px;
            margin-bottom: 4px;
        }

        .console-time {
            color: var(--text-muted);
            min-width: 70px;
        }

        .console-type {
            font-weight: 600;
            min-width: 60px;
        }

        .console-type.attack {
            color: var(--accent-red);
        }

        .console-type.response {
            color: var(--accent-green);
        }

        .console-type.info {
            color: var(--accent-cyan);
        }

        .console-type.layer {
            color: var(--accent-purple);
        }

        .console-message {
            color: var(--text-secondary);
        }

        /* Center Panel - Flow Graph */
        .flow-graph-panel {
            display: flex;
            flex-direction: column;
        }

        #flow-graph {
            width: 100%;
            height: 100%;
            min-height: 600px;
        }

        .layer-selector {
            display: flex;
            gap: 8px;
            padding: 10px 20px;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border-color);
            overflow-x: auto;
        }

        .layer-btn {
            padding: 8px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
            min-width: 80px;
        }

        .layer-btn .layer-number {
            font-size: 14px;
            font-weight: 600;
        }

        .layer-btn .layer-label {
            font-size: 10px;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
        }

        .layer-btn:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
        }

        .layer-btn.active {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            color: var(--bg-primary);
            border-color: var(--accent-cyan);
            box-shadow: var(--glow-cyan);
        }

        .layer-btn.active .layer-label {
            opacity: 1;
            font-weight: 600;
        }

        /* Layer tooltip */
        .layer-btn::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(-8px);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 11px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease, transform 0.2s ease;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 1000;
            margin-bottom: 5px;
        }

        .layer-btn:hover::after {
            opacity: 1;
            transform: translateX(-50%) translateY(-12px);
        }

        /* Right Panel - Analysis */
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Attention Heatmap */
        .attention-heatmap {
            flex: 0 0 auto;
        }

        #attention-canvas {
            width: 100%;
            height: 200px;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }

        /* Layer Activations */
        .layer-activations {
            flex: 1;
        }

        .activation-layer {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            margin-bottom: 8px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
        }

        .activation-layer:hover {
            border-left-color: var(--accent-cyan);
            background: rgba(0, 212, 255, 0.05);
        }

        .activation-layer:last-child {
            border-bottom: none;
        }

        .layer-info {
            display: flex;
            flex-direction: column;
            min-width: 100px;
            gap: 3px;
        }

        .layer-info .layer-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .layer-info .layer-description {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 500;
        }

        .activation-bars {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .activation-bar-container {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .bar-label {
            font-size: 10px;
            color: var(--text-muted);
            width: 50px;
        }

        .bar-track {
            flex: 1;
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .bar-fill.refusal {
            background: linear-gradient(90deg, var(--accent-red), var(--accent-orange));
        }

        .bar-fill.acceptance {
            background: linear-gradient(90deg, var(--accent-green), var(--accent-cyan));
        }

        .bar-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-secondary);
            width: 40px;
            text-align: right;
        }

        /* Output Probabilities */
        .output-probs {
            flex: 0 0 auto;
        }

        .prob-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            margin-bottom: 6px;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }

        .prob-rank {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--text-muted);
            width: 20px;
        }

        .prob-token {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--text-primary);
            flex: 1;
        }

        .prob-bar {
            width: 80px;
            height: 6px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
        }

        .prob-bar-fill {
            height: 100%;
            background: var(--accent-purple);
            transition: width 0.3s ease;
        }

        .prob-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--accent-purple);
            width: 50px;
            text-align: right;
        }

        /* Adversarial Suffix Display */
        .suffix-display {
            margin-top: 20px;
            padding: 15px;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
        }

        .suffix-label {
            font-size: 11px;
            color: var(--accent-red);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .suffix-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--text-primary);
            word-break: break-all;
        }

        /* Prompt & Response Panel */
        .prompt-response {
            flex: 1;
            min-height: 200px;
            max-height: 350px;
        }

        .prompt-section,
        .response-section {
            margin-bottom: 15px;
        }

        .section-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--accent-cyan);
            margin-bottom: 8px;
        }

        .prompt-text,
        .response-text {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--text-primary);
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            max-height: 100px;
            overflow-y: auto;
            line-height: 1.6;
            word-break: break-word;
        }

        .response-status {
            margin-top: 8px;
            font-size: 12px;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 4px;
            display: inline-block;
        }

        .response-status.success {
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
        }

        .response-status.failed {
            background: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
        }

        /* Delta indicator */
        .delta-indicator {
            font-size: 10px;
            margin-left: 4px;
            padding: 2px 4px;
            border-radius: 3px;
        }

        .delta-indicator.positive {
            color: var(--accent-red);
            background: rgba(239, 68, 68, 0.2);
        }

        .delta-indicator.negative {
            color: var(--accent-green);
            background: rgba(16, 185, 129, 0.2);
        }

        /* Pattern alert */
        .pattern-alert {
            padding: 12px;
            margin-top: 10px;
            background: rgba(139, 92, 246, 0.2);
            border: 1px solid var(--accent-purple);
            border-radius: 8px;
            font-size: 12px;
            color: var(--accent-purple);
        }

        .pattern-alert .pattern-type {
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Scrollbar Styling */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* Animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .processing {
            animation: pulse 1s ease-in-out infinite;
        }

        /* Connection Status */
        .connection-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-secondary);
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

        /* Multi-Head Selector */
        .head-selector {
            display: flex;
            gap: 6px;
            padding: 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }

        .head-btn {
            padding: 4px 10px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            color: var(--text-secondary);
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .head-btn:hover {
            border-color: var(--accent-purple);
            color: var(--accent-purple);
        }

        .head-btn.active {
            background: var(--accent-purple);
            color: var(--bg-primary);
            border-color: var(--accent-purple);
        }

        /* Layer Predictions Evolution */
        .layer-predictions {
            flex: 0 0 auto;
            max-height: 300px;
        }

        .prediction-row {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            margin-bottom: 4px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border-left: 3px solid transparent;
        }

        .prediction-row.changed {
            border-left-color: var(--accent-yellow);
            background: rgba(245, 158, 11, 0.1);
        }

        .prediction-layer {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: var(--text-muted);
            width: 30px;
        }

        .prediction-token {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--text-primary);
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .prediction-change {
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 3px;
            background: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
        }

        .prediction-change.negative {
            background: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
        }

        /* SSR Buffer Display */
        .ssr-buffer {
            flex: 0 0 auto;
        }

        .buffer-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
            gap: 6px;
        }

        .buffer-item {
            padding: 8px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border: 1px solid var(--border-color);
            text-align: center;
            transition: all 0.2s ease;
        }

        .buffer-item.best {
            border-color: var(--accent-green);
            background: rgba(16, 185, 129, 0.1);
        }

        .buffer-item.current {
            border-color: var(--accent-cyan);
            background: rgba(0, 212, 255, 0.1);
        }

        .buffer-rank {
            font-size: 10px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .buffer-tokens {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-primary);
            margin-bottom: 4px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .buffer-loss {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            color: var(--accent-yellow);
        }

        .buffer-stats {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-top: 10px;
        }

        .buffer-stat {
            text-align: center;
        }

        .buffer-stat-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .buffer-stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            font-weight: 600;
            color: var(--accent-cyan);
        }

        /* Collapsible panels */
        .panel-toggle {
            margin-left: auto;
            cursor: pointer;
            color: var(--text-muted);
            transition: transform 0.2s ease;
        }

        .panel-toggle.collapsed {
            transform: rotate(-90deg);
        }

        .panel-content.collapsed {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">
            <div class="logo-icon">M</div>
            <div>
                <div class="logo-text">MIRA Flow Analyzer</div>
                <div class="logo-subtitle">Transformer Internals Visualization</div>
            </div>
        </div>
        
        <!-- Phase Progress Display -->
        <div class="phase-progress">
            <div class="phase-info">
                <span class="phase-number" id="phase-number">--/--</span>
                <span class="phase-name" id="phase-name">Initializing...</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
            </div>
            <div class="phase-detail" id="phase-detail">Waiting for connection...</div>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <span class="status-label">Step</span>
                <span class="status-value" id="current-step">0</span>
            </div>
            <div class="status-item">
                <span class="status-label">Loss</span>
                <span class="status-value loss" id="current-loss">--</span>
            </div>
            <div class="status-item">
                <span class="status-label">Best</span>
                <span class="status-value success" id="best-loss">--</span>
            </div>
            <div class="status-item">
                <span class="status-label">Events</span>
                <span class="status-value events" id="event-count">0</span>
            </div>
            <div class="connection-indicator">
                <div class="connection-dot disconnected" id="connection-dot"></div>
                <span id="connection-status">Connecting...</span>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="main-layout">
            <!-- Left Panel -->
            <div class="left-panel">
                <div class="panel token-list">
                    <div class="panel-header">
                        <span class="panel-icon">📝</span>
                        <span class="panel-title">Input Tokens</span>
                        <span class="panel-description">Shows input text broken down into vocabulary tokens</span>
                    </div>
                    <div class="panel-content" id="token-container">
                        <div class="loading-hint">
                            <div class="spinner"></div>
                            <span class="hint-text">⏳ Tokens will appear when attack starts</span>
                        </div>
                    </div>
                </div>

                <div class="panel attack-console">
                    <div class="panel-header">
                        <span class="panel-icon">⚡</span>
                        <span class="panel-title">Attack Console</span>
                        <span class="panel-description">Real-time log messages during attack execution</span>
                    </div>
                    <div class="panel-content">
                        <div class="console-output" id="console-output">
                            <div class="console-line">
                                <span class="console-time">00:00:00</span>
                                <span class="console-type info">[INFO]</span>
                                <span class="console-message">MIRA Attack System Initialized</span>
                            </div>
                            <div class="console-line">
                                <span class="console-time">00:00:00</span>
                                <span class="console-type info">[INFO]</span>
                                <span class="console-message">Waiting for attack execution...</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="panel prompt-response">
                    <div class="panel-header">
                        <span class="panel-icon">💬</span>
                        <span class="panel-title">Prompt & Response</span>
                        <span class="panel-description">Shows current attack prompt and model's complete response</span>
                    </div>
                    <div class="panel-content" id="prompt-response-content">
                        <div class="prompt-section">
                            <div class="section-label">Current Prompt</div>
                            <div class="prompt-text" id="current-prompt">
                                <span class="loading-badge">⏳ Waiting for attack</span>
                            </div>
                        </div>
                        <div class="response-section">
                            <div class="section-label">Model Response</div>
                            <div class="response-text" id="current-response">
                                <span style="color: var(--text-muted);">Response will appear after attack completes</span>
                            </div>
                            <div class="response-status" id="response-status">--</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Center Panel - Flow Graph -->
            <div class="panel flow-graph-panel">
                <div class="panel-header">
                    <span class="panel-icon">🔄</span>
                    <span class="panel-title">Transformer Flow Graph</span>
                    <span class="panel-description">Visualizes information flow path through model layers</span>
                </div>
                <div class="panel-content" style="padding: 0;">
                    <div id="flow-graph"></div>
                </div>
                <div class="layer-selector" id="layer-selector">
                    <!-- Layer buttons will be generated dynamically -->
                </div>
            </div>

            <!-- Right Panel -->
            <div class="right-panel">
                <div class="panel attention-heatmap">
                    <div class="panel-header">
                        <span class="panel-icon">🎯</span>
                        <span class="panel-title">Attention Pattern</span>
                        <span class="panel-description">Shows model attention weights, which tokens attend to each other</span>
                        <span class="panel-toggle" onclick="togglePanel(this)">▼</span>
                    </div>
                    <div class="panel-content">
                        <!-- Multi-Head Selector -->
                        <div class="head-selector" id="head-selector">
                            <button class="head-btn active" data-head="0">H0</button>
                            <button class="head-btn" data-head="1">H1</button>
                            <button class="head-btn" data-head="2">H2</button>
                            <button class="head-btn" data-head="3">H3</button>
                            <button class="head-btn" data-head="4">H4</button>
                            <button class="head-btn" data-head="5">H5</button>
                            <button class="head-btn" data-head="6">H6</button>
                            <button class="head-btn" data-head="7">H7</button>
                        </div>
                        <canvas id="attention-canvas"></canvas>
                    </div>
                </div>

                <!-- Layer Predictions Evolution -->
                <div class="panel layer-predictions">
                    <div class="panel-header">
                        <span class="panel-icon">📈</span>
                        <span class="panel-title">Layer Predictions</span>
                        <span class="panel-description">Shows next token predictions at each layer and probability changes</span>
                        <span class="panel-toggle" onclick="togglePanel(this)">▼</span>
                    </div>
                    <div class="panel-content" id="layer-predictions-content">
                        <div class="loading-hint">
                            <div class="spinner"></div>
                            <span class="hint-text">📊 Layer predictions will update during analysis (may take time on CPU)</span>
                        </div>
                    </div>
                </div>

                <div class="panel layer-activations">
                    <div class="panel-header">
                        <span class="panel-icon">📊</span>
                        <span class="panel-title">Layer Analysis</span>
                        <span class="panel-description">Analyzes refusal and acceptance activation strength at each layer</span>
                        <span class="panel-toggle" onclick="togglePanel(this)">▼</span>
                    </div>
                    <div class="panel-content" id="layer-activations">
                        <!-- Layer activation bars will be generated dynamically -->
                    </div>
                </div>

                <!-- SSR Buffer Display -->
                <div class="panel ssr-buffer" id="ssr-buffer-panel" style="display: none;">
                    <div class="panel-header">
                        <span class="panel-icon">🔄</span>
                        <span class="panel-title">SSR Buffer</span>
                        <span class="panel-description">Shows candidate suffix buffer for sparse sampling replacement attack</span>
                        <span class="panel-toggle" onclick="togglePanel(this)">▼</span>
                    </div>
                    <div class="panel-content">
                        <div class="buffer-grid" id="buffer-grid">
                            <!-- Buffer items will be generated dynamically -->
                        </div>
                        <div class="buffer-stats">
                            <div class="buffer-stat">
                                <div class="buffer-stat-label">Buffer Size</div>
                                <div class="buffer-stat-value" id="buffer-size">0</div>
                            </div>
                            <div class="buffer-stat">
                                <div class="buffer-stat-label">Best Loss</div>
                                <div class="buffer-stat-value" id="buffer-best-loss">--</div>
                            </div>
                            <div class="buffer-stat">
                                <div class="buffer-stat-label">N Replace</div>
                                <div class="buffer-stat-value" id="buffer-n-replace">--</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="panel output-probs">
                    <div class="panel-header">
                        <span class="panel-icon">🎲</span>
                        <span class="panel-title">Output Predictions</span>
                        <span class="panel-description">Shows model's next token predictions and probability distribution</span>
                    </div>
                    <div class="panel-content">
                        <div id="output-probs-container">
                            <!-- Output probabilities will be generated dynamically -->
                        </div>
                        <div class="suffix-display">
                            <div class="suffix-label">Adversarial Suffix</div>
                            <div class="suffix-text" id="suffix-text">--</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State management
        const state = {
            step: 0,
            loss: null,
            bestLoss: Infinity,
            events: 0,
            tokens: [],
            currentLayer: 0,
            currentHead: 0,
            numHeads: 8,
            flowData: null,
            attentionMatrix: null,
            allAttentionMatrices: {},
            layerActivations: {},
            layerPredictions: {},
            outputProbs: [],
            consoleLines: [],
            startTime: Date.now(),
            asr: 0,
            totalAttacks: 0,
            successfulAttacks: 0,
            currentPrompt: '',
            currentResponse: '',
            ssrBuffer: [],
            ssrBestLoss: Infinity,
            ssrNReplace: 0,
            isSSRMode: false
        };

        // Initialize flow graph with empty Sankey
        function initFlowGraph() {
            const data = [{
                type: 'sankey',
                orientation: 'h',
                node: {
                    pad: 15,
                    thickness: 20,
                    line: { color: 'black', width: 0.5 },
                    label: ['Input', 'Embedding', 'Attention', 'MLP', 'Output'],
                    color: ['#00d4ff', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
                },
                link: {
                    source: [0, 1, 2, 3],
                    target: [1, 2, 3, 4],
                    value: [1, 1, 1, 1],
                    color: ['rgba(0,212,255,0.4)', 'rgba(139,92,246,0.4)', 'rgba(16,185,129,0.4)', 'rgba(245,158,11,0.4)']
                }
            }];

            const layout = {
                title: {
                    text: 'Transformer Processing Flow',
                    font: { size: 16, color: '#e2e8f0' }
                },
                font: { size: 11, color: '#94a3b8' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { t: 50, l: 20, r: 20, b: 20 }
            };

            const config = {
                responsive: true,
                displayModeBar: false
            };

            Plotly.newPlot('flow-graph', data, layout, config);
        }

        // Update flow graph with detailed Sankey data
        function updateFlowGraph(flowData) {
            if (!flowData || !flowData.nodes || !flowData.links) {
                console.log('Invalid flow data, skipping update');
                return;
            }

            const data = [{
                type: 'sankey',
                orientation: 'h',
                valueformat: '.3f',
                node: {
                    pad: 15,
                    thickness: 20,
                    line: { color: '#1a1a24', width: 0.5 },
                    label: flowData.nodes.map(n => n.label),
                    color: flowData.nodes.map(n => n.color || '#64748b'),
                    customdata: flowData.nodes.map(n => n.customdata || ''),
                    hovertemplate: '%{label}<br>%{customdata}<extra></extra>'
                },
                link: {
                    source: flowData.links.map(l => l.source),
                    target: flowData.links.map(l => l.target),
                    value: flowData.links.map(l => l.value),
                    color: flowData.links.map(l => l.color || 'rgba(100,116,139,0.4)'),
                    customdata: flowData.links.map(l => l.customdata || ''),
                    hovertemplate: '%{customdata}<br>Weight: %{value:.3f}<extra></extra>'
                }
            }];

            const layout = {
                title: {
                    text: `Layer ${state.currentLayer} - Step ${state.step}`,
                    font: { size: 16, color: '#e2e8f0' }
                },
                font: { size: 10, color: '#94a3b8' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { t: 50, l: 20, r: 20, b: 20 },
                hovermode: 'x'
            };

            Plotly.react('flow-graph', data, layout);
        }

        // Generate default flow graph for a layer
        function generateLayerFlowGraph(layerIdx, layerData) {
            // Create nodes for this layer
            const nodes = [
                { label: `L${layerIdx} Input`, color: '#00d4ff', customdata: 'Block input' },
                { label: 'LN1', color: '#64748b', customdata: 'Layer Norm 1' },
                { label: 'Q', color: '#8b5cf6', customdata: 'Query projection' },
                { label: 'K', color: '#8b5cf6', customdata: 'Key projection' },
                { label: 'V', color: '#8b5cf6', customdata: 'Value projection' },
                { label: 'Attn Out', color: '#10b981', customdata: 'Attention output' },
                { label: 'LN2', color: '#64748b', customdata: 'Layer Norm 2' },
                { label: 'MLP FC1', color: '#f59e0b', customdata: 'Feed-forward layer 1' },
                { label: 'MLP FC2', color: '#f59e0b', customdata: 'Feed-forward layer 2' },
                { label: `L${layerIdx} Output`, color: '#ef4444', customdata: 'Block output' }
            ];

            // Add head nodes if we have attention data
            if (layerData && layerData.heads) {
                layerData.heads.forEach((head, idx) => {
                    nodes.push({
                        label: `H${idx}`,
                        color: head.color || '#6366f1',
                        customdata: `Head ${idx}: attn_score=${head.score?.toFixed(3) || '--'}`
                    });
                });
            }

            // Create links
            const links = [
                { source: 0, target: 1, value: 1, color: 'rgba(0,212,255,0.4)', customdata: 'Input -> LN1' },
                { source: 1, target: 2, value: 0.33, color: 'rgba(139,92,246,0.4)', customdata: 'LN1 -> Q' },
                { source: 1, target: 3, value: 0.33, color: 'rgba(139,92,246,0.4)', customdata: 'LN1 -> K' },
                { source: 1, target: 4, value: 0.33, color: 'rgba(139,92,246,0.4)', customdata: 'LN1 -> V' },
                { source: 2, target: 5, value: 0.5, color: 'rgba(16,185,129,0.4)', customdata: 'Q -> Attn' },
                { source: 3, target: 5, value: 0.25, color: 'rgba(16,185,129,0.4)', customdata: 'K -> Attn' },
                { source: 4, target: 5, value: 0.25, color: 'rgba(16,185,129,0.4)', customdata: 'V -> Attn' },
                { source: 0, target: 5, value: 0.5, color: 'rgba(100,116,139,0.2)', customdata: 'Residual' },
                { source: 5, target: 6, value: 1, color: 'rgba(100,116,139,0.4)', customdata: 'Attn -> LN2' },
                { source: 6, target: 7, value: 1, color: 'rgba(245,158,11,0.4)', customdata: 'LN2 -> FC1' },
                { source: 7, target: 8, value: 1, color: 'rgba(245,158,11,0.4)', customdata: 'FC1 -> FC2' },
                { source: 5, target: 9, value: 0.5, color: 'rgba(100,116,139,0.2)', customdata: 'Residual' },
                { source: 8, target: 9, value: 1, color: 'rgba(239,68,68,0.4)', customdata: 'FC2 -> Output' }
            ];

            return { nodes, links };
        }

        // Update attention heatmap
        function updateAttentionHeatmap(matrix, tokens) {
            const canvas = document.getElementById('attention-canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;

            if (!matrix || matrix.length === 0) {
                ctx.fillStyle = '#1a1a24';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw loading indicator
                ctx.fillStyle = '#00d4ff';
                ctx.font = '14px Inter';
                ctx.textAlign = 'center';
                ctx.fillText('⏳ Attention Pattern', canvas.width / 2, canvas.height / 2 - 20);
                ctx.fillStyle = '#64748b';
                ctx.font = '11px Inter';
                ctx.fillText('Data will appear when attack runs', canvas.width / 2, canvas.height / 2 + 5);
                ctx.fillText('(may take time on CPU)', canvas.width / 2, canvas.height / 2 + 22);
                return;
            }

            const n = matrix.length;
            const cellWidth = (canvas.width - 40) / n;
            const cellHeight = (canvas.height - 40) / n;

            // Draw cells
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const val = matrix[i][j];
                    const intensity = Math.min(255, Math.floor(val * 255));
                    ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.3)}, ${255 - intensity})`;
                    ctx.fillRect(40 + j * cellWidth, 20 + i * cellHeight, cellWidth - 1, cellHeight - 1);
                }
            }

            // Draw labels
            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px JetBrains Mono';
            tokens = tokens || [];
            for (let i = 0; i < n; i++) {
                const label = tokens[i] ? tokens[i].slice(0, 3) : i.toString();
                ctx.save();
                ctx.translate(35, 20 + i * cellHeight + cellHeight / 2);
                ctx.textAlign = 'right';
                ctx.fillText(label, 0, 3);
                ctx.restore();
            }
        }

        // Update layer activations display with delta comparison
        function updateLayerActivations() {
            const container = document.getElementById('layer-activations');
            let html = '';

            for (let i = 0; i < 12; i++) {
                const data = state.layerActivations[i] || { refusal: 0, acceptance: 0, delta_refusal: 0, delta_acceptance: 0 };
                
                // Format delta indicator
                const deltaRef = data.delta_refusal || 0;
                const deltaAcc = data.delta_acceptance || 0;
                
                const refDeltaHtml = deltaRef !== 0 
                    ? `<span class="delta-indicator ${deltaRef > 0 ? 'positive' : 'negative'}">${deltaRef > 0 ? '+' : ''}${(deltaRef * 100).toFixed(1)}%</span>` 
                    : '';
                const accDeltaHtml = deltaAcc !== 0 
                    ? `<span class="delta-indicator ${deltaAcc > 0 ? 'positive' : 'negative'}">${deltaAcc > 0 ? '+' : ''}${(deltaAcc * 100).toFixed(1)}%</span>` 
                    : '';
                
                const description = getLayerDescription(i, 12);
                html += `
                    <div class="activation-layer">
                        <div class="layer-info">
                            <span class="layer-label">L${i}</span>
                            <span class="layer-description">${description}</span>
                        </div>
                        <div class="activation-bars">
                            <div class="activation-bar-container">
                                <span class="bar-label">Refusal</span>
                                <div class="bar-track">
                                    <div class="bar-fill refusal" style="width: ${data.refusal * 100}%"></div>
                                </div>
                                <span class="bar-value">${(data.refusal * 100).toFixed(1)}%${refDeltaHtml}</span>
                            </div>
                            <div class="activation-bar-container">
                                <span class="bar-label">Accept</span>
                                <div class="bar-track">
                                    <div class="bar-fill acceptance" style="width: ${data.acceptance * 100}%"></div>
                                </div>
                                <span class="bar-value">${(data.acceptance * 100).toFixed(1)}%${accDeltaHtml}</span>
                            </div>
                        </div>
                    </div>
                `;
            }

            container.innerHTML = html;
        }

        // Update output probabilities
        function updateOutputProbs(probs) {
            const container = document.getElementById('output-probs-container');
            let html = '';

            probs = probs || [];
            probs.slice(0, 5).forEach((p, i) => {
                html += `
                    <div class="prob-item">
                        <span class="prob-rank">${i + 1}</span>
                        <span class="prob-token">${p.token || '--'}</span>
                        <div class="prob-bar">
                            <div class="prob-bar-fill" style="width: ${(p.prob || 0) * 100}%"></div>
                        </div>
                        <span class="prob-value">${((p.prob || 0) * 100).toFixed(1)}%</span>
                    </div>
                `;
            });

            if (probs.length === 0) {
                html = '<div class="loading-hint"><div class="spinner"></div><span class="hint-text">🔮 Predictions will appear during attack</span></div>';
            }

            container.innerHTML = html;
        }

        // Update tokens display
        function updateTokens(tokens, adversarialStart) {
            const container = document.getElementById('token-container');
            let html = '';

            tokens = tokens || [];
            adversarialStart = adversarialStart || tokens.length;

            tokens.forEach((token, i) => {
                const isAdversarial = i >= adversarialStart;
                html += `
                    <div class="token-item ${isAdversarial ? 'adversarial' : ''}">
                        <span class="token-index">${i}</span>
                        <span class="token-text">${token}</span>
                    </div>
                `;
            });

            if (tokens.length === 0) {
                html = '<div class="loading-hint"><div class="spinner"></div><span class="hint-text">⏳ Tokens will appear when attack starts</span></div>';
            }

            container.innerHTML = html;
        }

        // Add console line
        function addConsoleLine(type, message) {
            const now = new Date();
            const elapsed = Math.floor((now - state.startTime) / 1000);
            const hours = String(Math.floor(elapsed / 3600)).padStart(2, '0');
            const mins = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
            const secs = String(elapsed % 60).padStart(2, '0');
            const time = `${hours}:${mins}:${secs}`;

            const line = { time, type, message };
            state.consoleLines.push(line);

            // Keep only last 50 lines for performance
            const maxLines = 50;
            if (state.consoleLines.length > maxLines) {
                state.consoleLines = state.consoleLines.slice(-maxLines);
            }

            // Re-render console with limited lines
            const container = document.getElementById('console-output');
            container.innerHTML = state.consoleLines.map(l => `
                <div class="console-line">
                    <span class="console-time">${l.time}</span>
                    <span class="console-type ${l.type.toLowerCase()}">[${l.type}]</span>
                    <span class="console-message">${l.message}</span>
                </div>
            `).join('');
            container.scrollTop = container.scrollHeight;
        }

        // Toggle panel collapse
        function togglePanel(toggle) {
            toggle.classList.toggle('collapsed');
            // Find the panel-content sibling more reliably
            const panel = toggle.closest('.panel');
            if (panel) {
                const content = panel.querySelector('.panel-content');
                if (content) {
                    content.classList.toggle('collapsed');
                }
            }
        }

        // Initialize multi-head selector
        function initHeadSelector() {
            const container = document.getElementById('head-selector');
            container.querySelectorAll('.head-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    container.querySelectorAll('.head-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    state.currentHead = parseInt(btn.dataset.head);
                    
                    // Update attention with selected head
                    const key = `${state.currentLayer}_${state.currentHead}`;
                    const matrix = state.allAttentionMatrices[key];
                    if (matrix) {
                        updateAttentionHeatmap(matrix.weights, matrix.tokens);
                    }
                    addConsoleLine('INFO', `Switched to head ${state.currentHead}`);
                });
            });
        }

        // Update layer predictions display
        function updateLayerPredictions() {
            const container = document.getElementById('layer-predictions-content');
            const predictions = state.layerPredictions;
            
            if (Object.keys(predictions).length === 0) {
                container.innerHTML = '<div class="loading-hint"><div class="spinner"></div><span class="hint-text">📊 Layer predictions will update during analysis (may take time on CPU)</span></div>';
                return;
            }
            
            let html = '';
            let prevToken = null;
            
            // Sort by layer index
            const sortedLayers = Object.keys(predictions).map(k => parseInt(k)).sort((a, b) => a - b);
            
            sortedLayers.forEach(layerIdx => {
                const pred = predictions[layerIdx];
                const token = pred.token || '--';
                const prob = pred.prob || 0;
                const changed = prevToken !== null && prevToken !== token;
                
                const changeClass = changed ? 'changed' : '';
                const probDelta = pred.delta || 0;
                const deltaHtml = probDelta !== 0 
                    ? `<span class="prediction-change ${probDelta < 0 ? 'negative' : ''}">${probDelta > 0 ? '+' : ''}${(probDelta * 100).toFixed(1)}%</span>` 
                    : '';
                
                html += `
                    <div class="prediction-row ${changeClass}">
                        <span class="prediction-layer">L${layerIdx}</span>
                        <span class="prediction-token">${token}</span>
                        <span style="color: var(--accent-purple); font-size: 11px;">${(prob * 100).toFixed(1)}%</span>
                        ${deltaHtml}
                    </div>
                `;
                
                prevToken = token;
            });
            
            container.innerHTML = html;
        }

        // Update SSR buffer display
        function updateSSRBuffer() {
            const panel = document.getElementById('ssr-buffer-panel');
            const grid = document.getElementById('buffer-grid');
            
            if (!state.isSSRMode || state.ssrBuffer.length === 0) {
                panel.style.display = 'none';
                return;
            }
            
            panel.style.display = 'block';
            
            let html = '';
            state.ssrBuffer.slice(0, 6).forEach((item, i) => {
                const itemClass = i === 0 ? 'best' : (i === 1 ? 'current' : '');
                html += `
                    <div class="buffer-item ${itemClass}">
                        <div class="buffer-rank">#${i + 1}</div>
                        <div class="buffer-tokens">${item.tokens || '--'}</div>
                        <div class="buffer-loss">${item.loss?.toFixed(4) || '--'}</div>
                    </div>
                `;
            });
            
            grid.innerHTML = html;
            
            // Update stats
            document.getElementById('buffer-size').textContent = state.ssrBuffer.length;
            document.getElementById('buffer-best-loss').textContent = state.ssrBestLoss !== Infinity 
                ? state.ssrBestLoss.toFixed(4) 
                : '--';
            document.getElementById('buffer-n-replace').textContent = state.ssrNReplace || '--';
        }

        // Layer descriptions based on position with detailed meanings
        function getLayerDescription(layerIdx, totalLayers) {
            const ratio = layerIdx / Math.max(totalLayers - 1, 1);
            
            if (ratio < 0.2) {
                return "Input";
            } else if (ratio < 0.4) {
                return "Early";
            } else if (ratio < 0.6) {
                return "Mid";
            } else if (ratio < 0.8) {
                return "Late";
            } else {
                return "Output";
            }
        }

        // Get detailed layer meaning for tooltips
        function getLayerMeaning(layerIdx, totalLayers) {
            const ratio = layerIdx / Math.max(totalLayers - 1, 1);
            
            if (layerIdx === 0) {
                return "Token embeddings - Initial input representation";
            } else if (ratio < 0.2) {
                return "Input Processing - Early feature extraction";
            } else if (ratio < 0.4) {
                return "Early Features - Building semantic understanding";
            } else if (ratio < 0.6) {
                return "Mid Processing - Core reasoning and pattern matching";
            } else if (ratio < 0.8) {
                return "Late Features - High-level semantic integration";
            } else if (layerIdx === totalLayers - 1) {
                return "Output Preparation - Final representation before prediction";
            } else {
                return "Output Processing - Preparing final response";
            }
        }

        // Initialize layer selector with descriptions
        function initLayerSelector(numLayers) {
            const container = document.getElementById('layer-selector');
            let html = '';

            for (let i = 0; i < numLayers; i++) {
                const description = getLayerDescription(i, numLayers);
                const meaning = getLayerMeaning(i, numLayers);
                const tooltip = `Layer ${i}: ${meaning}`;
                
                html += `
                    <button class="layer-btn ${i === 0 ? 'active' : ''}" 
                            data-layer="${i}" 
                            data-tooltip="${tooltip}">
                        <span class="layer-number">L${i}</span>
                        <span class="layer-label">${description}</span>
                    </button>
                `;
            }

            container.innerHTML = html;

            // Add click handlers
            container.querySelectorAll('.layer-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    container.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    state.currentLayer = parseInt(btn.dataset.layer);
                    updateFlowGraph(generateLayerFlowGraph(state.currentLayer, null));
                    
                    // Also update attention for new layer
                    const key = `${state.currentLayer}_${state.currentHead}`;
                    const matrix = state.allAttentionMatrices[key];
                    if (matrix) {
                        updateAttentionHeatmap(matrix.weights, matrix.tokens);
                    }
                });
            });
        }

        // Handle SSE events
        function handleEvent(event) {
            state.events++;
            document.getElementById('event-count').textContent = state.events;

            const data = event.data;
            const type = event.event_type || data.type;

            switch (type) {
                case 'attack_step':
                    state.step = data.step || 0;
                    
                    // Handle loss value - 0.0 is a valid loss value, so we check for undefined/null
                    if (data.loss !== undefined && data.loss !== null && !isNaN(data.loss) && isFinite(data.loss)) {
                        state.loss = data.loss;
                        
                        // Update best loss (0.0 is valid, so we compare normally)
                        if (state.bestLoss === Infinity || state.loss < state.bestLoss) {
                            state.bestLoss = state.loss;
                            document.getElementById('best-loss').textContent = state.bestLoss.toFixed(4);
                        }
                        
                        // Always display current loss, even if it's 0.0
                        document.getElementById('current-loss').textContent = state.loss.toFixed(4);
                    } else {
                        // If loss is truly not available, show '--' and don't update best
                        state.loss = null;
                        document.getElementById('current-loss').textContent = '--';
                        // Only update best loss if we have a valid loss value
                    }
                    
                    document.getElementById('current-step').textContent = state.step;
                    document.getElementById('suffix-text').textContent = data.suffix || '--';
                    
                    // Update current prompt
                    if (data.prompt) {
                        document.getElementById('current-prompt').textContent = data.prompt;
                        state.currentPrompt = data.prompt;
                    }
                    
                    // Update model response if available in attack_step event
                    if (data.response && data.response !== '--' && data.response.trim() !== '') {
                        const responseEl = document.getElementById('current-response');
                        if (responseEl) {
                            responseEl.textContent = data.response.length > 500 ? data.response.slice(0, 500) + '...' : data.response;
                        }
                        const statusEl = document.getElementById('response-status');
                        if (statusEl) {
                            const isSuccess = data.success || false;
                            statusEl.textContent = isSuccess ? 'SUCCESS - Attack Bypassed Safety' : 'FAILED - Model Refused';
                            statusEl.className = 'response-status ' + (isSuccess ? 'success' : 'failed');
                        }
                        addConsoleLine('RESPONSE', `Model response: ${data.response.slice(0, 100)}${data.response.length > 100 ? '...' : ''}`);
                    }
                    
                    const lossStr = state.loss !== null && !isNaN(state.loss) && isFinite(state.loss) 
                        ? state.loss.toFixed(4) 
                        : 'N/A';
                    const bestStr = state.bestLoss !== Infinity && !isNaN(state.bestLoss) && isFinite(state.bestLoss)
                        ? state.bestLoss.toFixed(4)
                        : 'N/A';
                    addConsoleLine('ATTACK', `Step ${state.step}: loss=${lossStr}, best=${bestStr}`);
                    break;

                case 'embeddings':
                    state.tokens = data.tokens || [];
                    updateTokens(state.tokens, data.adversarial_start);
                    addConsoleLine('INFO', `Received ${state.tokens.length} tokens`);
                    break;

                case 'flow_graph':
                    state.flowData = data;
                    updateFlowGraph(data);
                    addConsoleLine('LAYER', `Flow graph updated for layer ${data.layer || state.currentLayer}`);
                    break;

                case 'attention':
                case 'attention_matrix':
                    const attnMatrix = data.matrix || data.weights;
                    const attnLayer = data.layer || 0;
                    const attnHead = data.head || 0;
                    
                    // Store in multi-head cache
                    const attnKey = `${attnLayer}_${attnHead}`;
                    state.allAttentionMatrices[attnKey] = {
                        weights: attnMatrix,
                        tokens: data.tokens
                    };
                    
                    // Update display if current layer/head matches
                    if (attnLayer === state.currentLayer && attnHead === state.currentHead) {
                        state.attentionMatrix = attnMatrix;
                        updateAttentionHeatmap(attnMatrix, data.tokens);
                    }
                    addConsoleLine('INFO', `Attention matrix updated for L${attnLayer}H${attnHead}`);
                    break;

                case 'layer':
                case 'layer_update':
                    const layerIdx = data.layer_idx !== undefined ? data.layer_idx : data.layer;
                    if (layerIdx !== undefined) {
                        const deltaRef = data.delta_refusal || 0;
                        const deltaAcc = data.delta_acceptance || 0;
                        
                        state.layerActivations[layerIdx] = {
                            refusal: data.refusal_score || 0,
                            acceptance: data.acceptance_score || 0,
                            delta_refusal: deltaRef,
                            delta_acceptance: deltaAcc,
                            activation_norm: data.activation_norm || 1,
                            baseline_refusal: data.baseline_refusal || 0.5
                        };
                        updateLayerActivations();
                        
                        // Log with delta
                        const deltaStr = deltaRef !== 0 ? ` (delta: ${deltaRef > 0 ? '+' : ''}${(deltaRef * 100).toFixed(1)}%)` : '';
                        addConsoleLine('LAYER', `L${layerIdx}: refusal=${((data.refusal_score || 0) * 100).toFixed(1)}%${deltaStr}`);
                    }
                    break;

                case 'output_probs':
                case 'output_probabilities':
                    state.outputProbs = (data.tokens || []).map((t, i) => ({
                        token: t,
                        prob: (data.probabilities || data.probs || [])[i] || 0
                    }));
                    updateOutputProbs(state.outputProbs);
                    addConsoleLine('INFO', `Output probabilities updated (${data.tokens?.length || 0} tokens)`);
                    break;

                case 'qkv':
                    addConsoleLine('INFO', `QKV vectors received for layer ${data.layer || 0}`);
                    break;

                case 'mlp':
                    addConsoleLine('INFO', `MLP activations received for layer ${data.layer || 0}`);
                    break;

                case 'response':
                    // Update response display
                    const responseText = data.response || data.text || data.message || '';
                    const isSuccess = data.success || false;
                    
                    document.getElementById('current-response').textContent = responseText.slice(0, 500);
                    
                    const statusEl = document.getElementById('response-status');
                    statusEl.textContent = isSuccess ? 'SUCCESS - Attack Bypassed Safety' : 'FAILED - Model Refused';
                    statusEl.className = 'response-status ' + (isSuccess ? 'success' : 'failed');
                    
                    addConsoleLine('RESPONSE', `${isSuccess ? 'SUCCESS' : 'FAILED'}: ${responseText.slice(0, 80)}...`);
                    break;

                case 'pattern_detected':
                    // Show detected attack pattern
                    const patternAlert = document.createElement('div');
                    patternAlert.className = 'pattern-alert';
                    patternAlert.innerHTML = `
                        <span class="pattern-type">${data.pattern_type}</span>: ${data.description}
                        <br><small>Confidence: ${((data.confidence || 0) * 100).toFixed(0)}%</small>
                    `;
                    
                    const patternContainer = document.getElementById('layer-activations');
                    const existingAlert = patternContainer.querySelector('.pattern-alert');
                    if (existingAlert) existingAlert.remove();
                    patternContainer.appendChild(patternAlert);
                    
                    addConsoleLine('INFO', `Pattern detected: ${data.pattern_type} (${((data.confidence || 0) * 100).toFixed(0)}% confidence)`);
                    break;

                case 'layer_prediction':
                case 'layer_predictions':
                    // Update layer predictions for residual stream analysis
                    const predLayerIdx = data.layer !== undefined ? data.layer : data.layer_idx;
                    if (predLayerIdx !== undefined) {
                        state.layerPredictions[predLayerIdx] = {
                            token: data.token || data.top_token,
                            prob: data.prob || data.probability || 0,
                            delta: data.delta || 0,
                            before_ffn: data.before_ffn,
                            after_ffn: data.after_ffn
                        };
                        updateLayerPredictions();
                        addConsoleLine('INFO', `Layer ${predLayerIdx} prediction: ${data.token || data.top_token}`);
                    }
                    break;

                case 'ssr_buffer':
                case 'ssr_update':
                    // SSR buffer state update
                    state.isSSRMode = true;
                    if (data.buffer) {
                        state.ssrBuffer = data.buffer;
                    }
                    if (data.best_loss !== undefined) {
                        state.ssrBestLoss = data.best_loss;
                    }
                    if (data.n_replace !== undefined) {
                        state.ssrNReplace = data.n_replace;
                    }
                    updateSSRBuffer();
                    addConsoleLine('INFO', `SSR buffer updated: ${state.ssrBuffer.length} candidates, best=${state.ssrBestLoss.toFixed(4)}`);
                    break;

                case 'ssr_mode':
                    // Toggle SSR mode visibility
                    state.isSSRMode = data.enabled !== false;
                    updateSSRBuffer();
                    if (state.isSSRMode) {
                        addConsoleLine('INFO', 'SSR attack mode activated');
                    }
                    break;

                case 'phase':
                case 'phase_update':
                    // Update phase progress display
                    const phaseNum = data.current || data.phase || 0;
                    const totalPhases = data.total || 7;
                    const phaseName = data.name || 'Unknown';
                    const phaseDetail = data.detail || '';
                    const phaseProgress = data.progress !== undefined ? data.progress : (phaseNum / totalPhases * 100);
                    
                    // Update phase indicator in header or status area (if exists)
                    const phaseIndicator = document.getElementById('phase-indicator');
                    if (phaseIndicator) {
                        phaseIndicator.textContent = `PHASE ${phaseNum}/${totalPhases}: ${phaseName}`;
                    }
                    
                    // Update phase number, name, detail (if elements exist)
                    const phaseNumberEl = document.getElementById('phase-number');
                    if (phaseNumberEl) {
                        phaseNumberEl.textContent = `${phaseNum}/${totalPhases}`;
                    }
                    
                    const phaseNameEl = document.getElementById('phase-name');
                    if (phaseNameEl) {
                        phaseNameEl.textContent = phaseName;
                    }
                    
                    const phaseDetailEl = document.getElementById('phase-detail');
                    if (phaseDetailEl) {
                        phaseDetailEl.textContent = phaseDetail;
                    }
                    
                    // Update progress bar (if exists)
                    const progressBar = document.getElementById('phase-progress');
                    if (progressBar) {
                        progressBar.style.width = `${phaseProgress}%`;
                    }
                    
                    // Also try progress-bar (alternative ID)
                    const progressBarAlt = document.getElementById('progress-bar');
                    if (progressBarAlt) {
                        progressBarAlt.style.width = `${phaseProgress}%`;
                    }
                    
                    addConsoleLine('PHASE', `PHASE ${phaseNum}/${totalPhases}: ${phaseName}${phaseDetail ? ' - ' + phaseDetail : ''} (${phaseProgress.toFixed(0)}%)`);
                    break;
            }
        }

        // SSE Connection
        function connectSSE() {
            console.log('Connecting to SSE endpoint...');
            addConsoleLine('INFO', 'Connecting to MIRA server...');
            
            // Close existing connection if any
            if (window.currentEventSource) {
                window.currentEventSource.close();
            }
            
            const eventSource = new EventSource('/api/events');
            window.currentEventSource = eventSource; // Store for cleanup

            eventSource.onopen = () => {
                console.log('SSE connection opened');
                document.getElementById('connection-dot').classList.remove('disconnected');
                document.getElementById('connection-status').textContent = 'Connected';
                addConsoleLine('INFO', '[OK] SSE connection established');
            };

            eventSource.onerror = (err) => {
                console.error('SSE connection error:', err);
                const statusEl = document.getElementById('connection-status');
                const dotEl = document.getElementById('connection-dot');
                
                if (eventSource.readyState === EventSource.CONNECTING) {
                    statusEl.textContent = 'Connecting...';
                    dotEl.classList.add('disconnected');
                    addConsoleLine('INFO', 'Reconnecting to server...');
                } else if (eventSource.readyState === EventSource.CLOSED) {
                    statusEl.textContent = 'Disconnected';
                    dotEl.classList.add('disconnected');
                    addConsoleLine('ERROR', 'Connection closed. Refresh page to reconnect.');
                } else {
                    statusEl.textContent = 'Error';
                    dotEl.classList.add('disconnected');
                    addConsoleLine('ERROR', 'Connection error occurred');
                }
            };

            eventSource.onmessage = (e) => {
                try {
                    // SSE format: data: {json}
                    let dataStr = e.data;
                    if (dataStr && dataStr.startsWith('data: ')) {
                        dataStr = dataStr.substring(6);
                    }
                    
                    const event = JSON.parse(dataStr);
                    console.log('SSE event received:', event.event_type, event.data ? Object.keys(event.data) : 'no data');
                    
                    // Handle connected/ping/status events
                    if (event.event_type === 'connected') {
                        console.log('Received connected event');
                        document.getElementById('connection-dot').classList.remove('disconnected');
                        document.getElementById('connection-status').textContent = 'Connected';
                        addConsoleLine('INFO', '[OK] Connected to MIRA server');
                        if (event.data && event.data.message) {
                            addConsoleLine('INFO', event.data.message);
                        }
                        // Update phase display to show we're connected
                        const phaseNameEl = document.getElementById('phase-name');
                        const phaseDetailEl = document.getElementById('phase-detail');
                        if (phaseNameEl) {
                            phaseNameEl.textContent = 'Connected - Waiting for analysis...';
                        }
                        if (phaseDetailEl) {
                            phaseDetailEl.textContent = 'Ready to receive data';
                        }
                        return;
                    }
                    if (event.event_type === 'status') {
                        // Server status update
                        if (event.data && event.data.message) {
                            addConsoleLine('INFO', event.data.message);
                        }
                        return;
                    }
                    if (event.event_type === 'ping') {
                        // Heartbeat - keep connection alive indicator green
                        document.getElementById('connection-dot').classList.remove('disconnected');
                        document.getElementById('connection-status').textContent = 'Connected';
                        return;
                    }
                    if (event.event_type === 'phase' || event.event_type === 'phase_update') {
                        // Phase events should be handled by handleEvent, but also process here for onmessage
                        handleEvent(event);
                        return;
                    }
                    handleEvent(event);
                } catch (err) {
                    console.error('Failed to parse event:', err, 'Raw data:', e.data);
                    // Try to show connection is working even if event parsing fails
                    document.getElementById('connection-dot').classList.remove('disconnected');
                    document.getElementById('connection-status').textContent = 'Connected (parsing error)';
                }
            };

            // Listen for specific event types (matching server-side event types)
            // Note: SSE events with 'event:' prefix will trigger these listeners
            ['attack_step', 'embeddings', 'flow_graph', 'attention', 'attention_matrix', 
             'layer', 'layer_update', 'output_probs', 'output_probabilities', 
             'qkv', 'mlp', 'response', 'console', 'pattern_detected',
             'layer_prediction', 'layer_predictions', 'ssr_buffer', 'ssr_update', 'ssr_mode',
             'phase', 'phase_update'].forEach(eventType => {
                eventSource.addEventListener(eventType, (e) => {
                    try {
                        // e.data should be the JSON string from the event
                        let eventData;
                        if (typeof e.data === 'string') {
                            eventData = JSON.parse(e.data);
                        } else {
                            eventData = e.data;
                        }
                        // If eventData is the full event object, extract data field
                        if (eventData && eventData.data) {
                            handleEvent({ event_type: eventType, data: eventData.data });
                        } else {
                            handleEvent({ event_type: eventType, data: eventData });
                        }
                    } catch (err) {
                        console.error(`Failed to parse ${eventType} event:`, err, 'Raw data:', e.data);
                    }
                });
            });
        }

        // Test server connection
        async function testServerConnection() {
            try {
                const response = await fetch('/api/test');
                const data = await response.json();
                console.log('Server test response:', data);
                addConsoleLine('INFO', 'Server is running: ' + data.message);
                return true;
            } catch (err) {
                console.error('Server test failed:', err);
                addConsoleLine('ERROR', 'Cannot reach server. Is MIRA running?');
                document.getElementById('connection-status').textContent = 'Server Not Found';
                document.getElementById('connection-dot').classList.add('disconnected');
                return false;
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initFlowGraph();
            initLayerSelector(12);
            initHeadSelector();
            updateLayerActivations();
            updateOutputProbs([]);
            updateAttentionHeatmap(null, null);
            updateLayerPredictions();
            updateSSRBuffer();
            
            // Test server first, then connect SSE
            testServerConnection().then(connected => {
                if (connected) {
                    connectSSE();
                } else {
                    addConsoleLine('ERROR', 'Please start MIRA with: python main.py');
                }
            });

            // Update initial flow graph
            updateFlowGraph(generateLayerFlowGraph(0, null));

            addConsoleLine('INFO', 'Visualization initialized');
            addConsoleLine('INFO', 'New features: Multi-head attention, Layer predictions, SSR buffer');
            addConsoleLine('INFO', 'Waiting for attack data...');
        });
    </script>
</body>
</html>'''

