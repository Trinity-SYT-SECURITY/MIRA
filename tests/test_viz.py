#!/usr/bin/env python3
"""
Quick test script to verify live visualization is working.

This script:
1. Starts the visualization server
2. Opens browser automatically
3. Sends demo events to show the visualization
4. Keeps running so you can see the updates

Usage:
    python test_viz.py
"""

from mira.visualization.live_server import LiveVisualizationServer
import time

def main():
    print("Starting MIRA Live Visualization Test...")
    print("=" * 60)
    
    # Start server
    server = LiveVisualizationServer(port=5000)
    server.start(open_browser=True)
    
    print("\n✓ Server running at http://127.0.0.1:5000")
    print("✓ Browser should open automatically")
    print("\nSending demo events to test visualization...\n")
    
    time.sleep(2)
    
    # Demo 1: Send token embeddings
    print("1. Sending token embeddings...")
    server.send_embeddings(
        tokens=['How', 'to', 'make', 'a', 'bomb', '?'],
        embeddings=[]
    )
    time.sleep(1)
    
    # Demo 2: Send layer-by-layer updates
    print("2. Sending layer updates (simulating forward pass)...")
    for layer in range(6):
        server.send_layer_update(
            layer_idx=layer,
            refusal_score=0.3 + layer * 0.1,
            acceptance_score=0.1 + layer * 0.05,
            direction='forward'
        )
        print(f"   Layer {layer} processed")
        time.sleep(0.4)
    
    # Demo 3: Send attack steps
    print("\n3. Sending attack optimization steps...")
    for step in range(15):
        server.send_attack_step(
            step=step,
            loss=5.0 - step * 0.3,
            suffix=f'adversarial_token_{step}',
            success=step > 12
        )
        print(f"   Step {step}: loss={5.0 - step * 0.3:.3f}")
        time.sleep(0.5)
    
    # Demo 4: Send attention matrix
    print("\n4. Sending attention matrix...")
    server.send_attention_matrix(
        layer_idx=0,
        head_idx=0,
        attention_weights=[
            [0.8, 0.1, 0.05, 0.05],
            [0.2, 0.6, 0.1, 0.1],
            [0.1, 0.2, 0.5, 0.2],
            [0.1, 0.1, 0.2, 0.6]
        ],
        tokens=['How', 'to', 'make', 'bomb']
    )
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("Check your browser at http://127.0.0.1:5000")
    print("You should see:")
    print("  - Input tokens displayed")
    print("  - Layer-by-layer processing")
    print("  - Attack progress with loss decreasing")
    print("  - Attention matrix heatmap")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping server...")

if __name__ == "__main__":
    main()
