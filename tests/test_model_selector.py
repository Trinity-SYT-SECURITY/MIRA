"""
Test Model Selector - Quick demo of interactive model selection

Run this to see the model selector in action without running full main.py
"""

from mira.utils.model_selector import select_model_interactive

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MIRA Model Selector Demo")
    print("="*60)
    
    selected_model = select_model_interactive()
    
    print("\n" + "="*60)
    print(f"  ✓ You selected: {selected_model}")
    print("="*60)
    print("\n  To use this model, add to .env:")
    print(f"  MODEL_NAME={selected_model}")
    print()
