import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path to import NN_functions
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from NN_functions import create_model

def test_gru_model():
    """Test that the GRU model can be created and run a forward pass"""
    print("Testing GRU model creation and forward pass...")
    
    # Test parameters
    input_size = 5
    hidden_size = 32
    num_layers = 2
    output_size = 1
    batch_size = 4
    seq_len = 10
    
    try:
        # Create GRU model
        model = create_model(input_size, hidden_size, num_layers, output_size, 
                           activation_fn=nn.ReLU(), model_type='GRU')
        print("✓ GRU model created successfully")
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, input_size)
        print(f"✓ Dummy input created with shape: {dummy_input.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Forward pass successful! Output shape: {output.shape}")
            
            # Check output dimensions
            expected_shape = (batch_size, output_size)
            if output.shape == expected_shape:
                print("✓ Output dimensions are correct")
            else:
                print(f"✗ Expected output shape {expected_shape}, got {output.shape}")
                return False
        
        print("✓ All GRU tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ GRU test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_gru_model()
    if success:
        print("\n🎉 GRU implementation is working correctly!")
    else:
        print("\n❌ GRU implementation has issues that need to be fixed.")
