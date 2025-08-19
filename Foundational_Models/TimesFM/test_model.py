import torch
import sys
import os

# Add parent directories to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

from timesfm import create_timesfm_model, TimesFmHparams

def test_timesfm_model():
    """Test the TimesFM model with dummy data"""
    print("Testing TimesFM model...")
    
    # Create TimesFM hyperparameters
    hparams = TimesFmHparams(
        backend="cpu",  # Use CPU for testing
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=False,
        context_len=2048,
        hidden_size=512,
        num_heads=8,
        dropout=0.1
    )
    
    # Create a simple model
    model = create_timesfm_model(hparams=hparams)
    
    print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model architecture: {hparams.num_layers} layers, {hparams.hidden_size} hidden size, {hparams.num_heads} heads")
    
    # Test with dummy input (single feature - volatility)
    batch_size = 4
    seq_length = 100  # Within context_len
    input_size = 1    # Single feature (volatility)
    
    dummy_input = torch.randn(batch_size, seq_length, input_size)
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Forward pass successful! Output shape: {output.shape}")
        
        # Test that output has correct shape
        expected_shape = (batch_size, seq_length, 1)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print("✓ Output shape is correct")
        
        # Test that output contains finite values
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        print("✓ Output contains finite values")
        
        # Test with longer sequence (should be truncated to context_len)
        long_seq_length = 3000  # Exceeds context_len of 2048
        long_dummy_input = torch.randn(batch_size, long_seq_length, input_size)
        print(f"Testing with long sequence: {long_dummy_input.shape}")
        
        with torch.no_grad():
            long_output = model(long_dummy_input)
            print(f"Long sequence forward pass successful! Output shape: {long_output.shape}")
        
        # Should be truncated to context_len
        expected_long_shape = (batch_size, hparams.context_len, 1)
        assert long_output.shape == expected_long_shape, f"Expected {expected_long_shape}, got {long_output.shape}"
        print("✓ Long sequence is correctly truncated to context_len")
        
        print("✓ All tests passed! TimesFM model is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_timesfm_model()
    if success:
        print("\nModel is ready for experiments!")
    else:
        print("\nModel needs fixing before running experiments.")
