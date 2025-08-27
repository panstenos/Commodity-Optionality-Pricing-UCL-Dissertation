import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

# Add the parent directories to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to root
neural_nets_dir = os.path.dirname(current_dir)  # Go up one level to Neural_Nets

# Add both directories to Python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if neural_nets_dir not in sys.path:
    sys.path.insert(0, neural_nets_dir)

from functions import load_data
from NN_functions import create_model

def plot_vmd_modes_analysis(model, X_sample, save_path=None, show_plot=False):
    """
    Comprehensive analysis and plotting of VMD decomposed modes
    
    Args:
        model: VMD_LSTM model with vmd_decompose method
        X_sample: Input tensor of shape (batch_size, seq_len, features)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot (default: False)
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        # Ensure input is on the correct device
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.tensor(X_sample).float()
        
        # Ensure we have a batch dimension
        if X_sample.ndim == 2:
            X_sample = X_sample.unsqueeze(0)
        
        X_sample = X_sample.to(device)
        
        # Get the decomposed modes
        if hasattr(model, 'vmd_decompose'):
            imf_modes = model.vmd_decompose(X_sample)
            
            # Convert to numpy for analysis
            if isinstance(imf_modes, torch.Tensor):
                imf_modes_np = imf_modes.cpu().numpy()
            else:
                imf_modes_np = imf_modes
            
            # Get original signal
            original_signal = X_sample.cpu().numpy()[0, :, 0]
            
            # Create comprehensive plot
            fig, axes = plt.subplots(model.num_modes + 2, 2, figsize=(20, 4 * (model.num_modes + 2)))
            fig.suptitle('VMD Decomposition Analysis', fontsize=16)
            
            # Plot 1: Original signal
            axes[0, 0].plot(original_signal, 'b-', linewidth=2, label='Original Signal')
            axes[0, 0].set_title('Original Signal')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Original signal FFT
            fft_original = np.fft.fft(original_signal)
            freq_original = np.fft.fftfreq(len(original_signal))
            axes[0, 1].plot(freq_original[:len(freq_original)//2], np.abs(fft_original[:len(fft_original)//2]), 'b-', linewidth=2)
            axes[0, 1].set_title('Original Signal FFT')
            axes[0, 1].set_xlabel('Frequency')
            axes[0, 1].set_ylabel('Magnitude')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot each decomposed mode
            mode_names = ['Trend Mode', 'Seasonal Mode', 'Residual Mode']
            colors = ['r-', 'g-', 'm-']
            
            for i in range(model.num_modes):
                mode_signal = imf_modes_np[i, 0, :, 0]
                
                # Time domain plot
                axes[i+1, 0].plot(mode_signal, colors[i], linewidth=2, label=f'{mode_names[i]} (IMF {i+1})')
                axes[i+1, 0].set_title(f'{mode_names[i]} (IMF {i+1}) - Time Domain')
                axes[i+1, 0].set_ylabel('Amplitude')
                axes[i+1, 0].legend()
                axes[i+1, 0].grid(True, alpha=0.3)
                
                # Frequency domain plot
                fft_mode = np.fft.fft(mode_signal)
                freq_mode = np.fft.fftfreq(len(mode_signal))
                axes[i+1, 1].plot(freq_mode[:len(freq_mode)//2], np.abs(fft_mode[:len(fft_mode)//2]), colors[i], linewidth=2)
                axes[i+1, 1].set_title(f'{mode_names[i]} (IMF {i+1}) - Frequency Domain')
                axes[i+1, 1].set_xlabel('Frequency')
                axes[i+1, 1].set_ylabel('Magnitude')
                axes[i+1, 1].grid(True, alpha=0.3)
            
            # Plot reconstruction
            reconstructed = np.sum(imf_modes_np[:, 0, :, 0], axis=0)
            axes[-1, 0].plot(reconstructed, 'purple', linewidth=2, label='Reconstructed Signal')
            axes[-1, 0].plot(original_signal, 'b--', linewidth=1, alpha=0.7, label='Original Signal')
            axes[-1, 0].set_title('Signal Reconstruction')
            axes[-1, 0].set_xlabel('Time Steps')
            axes[-1, 0].set_ylabel('Amplitude')
            axes[-1, 0].legend()
            axes[-1, 0].grid(True, alpha=0.3)
            
            # Plot reconstruction error
            reconstruction_error = original_signal - reconstructed
            axes[-1, 1].plot(reconstruction_error, 'orange', linewidth=2, label='Reconstruction Error')
            axes[-1, 1].set_title('Reconstruction Error')
            axes[-1, 1].set_xlabel('Time Steps')
            axes[-1, 1].set_ylabel('Error')
            axes[-1, 1].legend()
            axes[-1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"VMD analysis plot saved to: {save_path}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            
            # Print analysis summary
            print("\n=== VMD Decomposition Analysis ===")
            print(f"Number of modes: {model.num_modes}")
            print(f"Signal length: {len(original_signal)}")
            print(f"Original signal variance: {np.var(original_signal):.6f}")
            
            for i in range(model.num_modes):
                mode_signal = imf_modes_np[i, 0, :, 0]
                print(f"\nMode {i+1} ({mode_names[i]}):")
                print(f"  Variance: {np.var(mode_signal):.6f}")
                print(f"  Mean: {np.mean(mode_signal):.6f}")
                print(f"  Max: {np.max(mode_signal):.6f}")
                print(f"  Min: {np.min(mode_signal):.6f}")
            
            print(f"\nReconstruction error:")
            print(f"  MSE: {np.mean(reconstruction_error**2):.6f}")
            print(f"  MAE: {np.mean(np.abs(reconstruction_error)):.6f}")
            print(f"  Max error: {np.max(np.abs(reconstruction_error)):.6f}")
            
            return imf_modes_np
            
        else:
            print("Model does not have vmd_decompose method")
            return None

def main():
    """
    Main function to demonstrate VMD modes plotting
    """
    # Load sample data
    data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
    df = load_data(data_path)
    
    # Create a VMD_LSTM model
    input_size = 1
    hidden_size = 32
    number_layers = 2
    output_size = 1
    
    model = create_model(input_size, hidden_size, number_layers, output_size, 
                        activation_fn=nn.ReLU(), model_type='VMD_LSTM')
    
    # Create sample input data (using log returns as example)
    sample_data = df['al_lme_prices_abs_log_returns'].values[:100]  # First 100 points
    sample_data = sample_data.reshape(1, -1, 1)  # (batch_size, seq_len, features)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'VMD_experiment_results', 'vmd_modes_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot VMD modes
    save_path = os.path.join(output_dir, 'vmd_modes_analysis.png')
    plot_vmd_modes_analysis(model, sample_data, save_path=save_path, show_plot=False)
    
    print(f"\nVMD modes analysis completed. Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
