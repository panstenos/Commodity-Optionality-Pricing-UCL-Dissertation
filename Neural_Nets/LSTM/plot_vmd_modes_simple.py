import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_vmd_modes_simple(model, X_sample, save_path=None, show_plot=False, mode_names=None):
    """
    Simple plotting of VMD decomposed modes
    
    Args:
        model: VMD_LSTM model with vmd_decompose method
        X_sample: Input tensor of shape (batch_size, seq_len, features)
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot (default: False)
        mode_names: Custom names for the modes (optional)
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
            
            # Convert to numpy for plotting
            if isinstance(imf_modes, torch.Tensor):
                imf_modes_np = imf_modes.cpu().numpy()
            else:
                imf_modes_np = imf_modes
            
            # Get original signal
            original_signal = X_sample.cpu().numpy()[0, :, 0]
            
            # Default mode names
            if mode_names is None:
                mode_names = [f'Mode {i+1}' for i in range(model.num_modes)]
            
            # Create simple plot
            fig, axes = plt.subplots(model.num_modes + 1, 1, figsize=(12, 2.5 * (model.num_modes + 1)))
            fig.suptitle('VMD Decomposition', fontsize=14)
            
            # Plot original signal
            axes[0].plot(original_signal, 'b-', linewidth=1.5, label='Original Signal')
            axes[0].set_title('Original Signal')
            axes[0].set_ylabel('Amplitude')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot each decomposed mode
            colors = ['r-', 'g-', 'm-', 'c-', 'y-']
            
            for i in range(model.num_modes):
                mode_signal = imf_modes_np[i, 0, :, 0]
                color_idx = i % len(colors)
                
                axes[i+1].plot(mode_signal, colors[color_idx], linewidth=1.5, label=mode_names[i])
                axes[i+1].set_title(f'{mode_names[i]} (IMF {i+1})')
                axes[i+1].set_ylabel('Amplitude')
                axes[i+1].legend()
                axes[i+1].grid(True, alpha=0.3)
            
            # Add x-label to the last subplot
            axes[-1].set_xlabel('Time Steps')
            
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"VMD modes plot saved to: {save_path}")
            
            # Show plot if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            
            return imf_modes_np
            
        else:
            print("Model does not have vmd_decompose method")
            return None

def plot_vmd_modes_comparison(model, X_samples, sample_names, save_path=None, show_plot=False):
    """
    Plot VMD modes for multiple samples for comparison
    
    Args:
        model: VMD_LSTM model with vmd_decompose method
        X_samples: List of input tensors
        sample_names: List of names for each sample
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot (default: False)
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        # Create comparison plot
        fig, axes = plt.subplots(len(X_samples), model.num_modes + 1, 
                                figsize=(15, 3 * len(X_samples)))
        
        if len(X_samples) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('VMD Decomposition Comparison', fontsize=16)
        
        for sample_idx, (X_sample, sample_name) in enumerate(zip(X_samples, sample_names)):
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
                
                # Convert to numpy for plotting
                if isinstance(imf_modes, torch.Tensor):
                    imf_modes_np = imf_modes.cpu().numpy()
                else:
                    imf_modes_np = imf_modes
                
                # Get original signal
                original_signal = X_sample.cpu().numpy()[0, :, 0]
                
                # Plot original signal
                axes[sample_idx, 0].plot(original_signal, 'b-', linewidth=1.5, label=f'{sample_name} - Original')
                axes[sample_idx, 0].set_title(f'{sample_name} - Original Signal')
                axes[sample_idx, 0].set_ylabel('Amplitude')
                axes[sample_idx, 0].legend()
                axes[sample_idx, 0].grid(True, alpha=0.3)
                
                # Plot each decomposed mode
                colors = ['r-', 'g-', 'm-', 'c-', 'y-']
                
                for i in range(model.num_modes):
                    mode_signal = imf_modes_np[i, 0, :, 0]
                    color_idx = i % len(colors)
                    
                    axes[sample_idx, i+1].plot(mode_signal, colors[color_idx], linewidth=1.5, 
                                             label=f'{sample_name} - Mode {i+1}')
                    axes[sample_idx, i+1].set_title(f'{sample_name} - Mode {i+1}')
                    axes[sample_idx, i+1].set_ylabel('Amplitude')
                    axes[sample_idx, i+1].legend()
                    axes[sample_idx, i+1].grid(True, alpha=0.3)
                
                # Add x-label to the last subplot of each row
                axes[sample_idx, -1].set_xlabel('Time Steps')
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"VMD comparison plot saved to: {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

# Example usage functions
def example_single_plot():
    """Example of how to use the simple VMD modes plotting"""
    print("Example: Single VMD modes plot")
    print("Usage: plot_vmd_modes_simple(model, X_sample, save_path='path/to/plot.png')")

def example_comparison_plot():
    """Example of how to use the comparison VMD modes plotting"""
    print("Example: Comparison VMD modes plot")
    print("Usage: plot_vmd_modes_comparison(model, [X1, X2], ['Sample1', 'Sample2'], save_path='path/to/comparison.png')")

if __name__ == "__main__":
    print("VMD Modes Plotting Functions")
    print("============================")
    print("\nAvailable functions:")
    print("1. plot_vmd_modes_simple() - Basic VMD modes visualization")
    print("2. plot_vmd_modes_comparison() - Compare VMD modes across samples")
    print("\nExamples:")
    example_single_plot()
    example_comparison_plot()
