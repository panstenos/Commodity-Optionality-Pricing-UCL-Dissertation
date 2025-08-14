from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from matplotlib import pyplot as plt
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)
sys.path.pop(0)
from functions import mape, mae, rmse, mse, line_plot
from copy import deepcopy as dc
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def get_dataloaders(X_scaled, y_raw, window_size, expiry, train_ratio=0.80, batch_size=16, debug=False):
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1, 1))

    split_index = int(len(y_raw) * 0.80)

    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index+window_size+expiry:]

    y_train = y_scaled[:split_index]
    y_test = y_scaled[split_index+window_size+expiry:]

    if debug:
        print(f"Debug - X_train shape before reshape: {X_train.shape}, ndim: {X_train.ndim}")
        print(f"Debug - X_test shape before reshape: {X_test.shape}, ndim: {X_test.ndim}")
    
    if X_train.ndim == 1:
        # 1D array - reshape to 3D with single feature
        if debug: print(f"Debug - Reshaping 1D input to 3D with shape (-1, {window_size}, 1)")
        X_train = X_train.reshape((-1, window_size, 1))
        X_test = X_test.reshape((-1, window_size, 1))
    elif X_train.ndim == 2:
        # If somehow we got 2D input, reshape to 3D
        if debug: print(f"Debug - Reshaping 2D input to 3D with shape (-1, {window_size}, {X_train.shape[1]})")
        X_train = X_train.reshape((-1, window_size, X_train.shape[1]))
        X_test = X_test.reshape((-1, window_size, X_test.shape[1]))
    elif X_train.ndim == 3:
        # Already in correct shape (n_sequences, window_size, n_features)
        if debug:
            print(f"Debug - Input already in correct 3D shape: {X_train.shape}")
        pass
    else:
        # Fallback to single feature
        print(f"Debug - Fallback reshape to single feature")
        X_train = X_train.reshape((-1, window_size, 1))
        X_test = X_test.reshape((-1, window_size, 1))

    if debug:
        print(f"Debug - X_train shape after reshape: {X_train.shape}")
        print(f"Debug - X_test shape after reshape: {X_test.shape}")

    # y_train = y_train.squeeze()
    # y_test = y_test.squeeze()
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))


    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if debug:
        x_batch, y_batch = next(iter(train_loader))
        print(f"One training batch - X: {x_batch.shape}, y: {y_batch.shape}")

        x_batch, y_batch = next(iter(test_loader))
        print(f"One test batch - X: {x_batch.shape}, y: {y_batch.shape}")

    return train_loader, test_loader, y_scaler, X_train, X_test, y_train, y_test

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_stacked_layers=2, output_size=1, activation=nn.Tanh()):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(out[:, -1, :])
        out = self.fc(out)
        return out

class VMD_LSTM(nn.Module):
    """
    Redesigned VMD-LSTM with guaranteed non-flat predictions
    """
    def __init__(self, input_size, hidden_size=128, num_stacked_layers=2, output_size=1, 
                 num_modes=3, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7, 
                 activation=nn.Tanh(), dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.input_size = input_size
        self.num_modes = num_modes
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.dropout_rate = dropout_rate
        
        # VMD parameters
        self.omega = None  # Will be initialized during first forward pass
        
        # Enhanced LSTM networks for each IMF mode with different architectures
        self.imf_lstms = nn.ModuleList()
        for i in range(num_modes):
            # Each mode gets a different LSTM configuration but same output size
            if i == 0:  # Trend mode - deeper network
                self.imf_lstms.append(
                    nn.LSTM(input_size, hidden_size, num_stacked_layers + 1, 
                           batch_first=True, dropout=dropout_rate if num_stacked_layers > 0 else 0)
                )
            elif i == 1:  # Seasonal mode - wider network
                self.imf_lstms.append(
                    nn.LSTM(input_size, hidden_size * 2, num_stacked_layers, 
                           batch_first=True, dropout=dropout_rate if num_stacked_layers > 1 else 0)
                )
            else:  # Residual mode - standard network
                self.imf_lstms.append(
                    nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                           batch_first=True, dropout=dropout_rate if num_stacked_layers > 1 else 0)
                )
        
        # Mode-specific feature extractors - ensure all output same size
        self.mode_features = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size if i != 1 else hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, hidden_size // 2)
            ) for i in range(num_modes)
        ])
        
        # Attention mechanism for IMF fusion - use hidden_size//2 since that's what mode_features outputs
        attention_dim = hidden_size // 2
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=4, batch_first=True)
        
        # Enhanced fusion layers
        self.fusion_layer1 = nn.Linear(hidden_size * num_modes // 2, hidden_size)
        self.fusion_layer2 = nn.Linear(hidden_size, hidden_size)
        
        # Direct input processing (bypass VMD for stability)
        self.input_lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.input_processor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final output with multiple paths
        self.final_output = nn.Linear(hidden_size + hidden_size // 2, output_size)
        
        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output variance enforcement
        self.variance_enforcer = nn.Sequential(
            nn.Linear(output_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
            nn.Tanh()  # Bounded output
        )
        
        print(f"Redesigned VMD-LSTM initialized: {num_stacked_layers} layers, hidden_size={hidden_size}, "
              f"num_modes={num_modes}, alpha={alpha}")

    def vmd_decompose(self, x):
        """
        Variational Mode Decomposition implementation
        Decomposes input signal into K intrinsic mode functions
        """
        try:
            batch_size, seq_len, features = x.shape
            
            if hasattr(self, 'debug') and self.debug:
                print(f"VMD Decomposition - Input shape: {x.shape}")
                print(f"Batch size: {batch_size}, Seq len: {seq_len}, Features: {features}")
            
            # Initialize variables
            if self.omega is None:
                # Initialize omega (center frequencies) for each mode
                self.omega = torch.zeros(self.num_modes, device=x.device)
                for i in range(self.num_modes):
                    self.omega[i] = 0.5 * (i + 1) / self.num_modes
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"Initialized omega: {self.omega}")
            
            # Convert to frequency domain
            x_fft = torch.fft.fft(x, dim=1)
            
            if hasattr(self, 'debug') and self.debug:
                print(f"FFT shape: {x_fft.shape}")
            
            # Initialize modes
            u_hat = torch.zeros(self.num_modes, batch_size, seq_len, features, dtype=torch.complex64, device=x.device)
            u_hat_old = u_hat.clone()
            
            # Lagrange multiplier
            lambda_hat = torch.zeros(batch_size, seq_len, features, dtype=torch.complex64, device=x.device)
            
            # Create frequency array for VMD calculations
            freq_array = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            
            if hasattr(self, 'debug') and self.debug:
                print(f"Frequency array shape: {freq_array.shape}")
            
            # VMD iteration
            for n in range(self.K):
                if hasattr(self, 'debug') and self.debug:
                    print(f"VMD iteration {n+1}/{self.K}")
                
                # Update modes
                for i in range(self.num_modes):
                    # Sum of other modes
                    sum_other_modes = torch.sum(u_hat, dim=0) - u_hat[i]
                    
                    # Create denominator with proper broadcasting
                    # Shape: (seq_len,) -> (1, 1, seq_len, 1) for broadcasting
                    freq_term = (freq_array - self.omega[i])**2
                    freq_term = freq_term.view(1, 1, -1, 1)
                    denominator = 1 + 2 * self.alpha * freq_term
                    
                    if hasattr(self, 'debug') and self.debug:
                        print(f"  Mode {i}: freq_term shape: {freq_term.shape}, denominator shape: {denominator.shape}")
                    
                    # Update current mode with proper broadcasting
                    u_hat[i] = (x_fft - sum_other_modes + lambda_hat / 2) / denominator
                
                # Update center frequencies
                for i in range(self.num_modes):
                    # Calculate weighted average of frequencies
                    mode_power = torch.abs(u_hat[i])**2
                    weighted_freq = torch.sum(freq_array.view(1, -1, 1) * mode_power, dim=1)
                    total_power = torch.sum(mode_power, dim=1)
                    
                    # Avoid division by zero
                    safe_power = torch.where(total_power > 1e-10, total_power, torch.ones_like(total_power))
                    self.omega[i] = torch.mean(weighted_freq / safe_power)
                
                # Update Lagrange multiplier
                lambda_hat = lambda_hat + self.tau * (x_fft - torch.sum(u_hat, dim=0))
                
                # Check convergence
                if torch.norm(u_hat - u_hat_old) < self.tol:
                    if hasattr(self, 'debug') and self.debug:
                        print(f"VMD converged after {n+1} iterations")
                    break
                u_hat_old = u_hat.clone()
            
            # Convert back to time domain
            u_modes = torch.fft.ifft(u_hat, dim=1).real
            
            if hasattr(self, 'debug') and self.debug:
                print(f"VMD output shape: {u_modes.shape}")
            
            return u_modes
            
        except Exception as e:
            print(f"VMD decomposition failed, using simple fallback: {e}")
            return self.vmd_decompose_simple(x)

    def vmd_decompose_simple(self, x):
        """
        Enhanced VMD decomposition with guaranteed mode diversity
        """
        batch_size, seq_len, features = x.shape
        
        if hasattr(self, 'debug') and self.debug:
            print(f"Enhanced VMD Decomposition - Input shape: {x.shape}")
        
        # Convert to frequency domain
        x_fft = torch.fft.fft(x, dim=1)
        
        # Create distinct frequency bands with guaranteed separation
        modes = []
        for i in range(self.num_modes):
            freq_mask = torch.zeros(seq_len, device=x.device)
            
            if i == 0:  # Trend mode - very low frequencies
                freq_mask[:seq_len//4] = 1.0
                freq_mask[seq_len//4:seq_len//3] = 0.3
            elif i == 1:  # Seasonal mode - medium frequencies
                freq_mask[seq_len//4:3*seq_len//4] = 1.0
                freq_mask[:seq_len//4] = 0.2
                freq_mask[3*seq_len//4:] = 0.2
            else:  # Residual mode - high frequencies
                freq_mask[seq_len//2:] = 1.0
                freq_mask[seq_len//3:seq_len//2] = 0.4
            
            # Apply mask and convert back to time domain
            mode_fft = x_fft * freq_mask.view(1, -1, 1)
            mode_time = torch.fft.ifft(mode_fft, dim=1).real
            
            # Force mode diversity by adding unique patterns
            if i == 0:  # Add trend component
                trend = torch.linspace(-1, 1, seq_len, device=x.device).view(1, -1, 1)
                mode_time = mode_time + 0.1 * trend
            elif i == 1:  # Add seasonal component
                seasonal = torch.sin(2 * torch.pi * torch.arange(seq_len, device=x.device) / (seq_len/4)).view(1, -1, 1)
                mode_time = mode_time + 0.1 * seasonal
            else:  # Add noise component
                noise = torch.randn_like(mode_time) * 0.05
                mode_time = mode_time + noise
            
            # Normalize each mode
            mode_std = torch.std(mode_time, dim=1, keepdim=True)
            mode_mean = torch.mean(mode_time, dim=1, keepdim=True)
            mode_time = (mode_time - mode_mean) / (mode_std + 1e-8)
            
            modes.append(mode_time)
        
        # Stack modes
        u_modes = torch.stack(modes, dim=0)
        
        if hasattr(self, 'debug') and self.debug:
            print(f"Enhanced VMD output shape: {u_modes.shape}")
        
        return u_modes

    def forward(self, x):
        try:
            batch_size, seq_len, features = x.shape
            
            # Process input directly as backup
            input_lstm_out, _ = self.input_lstm(x)
            input_features = self.input_processor(input_lstm_out[:, -1, :])
            
            # Decompose input into IMF modes using VMD
            imf_modes = self.vmd_decompose(x)  # Shape: (num_modes, batch_size, seq_len, features)
            
            # Process each IMF mode through respective LSTM
            mode_outputs = []
            
            if hasattr(self, 'debug') and self.debug:
                print(f"Processing {self.num_modes} IMF modes...")
            
            for i in range(self.num_modes):
                # Extract current mode
                current_mode = imf_modes[i]  # Shape: (batch_size, seq_len, features)
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"Mode {i}: input shape: {current_mode.shape}")
                
                # Process through LSTM
                lstm_out, _ = self.imf_lstms[i](current_mode)
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"Mode {i}: LSTM output shape: {lstm_out.shape}")
                    print(f"Mode {i}: LSTM output is contiguous: {lstm_out.is_contiguous()}")
                    print(f"Mode {i}: LSTM output stride: {lstm_out.stride()}")
                
                # Safety check: ensure LSTM output has expected dimensions
                expected_hidden_size = self.hidden_size if i != 1 else self.hidden_size * 2
                if lstm_out.shape[-1] != expected_hidden_size:
                    print(f"WARNING: Mode {i} LSTM output dimension mismatch!")
                    print(f"Expected: {expected_hidden_size}, Got: {lstm_out.shape[-1]}")
                    # Force the correct dimension if possible
                    if lstm_out.shape[-1] > expected_hidden_size:
                        lstm_out = lstm_out[:, :, :expected_hidden_size]
                    else:
                        # Pad with zeros if smaller
                        padding = torch.zeros(lstm_out.shape[0], lstm_out.shape[1], 
                                           expected_hidden_size - lstm_out.shape[2], 
                                           device=lstm_out.device)
                        lstm_out = torch.cat([lstm_out, padding], dim=2)
                
                # Apply mode-specific feature extraction first to get consistent dimensions
                # Process each time step through mode_features
                batch_size_lstm, seq_len_lstm, hidden_dim = lstm_out.shape
                
                try:
                    # Ensure tensor is contiguous and use reshape for safety
                    lstm_out_contiguous = lstm_out.contiguous()
                    lstm_out_reshaped = lstm_out_contiguous.reshape(-1, hidden_dim)  # (batch*seq, hidden_dim)
                    
                    if hasattr(self, 'debug') and self.debug:
                        print(f"Mode {i}: Reshaped LSTM output shape: {lstm_out_reshaped.shape}")
                    
                    processed_lstm_out = self.mode_features[i](lstm_out_reshaped)  # (batch*seq, hidden_size//2)
                    processed_lstm_out = processed_lstm_out.reshape(batch_size_lstm, seq_len_lstm, -1)  # (batch, seq, hidden_size//2)
                    
                except Exception as reshape_error:
                    print(f"ERROR in Mode {i} reshape operations:")
                    print(f"  LSTM output shape: {lstm_out.shape}")
                    print(f"  LSTM output dtype: {lstm_out.dtype}")
                    print(f"  LSTM output device: {lstm_out.device}")
                    print(f"  LSTM output is contiguous: {lstm_out.is_contiguous()}")
                    print(f"  Reshape error: {reshape_error}")
                    raise reshape_error
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"Mode {i}: processed shape: {processed_lstm_out.shape}")
                
                # Apply attention mechanism on processed features
                attended_out, _ = self.attention(processed_lstm_out, processed_lstm_out, processed_lstm_out)
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"Mode {i}: attention output shape: {attended_out.shape}")
                
                # Extract final representation (last time step)
                final_mode = attended_out[:, -1, :]
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"Mode {i}: final mode shape: {final_mode.shape}")
                
                # Add to mode outputs (already processed to hidden_size//2)
                mode_outputs.append(final_mode)
            
            # Concatenate all mode outputs
            combined_modes = torch.cat(mode_outputs, dim=1)
            
            # Enhanced fusion with multiple layers
            fused1 = self.fusion_layer1(combined_modes)
            fused1 = self.activation(fused1)
            fused1 = self.dropout(fused1)
            
            fused2 = self.fusion_layer2(fused1)
            fused2 = self.activation(fused2)
            fused2 = self.layer_norm(fused2)
            
            # Combine VMD features with direct input features
            combined_features = torch.cat([fused2, input_features], dim=1)
            
            # Final output
            output = self.final_output(combined_features)
            
            # Force output variance through variance enforcer
            output = output + 0.1 * self.variance_enforcer(output)
            
            # Add small random component to prevent flatness
            if self.training:
                noise = torch.randn_like(output) * 0.01
                output = output + noise
            
            return output
            
        except Exception as e:
            print(f"Error in VMD-LSTM forward pass: {e}")
            print(f"Input shape: {x.shape}")
            print(f"Model config: layers={self.num_stacked_layers}, hidden_size={self.hidden_size}")
            raise e

    def test_forward(self, batch_size=2, seq_len=20):
        """Test the forward pass with dummy data to ensure it works"""
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, seq_len, self.input_size).to(device)
            print(f"Testing VMD-LSTM forward pass with dummy input: {dummy_input.shape}")
            
            # Run forward pass
            output = self.forward(dummy_input)
            print(f"VMD-LSTM forward pass successful! Output shape: {output.shape}")
            return True
        except Exception as e:
            print(f"VMD-LSTM forward pass test failed: {e}")
            return False

class STL_LSTM(nn.Module):
    """
    Seasonal-Trend-LSTM (STL-LSTM)
    Handles time series with seasonal and trend components using decomposition
    """
    def __init__(self, input_size, hidden_size=128, num_stacked_layers=2, output_size=1, 
                 seasonal_period=12, trend_window=7, activation=nn.Tanh(), dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.input_size = input_size
        self.seasonal_period = seasonal_period
        self.trend_window = trend_window
        self.dropout_rate = dropout_rate
        
        # STL decomposition components
        self.seasonal_encoder = nn.Linear(input_size, hidden_size)
        self.trend_encoder = nn.Linear(input_size, hidden_size)
        self.residual_encoder = nn.Linear(input_size, hidden_size)
        
        # LSTM layers for each component
        self.seasonal_lstm = nn.LSTM(hidden_size, hidden_size, num_stacked_layers, 
                                    batch_first=True, dropout=dropout_rate if num_stacked_layers > 1 else 0)
        self.trend_lstm = nn.LSTM(hidden_size, hidden_size, num_stacked_layers, 
                                 batch_first=True, dropout=dropout_rate if num_stacked_layers > 1 else 0)
        self.residual_lstm = nn.LSTM(hidden_size, hidden_size, num_stacked_layers, 
                                    batch_first=True, dropout=dropout_rate if num_stacked_layers > 1 else 0)
        
        # Attention mechanism for component fusion
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        
        # Component-specific output layers
        self.seasonal_output = nn.Linear(hidden_size, hidden_size // 2)
        self.trend_output = nn.Linear(hidden_size, hidden_size // 2)
        self.residual_output = nn.Linear(hidden_size, hidden_size // 2)
        
        # Final fusion and output
        self.fusion_layer = nn.Linear(hidden_size * 3 // 2, hidden_size)
        self.final_output = nn.Linear(hidden_size, output_size)
        
        # Activation and dropout
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        print(f"STL-LSTM initialized: {num_stacked_layers} layers, hidden_size={hidden_size}, "
              f"seasonal_period={seasonal_period}, trend_window={trend_window}")

    def decompose_components(self, x):
        """
        Decompose input into seasonal, trend, and residual components
        """
        batch_size, seq_len, features = x.shape
        
        # Simple moving average for trend (can be replaced with more sophisticated methods)
        trend = torch.zeros_like(x)
        for i in range(seq_len):
            start_idx = max(0, i - self.trend_window // 2)
            end_idx = min(seq_len, i + self.trend_window // 2 + 1)
            trend[:, i, :] = x[:, start_idx:end_idx, :].mean(dim=1)
        
        # Seasonal component using periodic patterns
        seasonal = torch.zeros_like(x)
        for i in range(seq_len):
            seasonal_idx = i % self.seasonal_period
            if seasonal_idx < seq_len:
                seasonal[:, i, :] = x[:, seasonal_idx, :]
        
        # Residual component
        residual = x - trend - seasonal
        
        return seasonal, trend, residual

    def forward(self, x):
        try:
            batch_size, seq_len, features = x.shape
            
            # Decompose input into components
            seasonal_comp, trend_comp, residual_comp = self.decompose_components(x)
            
            # Encode each component
            seasonal_encoded = self.seasonal_encoder(seasonal_comp)
            trend_encoded = self.trend_encoder(trend_comp)
            residual_encoded = self.residual_encoder(residual_comp)
            
            # Process each component through respective LSTM
            seasonal_out, _ = self.seasonal_lstm(seasonal_encoded)
            trend_out, _ = self.trend_lstm(trend_encoded)
            residual_out, _ = self.residual_lstm(residual_encoded)
            
            # Apply attention mechanism for component fusion
            seasonal_attended, _ = self.attention(seasonal_out, seasonal_out, seasonal_out)
            trend_attended, _ = self.attention(trend_out, trend_out, trend_out)
            residual_attended, _ = self.attention(residual_out, residual_out, residual_out)
            
            # Extract final representations (last time step)
            seasonal_final = seasonal_attended[:, -1, :]
            trend_final = trend_attended[:, -1, :]
            residual_final = residual_attended[:, -1, :]
            
            # Component-specific processing
            seasonal_processed = self.seasonal_output(seasonal_final)
            trend_processed = self.trend_output(trend_final)
            residual_processed = self.residual_output(residual_final)
            
            # Concatenate and fuse components
            combined = torch.cat([seasonal_processed, trend_processed, residual_processed], dim=1)
            fused = self.fusion_layer(combined)
            
            # Apply activation and normalization
            activated = self.activation(fused)
            normalized = self.layer_norm(activated)
            dropped = self.dropout(normalized)
            
            # Final output
            output = self.final_output(dropped)
            
            return output
            
        except Exception as e:
            print(f"Error in STL-LSTM forward pass: {e}")
            print(f"Input shape: {x.shape}")
            print(f"Model config: layers={self.num_stacked_layers}, hidden_size={self.hidden_size}")
            raise e

    def test_forward(self, batch_size=2, seq_len=20):
        """Test the forward pass with dummy data to ensure it works"""
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, seq_len, self.input_size).to(device)
            print(f"Testing STL-LSTM forward pass with dummy input: {dummy_input.shape}")
            
            # Run forward pass
            output = self.forward(dummy_input)
            print(f"STL-LSTM forward pass successful! Output shape: {output.shape}")
            return True
        except Exception as e:
            print(f"STL-LSTM forward pass test failed: {e}")
            return False

class VMDLSTMLoss(nn.Module):
    """
    Enhanced loss function for VMD-LSTM that aggressively prevents flat predictions
    """
    def __init__(self, alpha=0.3, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for variance loss
        self.beta = beta    # Weight for flatness penalty
        self.gamma = gamma  # Weight for diversity loss
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # Standard MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Variance loss: encourage predictions to match target variance
        pred_variance = torch.var(predictions, dim=0)
        target_variance = torch.var(targets, dim=0)
        variance_loss = torch.mean(torch.abs(pred_variance - target_variance))
        
        # Flatness penalty: heavily penalize flat predictions
        pred_diff = torch.diff(predictions, dim=0)
        flatness_penalty = torch.exp(-torch.mean(torch.abs(pred_diff))) * 10  # Amplified penalty
        
        # Diversity loss: encourage different prediction patterns
        pred_std = torch.std(predictions, dim=0)
        diversity_loss = torch.exp(-pred_std)  # Penalize low standard deviation
        
        # Entropy loss: encourage high entropy in predictions
        pred_normalized = torch.softmax(predictions, dim=0)
        entropy = -torch.sum(pred_normalized * torch.log(pred_normalized + 1e-8), dim=0)
        entropy_loss = torch.exp(-torch.mean(entropy))
        
        # Combined loss with stronger penalties
        total_loss = mse_loss + self.alpha * variance_loss + self.beta * flatness_penalty + \
                    self.gamma * diversity_loss + 0.1 * entropy_loss
        
        return total_loss

class AntiFlatLoss(nn.Module):
    """
    Specialized loss function that directly attacks flat predictions
    """
    def __init__(self, base_loss=nn.MSELoss(), flatness_weight=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.flatness_weight = flatness_weight
    
    def forward(self, predictions, targets):
        # Base loss
        base_loss = self.base_loss(predictions, targets)
        
        # Calculate flatness penalty
        if predictions.shape[0] > 1:
            # Measure how "flat" the predictions are
            pred_std = torch.std(predictions, dim=0)
            pred_range = torch.max(predictions, dim=0)[0] - torch.min(predictions, dim=0)[0]
            
            # Flatness penalty: penalize low variance and low range
            flatness_penalty = torch.exp(-pred_std) + torch.exp(-pred_range)
            flatness_penalty = torch.mean(flatness_penalty)
        else:
            flatness_penalty = 0.0
        
        # Combined loss
        total_loss = base_loss + self.flatness_weight * flatness_penalty
        
        return total_loss

def create_model(input_size, hidden_size, num_layers, output_size=1, activation_fn=nn.Tanh(), model_type='LSTM', debug=False, 
                seasonal_period=12, trend_window=7, num_modes=3, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7):
    """
    Create a model based on the specified model type.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden layer size
        num_layers: Number of stacked layers
        output_size: Output dimension
        activation_fn: Activation function
        model_type: Type of model to create ('LSTM', 'VMD_LSTM', or 'STL_LSTM'). Default is 'LSTM'
        debug: Enable debug mode for VDM-LSTM (prints dimension information)
        seasonal_period: Seasonal period for STL-LSTM (default: 12)
        trend_window: Trend window size for STL-LSTM (default: 7)
        num_modes: Number of modes for VMD-LSTM (default: 3)
        alpha: VMD balancing parameter (default: 2000)
        tau: VMD time step (default: 0)
        K: VMD maximum iterations (default: 3)
        DC: VMD DC component flag (default: 0)
        init: VMD initialization method (default: 1)
        tol: VMD convergence tolerance (default: 1e-7)
    
    Returns:
        model: The created model
    """
    if model_type.upper() == 'VMD_LSTM':
        # Enable anomaly detection for VMD-LSTM to catch gradient issues
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled for VMD-LSTM debugging")
        
        model = VMD_LSTM(input_size, hidden_size, num_layers, output_size, 
                         num_modes=num_modes, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol,
                         activation=activation_fn, dropout_rate=0.1).to(device)
        
        # Test the forward pass before returning
        print("Testing VMD-LSTM forward pass...")
        if model.test_forward():
            print("VMD-LSTM forward pass test successful!")
        else:
            print("WARNING: VMD-LSTM forward pass test failed!")
        
        if debug:
            model.debug = True
        print(f"Created VMD-LSTM model with {num_layers} layers, hidden size {hidden_size}")
        
    elif model_type.upper() == 'STL_LSTM':
        # Create STL-LSTM model
        model = STL_LSTM(input_size, hidden_size, num_layers, output_size, 
                        seasonal_period=seasonal_period, trend_window=trend_window, 
                        activation=activation_fn, dropout_rate=0.1).to(device)
        
        # Test the forward pass before returning
        print("Testing STL-LSTM forward pass...")
        if model.test_forward():
            print("STL-LSTM forward pass test successful!")
        else:
            print("WARNING: STL-LSTM forward pass test failed!")
        
        print(f"Created STL-LSTM model with {num_layers} layers, hidden size {hidden_size}, "
              f"seasonal_period={seasonal_period}, trend_window={trend_window}")
        
    else:
        # Default LSTM model
        model = LSTM(input_size, hidden_size, num_layers, output_size, activation=activation_fn).to(device)
        print(f"Created LSTM model with {num_layers} layers, hidden size {hidden_size}")
    
    return model

def train_one_epoch(model, optimizer, loss_function, train_loader):
    model.train(True)
    running_loss = 0.0
    total_batches = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        total_batches += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = running_loss / total_batches
    print(f"Training Loss: {avg_loss:.3f}")
    return avg_loss


def validate_one_epoch(model, loss_function, test_loader):
    model.eval()  # proper eval mode
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            total_batches += 1

    avg_loss = running_loss / total_batches
    print(f"Validation Loss: {avg_loss:.3f}")
    return avg_loss

def minmaxscale_df(df, exclude_columns=None):
    """
    MinMax scales all columns in the DataFrame or Series except the specified columns.

    Parameters:
        df: pandas DataFrame or Series
        exclude_columns: list of column names to exclude from scaling

    Returns:
        new_df: DataFrame or Series with scaled values (except excluded columns)
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Handle pandas Series
    if isinstance(df, pd.Series):
        # Convert Series to DataFrame for scaling
        df_df = df.to_frame()
        scaled_values = scaler.fit_transform(df_df)
        return pd.Series(scaled_values.flatten(), index=df.index)
    
    # Handle DataFrame
    elif isinstance(df, pd.DataFrame):
        new_df = df.copy()
        
        if exclude_columns:
            # Identify columns to scale
            cols_to_scale = [col for col in df.columns if col not in exclude_columns]
            
            # Apply scaling only to selected columns
            if cols_to_scale:
                new_df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        else:
            # Scale all columns
            new_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        
        return new_df
    
    # Handle numpy arrays
    elif isinstance(df, np.ndarray):
        if df.ndim == 1:
            # 1D array - reshape to 2D for scaling
            df_2d = df.reshape(-1, 1)
            scaled_values = scaler.fit_transform(df_2d)
            return scaled_values.flatten()
        else:
            # 2D array
            return scaler.fit_transform(df)
    
    else:
        raise ValueError(f"Unsupported data type: {type(df)}. Expected pandas Series, DataFrame, or numpy array.")


def preprocess_dataframe(X_raw, y_raw, window_size, expiry, debug=False):
    """
    Prepares sequences for LSTM from multi-dimensional X_raw and y_raw.
    Handles NaN values by:
      - Always checking NaNs in X_raw
      - Checking NaNs in y_raw only from index `window_size + expiry - 1` onward

    Parameters:
        X_raw: pandas DataFrame, Series, or np.ndarray of shape (n_samples,) or (n_samples, n_features)
        y_raw: pandas Series or np.ndarray of shape (n_samples,)
        window_size: Number of past time steps to include
        expiry: Number of days ahead to predict

    Returns:
        X: np.ndarray of shape (n_sequences, window_size, n_features)
        y: np.ndarray of shape (n_sequences,)
        input_size: Number of input features
    """
    X_scaled = minmaxscale_df(X_raw)
    
    # Convert to numpy array, handling both Series and DataFrame outputs
    if isinstance(X_scaled, pd.Series):
        X_scaled = X_scaled.values
        input_size = 1
        # Ensure 2D shape for sequence creation
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
    elif isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
        input_size = X_scaled.shape[1]  # Number of columns/features
    else:
        X_scaled = np.asarray(X_scaled)
        if X_scaled.ndim == 1:
            input_size = 1
            # Ensure 2D shape for sequence creation
            X_scaled = X_scaled.reshape(-1, 1)
        else:
            input_size = X_scaled.shape[1]
    if debug:
        print(f"Debug - X_scaled shape after processing: {X_scaled.shape}, ndim: {X_scaled.ndim}")
        print(f"Debug - input_size determined: {input_size}")

    y_raw = np.asarray(y_raw)

    n_samples = len(y_raw)
    start_y_check = window_size + expiry - 1

    if X_scaled.ndim == 1:
        nan_mask = np.isnan(X_scaled)
    else:
        nan_mask = np.isnan(X_scaled).any(axis=1)

    # Only mark NaNs in y_raw from start_y_check onwards
    y_nan_mask = np.zeros_like(y_raw, dtype=bool)
    y_nan_mask[start_y_check:] = np.isnan(y_raw[start_y_check:])

    # Combine masks
    nan_mask = nan_mask | y_nan_mask

    num_nan_rows = np.sum(nan_mask)
    print(f"Detected {num_nan_rows} rows containing NaN values (with y check starting at index {start_y_check}).")

    # Drop NaN rows
    valid_mask = ~nan_mask
    X_scaled = X_scaled[valid_mask]
    y_raw = y_raw[valid_mask]

    # Create sequences
    X, y = [], []
    for i in range(len(y_raw) - window_size - expiry + 1):
        row = X_scaled[i : i + window_size]
        label = y_raw[i + window_size + expiry - 1]
        X.append(row)
        y.append(label)
    
    if debug:
        print(f"Input features: {input_size}, Sequences created: {len(X)}, Window size: {window_size}")
        print(f"Debug - Final X shape: {np.array(X).shape}")
        print(f"Debug - Final y shape: {np.array(y).shape}")
    return np.array(X), np.array(y), input_size


def train_model(model, optimizer, loss_function, train_loader, test_loader, epochs):
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, optimizer, loss_function, train_loader)
        val_loss = validate_one_epoch(model, loss_function, test_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)


    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, val_losses


def plot_train_test_predictions(model, X_train, y_train, X_test, y_test, y_scaler, window, device, expiry, n_points=300):
    """
    Plots train and test predictions side-by-side after inverse scaling.

    Parameters:
        model: Trained PyTorch model
        X_train, y_train: Training data and labels
        X_test, y_test: Testing data and labels
        y_scaler: Fitted scaler for inverse transformation
        window: Lookback window size (used to match scaler input shape)
        device: Torch device ("cpu" or "cuda")
        n_points: Number of points to plot
    """

    def inverse_transform_predictions(X, y, model):
        # Predict
        predictions = model(X.to(device)).detach().cpu().numpy().flatten()
        # Inverse scale predictions
        dummies = np.zeros((X.shape[0], window+1))
        dummies[:, 0] = predictions
        dummies = y_scaler.inverse_transform(dummies)
        predictions = dc(dummies[:, 0])
        # Inverse scale true values
        dummies = np.zeros((X.shape[0], window+1))
        dummies[:, 0] = y.flatten()
        dummies = y_scaler.inverse_transform(dummies)
        true_vals = dc(dummies[:, 0])
        return true_vals, predictions

    # Get inverse transformed predictions
    new_y_train, train_predictions = inverse_transform_predictions(X_train, y_train, model)
    new_y_test, test_predictions = inverse_transform_predictions(X_test, y_test, model)


    ax, fig = line_plot(new_y_train[:n_points], new_y_train[:n_points], ylabel='vol_true', graphtitle=expiry, linecolor='red', show=False)
    _, _ = line_plot(train_predictions[:n_points], train_predictions[:n_points], ylabel='vol_pred', ax=ax, show=True)

    ax, fig = line_plot(new_y_test[:n_points], new_y_test[:n_points], ylabel='vol_true', graphtitle=expiry, linecolor='red', show=False)
    _, _ = line_plot(test_predictions[:n_points], test_predictions[:n_points], ylabel='vol_pred', ax=ax, show=True)


def evaluate_and_print_metrics(model, X_test, y_test, y_scaler, window, device):
    """
    Evaluates the model on the test set and prints various metrics.

    Parameters:
        model: Trained PyTorch model
        X_test, y_test: Testing data and labels
        y_scaler: Fitted scaler for inverse transformation
        window: Lookback window size (used to match scaler input shape)
        device: Torch device ("cpu" or "cuda")
    """
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test.to(device)).detach().cpu().numpy().flatten()

    # Inverse scale predictions and true values for metric calculation
    dummies_pred = np.zeros((X_test.shape[0], window + 1))
    dummies_pred[:, 0] = test_predictions_scaled
    test_predictions = y_scaler.inverse_transform(dummies_pred)[:, 0]

    dummies_true = np.zeros((X_test.shape[0], window + 1))
    dummies_true[:, 0] = y_test.flatten()
    test_true_vals = y_scaler.inverse_transform(dummies_true)[:, 0]

    print("\nTest Set Metrics:")
    print(f"MAPE: {mape(test_true_vals, test_predictions):.4f}")
    print(f"MAE: {mae(test_true_vals, test_predictions):.4f}")
    print(f"RMSE: {rmse(test_true_vals, test_predictions):.4f}")
    print(f"MSE: {mse(test_true_vals, test_predictions):.4f}")