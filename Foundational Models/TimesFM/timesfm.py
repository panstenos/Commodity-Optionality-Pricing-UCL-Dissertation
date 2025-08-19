import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import math

class TimesFM(nn.Module):
    """
    TimesFM: Time Series Foundation Model
    A transformer-based architecture specifically designed for time series forecasting
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_length: int = 1000,
                 use_positional_encoding: bool = True,
                 use_fourier_features: bool = True):
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.use_positional_encoding = use_positional_encoding
        self.use_fourier_features = use_fourier_features
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Fourier features for time encoding
        if use_fourier_features:
            self.fourier_projection = nn.Linear(2 * (max_seq_length // 4), hidden_size)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, hidden_size))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_fourier_features(self, seq_length: int) -> torch.Tensor:
        """Create Fourier features for time encoding"""
        freqs = torch.arange(1, seq_length // 4 + 1, dtype=torch.float32)
        time_steps = torch.arange(seq_length, dtype=torch.float32)
        
        # Create sin and cos features
        sin_features = torch.sin(2 * math.pi * freqs.unsqueeze(0) * time_steps.unsqueeze(1) / seq_length)
        cos_features = torch.cos(2 * math.pi * freqs.unsqueeze(0) * time_steps.unsqueeze(1) / seq_length)
        
        # Concatenate and flatten
        fourier_features = torch.cat([sin_features, cos_features], dim=1).T  # (2*freqs, seq_len)
        return fourier_features
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of TimesFM
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_length, 1)
        """
        batch_size, seq_length, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_length, hidden_size)
        
        # Add Fourier features if enabled
        if self.use_fourier_features and seq_length <= self.max_seq_length:
            fourier_features = self._create_fourier_features(seq_length).to(x.device)
            fourier_features = fourier_features.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 2*freqs, seq_len)
            fourier_features = self.fourier_projection(fourier_features.T).T  # (batch_size, seq_length, hidden_size)
            x = x + fourier_features
        
        # Add positional encoding if enabled
        if self.use_positional_encoding and seq_length <= self.max_seq_length:
            x = x + self.pos_encoding[:, :seq_length, :]
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout_layer(x)
        
        # Create attention mask if not provided
        if mask is None:
            mask = self._create_causal_mask(seq_length).to(x.device)
        
        # Apply transformer layers
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Output projection
        output = self.output_projection(x)  # (batch_size, seq_length, 1)
        
        return output
    
    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask
    
    def predict_future(self, x: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """
        Predict future values using autoregressive generation
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            horizon: Number of future steps to predict
            
        Returns:
            Predictions of shape (batch_size, horizon, 1)
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]
            predictions = []
            
            # Generate predictions autoregressively
            for _ in range(horizon):
                # Get model output
                output = self.forward(x)
                
                # Take the last prediction
                last_pred = output[:, -1:, :]  # (batch_size, 1, 1)
                predictions.append(last_pred)
                
                # Append prediction to input for next step
                # Create dummy features for the predicted value (you might want to adjust this)
                dummy_features = torch.zeros(batch_size, 1, self.input_size, device=x.device)
                x = torch.cat([x, dummy_features], dim=1)
            
            return torch.cat(predictions, dim=1)  # (batch_size, horizon, 1)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from the last transformer layer
        
        Args:
            x: Input tensor
            
        Returns:
            Embeddings of shape (batch_size, seq_length, hidden_size)
        """
        # Get embeddings before the final output projection
        batch_size, seq_length, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add Fourier features if enabled
        if self.use_fourier_features and seq_length <= self.max_seq_length:
            fourier_features = self._create_fourier_features(seq_length).to(x.device)
            fourier_features = fourier_features.unsqueeze(0).expand(batch_size, -1, -1)
            fourier_features = self.fourier_projection(fourier_features.T).T
            x = x + fourier_features
        
        # Add positional encoding if enabled
        if self.use_positional_encoding and seq_length <= self.max_seq_length:
            x = x + self.pos_encoding[:, :seq_length, :]
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout_layer(x)
        
        # Apply transformer layers
        x = self.transformer(x)
        
        return x


class TimesFMForForecasting(nn.Module):
    """
    TimesFM wrapper specifically for time series forecasting tasks
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_length: int = 1000,
                 use_positional_encoding: bool = True,
                 use_fourier_features: bool = True):
        
        super().__init__()
        
        self.timesfm = TimesFM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
            use_positional_encoding=use_positional_encoding,
            use_fourier_features=use_fourier_features
        )
        
        # Additional forecasting head
        self.forecasting_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize forecasting head
        for module in self.forecasting_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for forecasting
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            mask: Optional attention mask
            
        Returns:
            Forecast of shape (batch_size, seq_length, 1)
        """
        # Get embeddings from TimesFM
        embeddings = self.timesfm.get_embeddings(x)
        
        # Apply forecasting head
        forecast = self.forecasting_head(embeddings)
        
        return forecast
    
    def predict_future(self, x: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """Predict future values"""
        return self.timesfm.predict_future(x, horizon)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings"""
        return self.timesfm.get_embeddings(x)


def create_timesfm_model(input_size: int, 
                        hidden_size: int = 256,
                        num_layers: int = 6,
                        num_heads: int = 8,
                        dropout: float = 0.1,
                        max_seq_length: int = 1000) -> TimesFMForForecasting:
    """
    Factory function to create a TimesFM model for forecasting
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        max_seq_length: Maximum sequence length
        
    Returns:
        TimesFMForForecasting model
    """
    return TimesFMForForecasting(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_length=max_seq_length
    )
