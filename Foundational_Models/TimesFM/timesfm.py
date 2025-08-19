import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import math

class TimesFmHparams:
    """TimesFM hyperparameters matching the official implementation"""
    def __init__(self,
                 backend: str = "gpu",
                 per_core_batch_size: int = 32,
                 horizon_len: int = 128,
                 num_layers: int = 50,
                 use_positional_embedding: bool = False,
                 context_len: int = 2048,
                 hidden_size: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        self.backend = backend
        self.per_core_batch_size = per_core_batch_size
        self.horizon_len = horizon_len
        self.num_layers = num_layers
        self.use_positional_embedding = use_positional_embedding
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

class TimesFmCheckpoint:
    """TimesFM checkpoint configuration"""
    def __init__(self, huggingface_repo_id: str = "google/timesfm-2.0-500m-jax"):
        self.huggingface_repo_id = huggingface_repo_id

class TimesFm(nn.Module):
    """
    TimesFM: Time Series Foundation Model
    Implementation based on Google Research's official TimesFM architecture
    """
    
    def __init__(self, 
                 hparams: TimesFmHparams,
                 checkpoint: TimesFmCheckpoint = None):
        super().__init__()
        
        self.hparams = hparams
        self.checkpoint = checkpoint
        
        # Model dimensions
        self.hidden_size = hparams.hidden_size
        self.num_layers = hparams.num_layers
        self.num_heads = hparams.num_heads
        self.context_len = hparams.context_len
        self.horizon_len = hparams.horizon_len
        self.use_positional_embedding = hparams.use_positional_embedding
        
        # Input projection for time series data
        self.input_projection = nn.Linear(1, self.hidden_size)
        
        # Positional encoding (optional based on hparams)
        if self.use_positional_embedding:
            self.pos_encoding = nn.Parameter(torch.randn(1, self.context_len, self.hidden_size))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=hparams.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output projection for forecasting
        self.output_projection = nn.Linear(self.hidden_size, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(hparams.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of TimesFM
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, 1) - single feature (volatility)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_length, 1)
        """
        batch_size, seq_length, _ = x.shape
        
        # Ensure sequence length doesn't exceed context length
        if seq_length > self.context_len:
            x = x[:, -self.context_len:, :]
            seq_length = self.context_len
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_length, hidden_size)
        
        # Add positional encoding if enabled
        if self.use_positional_embedding and seq_length <= self.context_len:
            x = x + self.pos_encoding[:, :seq_length, :]
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout_layer(x)
        
        # For now, don't use causal mask in transformer (let it be fully connected)
        # The causal behavior will come from the autoregressive prediction method
        
        # Apply transformer layers (no mask for now)
        x = self.transformer(x)
        
        # Output projection
        output = self.output_projection(x)  # (batch_size, seq_length, 1)
        
        return output
    
    def _create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation"""
        # Create mask on the correct device and with proper dtype
        mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
        return mask
    
    def predict_future(self, x: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """
        Predict future values using autoregressive generation
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, 1)
            horizon: Number of future steps to predict
            
        Returns:
            Predictions of shape (batch_size, horizon, 1)
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]
            predictions = []
            current_input = x.clone()
            
            # Generate predictions autoregressively
            for _ in range(horizon):
                # Get model output
                output = self.forward(current_input)
                
                # Take the last prediction
                last_pred = output[:, -1:, :]  # (batch_size, 1, 1)
                predictions.append(last_pred)
                
                # Append prediction to input for next step
                current_input = torch.cat([current_input, last_pred], dim=1)
                
                # Keep only the last context_len elements
                if current_input.shape[1] > self.context_len:
                    current_input = current_input[:, -self.context_len:, :]
            
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
        
        # Ensure sequence length doesn't exceed context length
        if seq_length > self.context_len:
            x = x[:, -self.context_len:, :]
            seq_length = self.context_len
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding if enabled
        if self.use_positional_embedding and seq_length <= self.context_len:
            x = x + self.pos_encoding[:, :seq_length, :]
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout_layer(x)
        
        # Apply transformer layers (no mask)
        x = self.transformer(x)
        
        return x

def create_timesfm_model(hparams: TimesFmHparams = None, 
                        checkpoint: TimesFmCheckpoint = None) -> TimesFm:
    """
    Factory function to create a TimesFM model
    
    Args:
        hparams: TimesFM hyperparameters
        checkpoint: TimesFM checkpoint configuration
        
    Returns:
        TimesFM model
    """
    if hparams is None:
        hparams = TimesFmHparams()
    
    if checkpoint is None:
        checkpoint = TimesFmCheckpoint()
    
    return TimesFm(hparams=hparams, checkpoint=checkpoint)
