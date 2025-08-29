from os import path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import sys
import os

# Add the parent directories to the Python path to import functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to root
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from functions import load_data
from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
import os


class TimeSeriesDataset(Dataset):
  """Dataset for time series data compatible with TimesFM."""

  def __init__(self,
               series: np.ndarray,
               context_length: int,
               horizon_length: int,
               freq_type: int = 0):
    """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
    if freq_type not in [0, 1, 2]:
      raise ValueError("freq_type must be 0, 1, or 2")

    self.series = series
    self.context_length = context_length
    self.horizon_length = horizon_length
    self.freq_type = freq_type
    self._prepare_samples()

  def _prepare_samples(self) -> None:
    """Prepare sliding window samples from the time series."""
    self.samples = []
    total_length = self.context_length + self.horizon_length

    for start_idx in range(0, len(self.series) - total_length + 1):
      end_idx = start_idx + self.context_length
      x_context = self.series[start_idx:end_idx]
      x_future = self.series[end_idx:end_idx + self.horizon_length]
      self.samples.append((x_context, x_future))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(
      self, index: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_context, x_future = self.samples[index]

    x_context = torch.tensor(x_context, dtype=torch.float32)
    x_future = torch.tensor(x_future, dtype=torch.float32)

    input_padding = torch.zeros_like(x_context)
    # Frequency should match the context length for proper broadcasting
    freq = torch.full((self.context_length,), self.freq_type, dtype=torch.long)

    return x_context, input_padding, freq, x_future

def prepare_datasets(series: np.ndarray,
                     context_length: int,
                     horizon_length: int,
                     freq_type: int = 0,
                     train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
  """
    Prepare training and validation datasets from time series data.

    Args:
        series: Input time series data
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
  train_size = int(len(series) * train_split)
  train_data = series[:train_size]
  val_data = series[train_size:]

  # Create datasets with specified frequency type
  train_dataset = TimeSeriesDataset(train_data,
                                    context_length=context_length,
                                    horizon_length=horizon_length,
                                    freq_type=freq_type)

  val_dataset = TimeSeriesDataset(val_data,
                                  context_length=context_length,
                                  horizon_length=horizon_length,
                                  freq_type=freq_type)

  return train_dataset, val_dataset


def get_model(load_weights: bool = False):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  repo_id = "google/timesfm-2.0-500m-pytorch"
  hparams = TimesFmHparams(
      backend=device,
      per_core_batch_size=32,
      horizon_len=128,
      num_layers=50,
      use_positional_embedding=False,
      context_len=
      192,  # Context length can be anything up to 2048 in multiples of 32
  )
  tfm = TimesFm(hparams=hparams,
                checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

  model = PatchedTimeSeriesDecoder(tfm._model_config)
  if load_weights:
    checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
    loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(loaded_checkpoint)
  return model, hparams, tfm._model_config


def plot_predictions(
    model: TimesFm,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
  """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
      model: Trained TimesFM model
      val_dataset: Validation dataset
      save_path: Path to save the plot
    """
  import matplotlib.pyplot as plt

  model.eval()

  x_context, x_padding, freq, x_future = val_dataset[0]
  x_context = x_context.unsqueeze(0)  # Add batch dimension
  x_padding = x_padding.unsqueeze(0)
  freq = freq.unsqueeze(0)
  x_future = x_future.unsqueeze(0)

  device = next(model.parameters()).device
  x_context = x_context.to(device)
  x_padding = x_padding.to(device)
  freq = freq.to(device)
  x_future = x_future.to(device)

  with torch.no_grad():
    predictions = model(x_context, x_padding.float(), freq)
    predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
    last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

  context_vals = x_context[0].cpu().numpy()
  future_vals = x_future[0].cpu().numpy()
  pred_vals = last_patch_pred[0].cpu().numpy()

  context_len = len(context_vals)
  horizon_len = len(future_vals)

  plt.figure(figsize=(12, 6))

  plt.plot(range(context_len),
           context_vals,
           label="Historical Data",
           color="blue",
           linewidth=2)

  plt.plot(
      range(context_len, context_len + horizon_len),
      future_vals,
      label="Ground Truth",
      color="green",
      linestyle="--",
      linewidth=2,
  )

  plt.plot(range(context_len, context_len + horizon_len),
           pred_vals,
           label="Prediction",
           color="red",
           linewidth=2)

  plt.xlabel("Time Step")
  plt.ylabel("Value")
  plt.title("TimesFM Predictions vs Ground Truth")
  plt.legend()
  plt.grid(True)

  if save_path:
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

  plt.close()  

def get_data(context_len: int,
             horizon_len: int,
             freq_type: int = 0) -> Tuple[Dataset, Dataset]:
  # Load aluminium data using load_data function
  data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
  df = load_data(data_path)
  
  # Extract 1w_vol column and drop NaN values
  time_series = df['1w_vol'].dropna().values

  train_dataset, val_dataset = prepare_datasets(
      series=time_series,
      context_length=context_len,
      horizon_length=horizon_len,
      freq_type=freq_type,
      train_split=0.8,
  )

  print(f"Created datasets:")
  print(f"- Training samples: {len(train_dataset)}")
  print(f"- Validation samples: {len(val_dataset)}")
  print(f"- Using frequency type: {freq_type}")
  return train_dataset, val_dataset



def single_gpu_example():
  """Basic example of finetuning TimesFM on volatility data."""
  model, hparams, tfm_config = get_model(load_weights=True)
  
  # Use frequency type 0 which is supported by the model
  # The model only supports frequency types 0-5, and type 1 causes dimension mismatch
  config = FinetuningConfig(batch_size=128,  # Reduced batch size
                            num_epochs=5,
                            learning_rate=1e-4,
                            use_wandb=False,  # Disable wandb for now
                            freq_type=0,      # Use frequency type 0 (supported by model)
                            log_every_n_steps=10,
                            val_check_interval=0.5,
                            use_quantile_loss=True)

  # Use shorter context length for better compatibility
  context_len = 64
  print(f"Using context length: {context_len}")
  print(f"Using frequency type: {config.freq_type}")
  
  train_dataset, val_dataset = get_data(context_len,
                                        tfm_config.horizon_len,
                                        freq_type=config.freq_type)
  
  # Debug: Check tensor shapes
  if len(train_dataset) > 0:
    sample = train_dataset[0]
    print(f"Sample tensor shapes:")
    print(f"- x_context: {sample[0].shape}")
    print(f"- x_padding: {sample[1].shape}")
    print(f"- freq: {sample[2].shape}")
    print(f"- x_future: {sample[3].shape}")
    print(f"- freq values: {sample[2]}")
    print(f"- freq dtype: {sample[2].dtype}")
    
    # Test a single forward pass to debug the issue
    print("\nTesting single forward pass...")
    model.eval()
    with torch.no_grad():
      try:
        # Add batch dimension
        x_context = sample[0].unsqueeze(0).to(next(model.parameters()).device)
        x_padding = sample[1].unsqueeze(0).to(next(model.parameters()).device)
        freq = sample[2].unsqueeze(0).to(next(model.parameters()).device)
        
        print(f"Input shapes for forward pass:")
        print(f"- x_context: {x_context.shape}")
        print(f"- x_padding: {x_padding.shape}")
        print(f"- freq: {freq.shape}")
        
        predictions = model(x_context, x_padding.float(), freq)
        print(f"Forward pass successful! Output shape: {predictions.shape}")
        
      except Exception as e:
        print(f"Forward pass failed: {e}")
        print("This suggests the model input format is still incorrect")
        raise
  
  finetuner = TimesFMFinetuner(model, config)

  print("\nStarting finetuning...")
  results = finetuner.finetune(train_dataset=train_dataset,
                               val_dataset=val_dataset)

  print("\nFinetuning completed!")
  print(f"Training history: {len(results['history']['train_loss'])} epochs")

  plot_predictions(
      model=model,
      val_dataset=val_dataset,
      save_path="timesfm_predictions.png",
  )

single_gpu_example()