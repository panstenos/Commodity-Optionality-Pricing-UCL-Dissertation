from os import path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import yfinance as yf
from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
import os
import matplotlib.pyplot as plt

def get_model(load_weights: bool = False):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	repo_id = "google/timesfm-2.0-500m-pytorch"
	hparams = TimesFmHparams(
		backend=device,
		per_core_batch_size=32,
		horizon_len=128,
		num_layers=50,
		use_positional_embedding=False,
		context_len=192, # multiples of 32
	)
	tfm = TimesFm(hparams=hparams, checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

	model = PatchedTimeSeriesDecoder(tfm._model_config)
	if load_weights:
		checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
		loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
		model.load_state_dict(loaded_checkpoint)
	return model, hparams, tfm._model_config

def plot_predictions(model, val_dataset, save_path="predictions.png"):
	"""
		Plot model predictions against ground truth for a batch of validation data.

		Args:
			model: Trained TimesFM model
			val_dataset: Validation dataset
			save_path: Path to save the plot
		"""

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

def plot_predictions(model, val_dataset, save_path="predictions.png"):
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
	df = yf.download("AAPL", start="2010-01-01", end="2019-01-01")
	time_series = df["Close"].values

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
	"""Basic example of finetuning TimesFM on stock data."""
	model, hparams, tfm_config = get_model(load_weights=True)
	config = FinetuningConfig(batch_size=256,
							num_epochs=5,
							learning_rate=1e-4,
							use_wandb=True,
							freq_type=1,
							log_every_n_steps=10,
							val_check_interval=0.5,
							use_quantile_loss=True)

	train_dataset, val_dataset = get_data(128,
										tfm_config.horizon_len,
										freq_type=config.freq_type)
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