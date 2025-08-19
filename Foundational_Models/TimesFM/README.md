# TimesFM: Time Series Foundation Model

This directory contains implementations of TimesFM (Time Series Foundation Model) for both zero-shot and fine-tuning approaches to time series forecasting.

## Overview

TimesFM is a transformer-based architecture specifically designed for time series forecasting tasks. It combines:

- **Multi-head attention mechanisms** for capturing temporal dependencies
- **Fourier features** for time encoding
- **Positional encoding** for sequence position awareness
- **Transformer layers** for complex pattern recognition

## Files

### Core Model
- `timesfm.py` - The main TimesFM model implementation with `TimesFM` and `TimesFMForForecasting` classes

### Experiment Scripts
- `timesfm_zeroshot.py` - Zero-shot evaluation without training
- `timesfm_finetuned.py` - Fine-tuning experiments with training

### Configuration
- `requirements.txt` - Required Python packages
- `README.md` - This documentation file

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have access to the data files and utility functions from the parent directories.

## Usage

### Zero-Shot Evaluation

Run the zero-shot experiments to evaluate TimesFM without training:

```bash
python timesfm_zeroshot.py
```

This will:
- Load the aluminium price data
- Create TimesFM models with random weights
- Evaluate performance on test sets
- Save results to `zeroshot_results/` directory

### Fine-Tuning Experiments

Run the fine-tuning experiments to train TimesFM on your data:

```bash
python timesfm_finetuned.py
```

This will:
- Load and preprocess the data
- Train TimesFM models for 30 epochs
- Evaluate performance on test sets
- Save training plots and results to `finetuned_results/` directory

## Model Architecture

### TimesFM Class
- **Input Projection**: Linear layer to project input features to hidden dimension
- **Fourier Features**: Time encoding using sin/cos transformations
- **Positional Encoding**: Learnable positional embeddings
- **Transformer Encoder**: Multi-layer transformer with attention
- **Output Projection**: Linear layer for final predictions

### TimesFMForForecasting Class
- Wrapper around TimesFM with additional forecasting head
- Includes dropout and layer normalization for better training stability

## Hyperparameters

Default configuration:
- **Hidden Size**: 128
- **Number of Layers**: 4
- **Number of Heads**: 8
- **Dropout**: 0.1
- **Max Sequence Length**: 1000
- **Learning Rate**: 0.001
- **Batch Size**: 64 (fine-tuning), variable (zero-shot)
- **Epochs**: 30 (fine-tuning)

## Data Processing

The models expect:
- **Input**: Time series sequences with shape `(batch_size, seq_length, num_features)`
- **Target**: Volatility values for different prediction horizons (1w, 1m, 3m, 1y)
- **Window Size**: `expiry * 2` for lookback period

## Feature Selection

The experiments test different feature combinations:
1. **Log Returns**: Absolute log returns of aluminium prices
2. **Best Metric**: Single best correlated feature
3. **Best 5**: Top 5 correlated features
4. **Best 10**: Top 10 correlated features
5. **Best 20**: Top 20 correlated features

## Output

### Metrics
- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MSE**: Mean Squared Error
- **MASE**: Mean Absolute Scaled Error

### Plots
- Training history (loss curves)
- True vs predicted values
- All plots are saved automatically (no display during execution)

### Results
- CSV files with metrics for each expiry period
- Organized by feature selection method
- Separate results for zero-shot and fine-tuned approaches

## Performance Considerations

- **Memory**: Transformer models can be memory-intensive; adjust batch size if needed
- **Training Time**: Fine-tuning takes longer than zero-shot evaluation
- **GPU**: CUDA acceleration is automatically detected and used if available

## Customization

To modify the model architecture:
1. Edit the hyperparameters in the experiment functions
2. Modify the `create_timesfm_model` function in `timesfm.py`
3. Adjust the data preprocessing in the experiment scripts

## Troubleshooting

- **CUDA Errors**: Reduce batch size or model complexity
- **Memory Issues**: Decrease hidden size or number of layers
- **Import Errors**: Ensure all parent directories are in the Python path
- **Data Issues**: Check that the data files exist and have the expected format

## Comparison with LSTM

TimesFM offers several advantages over traditional LSTM models:
- **Attention Mechanism**: Better capture of long-range dependencies
- **Parallel Processing**: Faster training and inference
- **Scalability**: Handles longer sequences more effectively
- **Interpretability**: Attention weights provide insights into model decisions

However, it may require more data and computational resources for optimal performance.
