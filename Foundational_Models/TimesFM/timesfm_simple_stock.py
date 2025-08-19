import torch
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add parent directories to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

from functions import pred_value_to_char, load_data, line_plot, mape, mae, rmse, mse, mase
from timesfm import create_timesfm_model, TimesFmHparams

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the data using the load_data function with proper path - use raw inputs for stock prices
data_path = os.path.join(parent_dir, 'Data', 'aluminium_raw_inputs.csv')
df = load_data(data_path)

# Create results directory structure
results_dir = os.path.join(os.path.dirname(__file__), 'simple_stock_results')
plots_dir = os.path.join(results_dir, 'plots')
metrics_dir = os.path.join(results_dir, 'metrics')

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

print(f"Results will be saved to: {results_dir}")
print(f"Plots will be saved to: {plots_dir}")
print(f"Metrics will be saved to: {metrics_dir}")

def preprocess_stock_data(df, expiry, window):
    """
    Preprocess data using only stock price feature
    
    Args:
        df: DataFrame with stock price data
        expiry: Prediction horizon
        window: Lookback window size
        
    Returns:
        X: Input features tensor (batch_size, seq_length, 1)
        y: Target values tensor
        scaler: Fitted scaler for inverse transformation
    """
    # Use the stock price column
    stock_column = 'al_lme_prices'
    
    if stock_column not in df.columns:
        raise ValueError(f"Stock price column {stock_column} not found in data")
    
    # Extract stock price data
    stock_data = df[stock_column].values
    
    # Remove rows with NaN values
    valid_indices = ~np.isnan(stock_data)
    stock_data = stock_data[valid_indices]
    
    # Create sequences
    X, y = [], []
    for i in range(window, len(stock_data) - expiry + 1):
        X.append(stock_data[i-window:i])
        y.append(stock_data[i+expiry-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to (batch_size, seq_length, 1) for single feature
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Scale features
    feature_scaler = StandardScaler()
    X_reshaped = X.reshape(-1, 1)
    X_scaled = feature_scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    # Scale target
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled), target_scaler

def evaluate_timesfm_stock(model, X_test, y_test, y_scaler, device):
    """
    Evaluate TimesFM model using only stock price feature
    
    Args:
        model: TimesFM model
        X_test: Test input features
        y_test: Test target values
        y_scaler: Scaler for inverse transformation
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    print(f"Debug: X_test shape: {X_test.shape}")
    print(f"Debug: y_test shape: {y_test.shape}")
    
    with torch.no_grad():
        # Move data to device
        X_test_device = X_test.to(device)
        
        # Get predictions
        try:
            predictions = model(X_test_device)
            print(f"Debug: Raw predictions shape: {predictions.shape}")
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            raise e
        
        # Take the last prediction for each sequence
        predictions = predictions[:, -1, :].cpu().numpy().flatten()
        print(f"Debug: Final predictions shape: {predictions.shape}")
        
        # Inverse transform predictions
        predictions_inv = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test_inv = y_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mape_score = mape(y_test_inv, predictions_inv)
        mae_score = mae(y_test_inv, predictions_inv)
        rmse_score = rmse(y_test_inv, predictions_inv)
        mse_score = mse(y_test_inv, predictions_inv)
        mase_score = mase(y_test_inv, predictions_inv)
        
        metrics = {
            'MAPE': mape_score,
            'MAE': mae_score,
            'RMSE': rmse_score,
            'MSE': mse_score,
            'MASE': mase_score
        }
        
        print(f"Test Set Metrics - MAPE: {mape_score:.2f}, MAE: {mae_score:.2f}, "
              f"RMSE: {rmse_score:.2f}, MSE: {mse_score:.4f}, MASE: {mase_score:.2f}")
        
        return metrics, predictions_inv, y_test_inv

def plot_predictions_using_line_plot(y_true, y_pred, expiry, save_path):
    """
    Plot true vs predicted values using line_plot function
    
    Args:
        y_true: True values
        y_pred: Predicted values
        expiry: Prediction horizon
        save_path: Path to save the plot
    """
    # Create time index
    time_index = range(len(y_true))
    
    # Use line_plot function like in LSTM.py
    # Plot predictions in blue first
    ax, fig = line_plot(time_index, y_pred, ylabel='pred_stock', 
                        graphtitle=f'TimesFM Stock: True vs Predicted {pred_value_to_char(expiry)}_stock', 
                        linecolor='blue', show=False)
    # Plot true values in red
    _, _ = line_plot(time_index, y_true, ylabel='true_stock', linecolor='red', ax=ax, show=True)
    
    # Save plot
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_timesfm_stock_experiment(expiry):
    """
    Run TimesFM experiment using only stock price feature
    
    Args:
        expiry: Prediction horizon
        
    Returns:
        Dictionary with evaluation metrics
    """
    window = expiry * 2
    
    print(f"\nRunning TimesFM Stock experiment for {pred_value_to_char(expiry)} expiry")
    print(f"Using only stock price feature: al_lme_prices")
    print(f"Window size: {window}")
    
    # Preprocess data using only stock price
    X_raw, y_raw, y_scaler = preprocess_stock_data(
        df, expiry, window
    )
    
    # Split data (use last 20% for testing)
    split_idx = int(0.8 * len(X_raw))
    X_train, X_test = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create TimesFM model with official architecture
    hparams = TimesFmHparams(
        backend="gpu" if device.startswith('cuda') else "cpu",
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=50,
        use_positional_embedding=False,
        context_len=2048,
        hidden_size=512,
        num_heads=8,
        dropout=0.1
    )
    
    model = create_timesfm_model(hparams=hparams).to(device)
    
    print(f"Created TimesFM model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model architecture: {hparams.num_layers} layers, {hparams.hidden_size} hidden size, {hparams.num_heads} heads")
    
    # Evaluate model (no training - zero-shot)
    metrics, predictions, y_test_inv = evaluate_timesfm_stock(
        model, X_test, y_test, y_scaler, device
    )
    
    # Create and save prediction plot using line_plot
    plot_path = os.path.join(plots_dir, f"{expiry}_stock_vs_true_test_TimesFM_stock.png")
    print(f"Saving plot to: {plot_path}")
    plot_predictions_using_line_plot(y_test_inv, predictions, expiry, plot_path)
    print(f"Plot saved successfully!")
    
    return metrics

def main():
    """Main function to run TimesFM stock experiments"""
    print("Starting TimesFM Stock Experiments (Stock Price Only)")
    print("=" * 60)
    
    # Run experiments for different expiries
    all_metrics = {}
    
    for expiry in [5, 22, 66, 252]:
        print(f"\n{'='*20} {pred_value_to_char(expiry)} Expiry {'='*20}")
        
        try:
            metrics = run_timesfm_stock_experiment(expiry=expiry)
            all_metrics[expiry] = metrics
        except Exception as e:
            print(f"Error in {pred_value_to_char(expiry)} expiry experiment: {e}")
            all_metrics[expiry] = None
    
    # Save all metrics to CSV
    if all_metrics:
        metrics_data = []
        for expiry, metrics in all_metrics.items():
            if metrics:
                metrics_data.append({
                    'Expiry': pred_value_to_char(expiry),
                    'MAPE': metrics['MAPE'],
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MSE': metrics['MSE'],
                    'MASE': metrics['MASE']
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_csv_path = os.path.join(metrics_dir, 'metrics_simple_stock.csv')
            print(f"Saving metrics to: {metrics_csv_path}")
            metrics_df.to_csv(metrics_csv_path, index=False)
            print(f"Metrics saved successfully!")
            print(f"\nAll metrics saved to: {metrics_csv_path}")
        else:
            print("No metrics data to save")
    else:
        print("No experiments completed successfully")
    
    print("\n" + "=" * 60)
    print("All TimesFM Stock experiments completed!")

if __name__ == "__main__":
    main()
