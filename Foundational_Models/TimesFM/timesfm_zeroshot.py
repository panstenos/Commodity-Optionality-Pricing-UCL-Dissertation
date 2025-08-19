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
neural_nets_dir = os.path.abspath(os.path.join(parent_dir, 'Neural_Nets'))
sys.path.insert(0, neural_nets_dir)

from functions import pred_value_to_char, load_data, line_plot, mape, mae, rmse, mse, mase
from timesfm import create_timesfm_model

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the data using the load_data function with proper path
data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
df = load_data(data_path)

# Create results directory structure
results_dir = os.path.join(os.path.dirname(__file__), 'zeroshot_results')
plots_dir = os.path.join(results_dir, 'plots')
metrics_dir = os.path.join(results_dir, 'metrics')

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

def find_n_best_features(expiry, n):
    """Find the n best features based on correlation with volatility"""
    corr_file = os.path.join(parent_dir, 'Feature_selection', 'absolute_feature_correlations.csv')
    best_features = load_data(corr_file, index_col=0)
    top_rows = best_features.sort_values(by=f'{pred_value_to_char(expiry)}_exp', ascending=False).head(n)
    best_features = top_rows.index.tolist()
    return best_features

def preprocess_data_for_timesfm(df, feature_columns, target_column, window, expiry):
    """
    Preprocess data for TimesFM model
    
    Args:
        df: DataFrame with features and target
        feature_columns: List of feature column names or single column name
        target_column: Target column name
        window: Lookback window size
        expiry: Prediction horizon
        
    Returns:
        X: Input features tensor
        y: Target values tensor
        scaler: Fitted scaler for inverse transformation
    """
    # Handle single feature column vs list of features
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    
    # Prepare features and target
    features = df[feature_columns].values
    target = df[target_column].values
    
    # Remove rows with NaN values
    valid_indices = ~(np.isnan(features).any(axis=1) | np.isnan(target))
    features = features[valid_indices]
    target = target[valid_indices]
    
    # Create sequences
    X, y = [], []
    for i in range(window, len(features) - expiry + 1):
        X.append(features[i-window:i])
        y.append(target[i+expiry-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Scale features
    feature_scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = feature_scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    # Scale target
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled), target_scaler

def evaluate_timesfm_zeroshot(model, X_test, y_test, y_scaler, window, device):
    """
    Evaluate TimesFM model in zero-shot mode
    
    Args:
        model: TimesFM model
        X_test: Test input features
        y_test: Test target values
        y_scaler: Scaler for inverse transformation
        window: Lookback window
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    print(f"Debug: X_test shape: {X_test.shape}")
    print(f"Debug: y_test shape: {y_test.shape}")
    print(f"Debug: Device: {device}")
    
    with torch.no_grad():
        # Move data to device
        X_test_device = X_test.to(device)
        print(f"Debug: X_test_device shape: {X_test_device.shape}")
        
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
        
        print(f"Debug: Predictions_inv shape: {predictions_inv.shape}")
        print(f"Debug: y_test_inv shape: {y_test_inv.shape}")
        
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
        
        print(f"Zero-shot Test Set Metrics - MAPE: {mape_score:.2f}, MAE: {mae_score:.2f}, "
              f"RMSE: {rmse_score:.2f}, MSE: {mse_score:.4f}, MASE: {mase_score:.2f}")
        
        return metrics, predictions_inv, y_test_inv

def plot_predictions_using_line_plot(y_true, y_pred, expiry, features_names_suffix, save_path):
    """
    Plot true vs predicted values using line_plot function
    
    Args:
        y_true: True values
        y_pred: Predicted values
        expiry: Prediction horizon
        features_names_suffix: Suffix for plot title
        save_path: Path to save the plot
    """
    # Create time index
    time_index = range(len(y_true))
    
    # Use line_plot function like in LSTM.py
    # Plot predictions in blue first
    ax, fig = line_plot(time_index, y_pred, ylabel='pred_vol', 
                        graphtitle=f'TimesFM Zero-shot: True vs Predicted {features_names_suffix}', 
                        linecolor='blue', show=False)
    # Plot true values in red
    _, _ = line_plot(time_index, y_true, ylabel='true_vol', linecolor='red', ax=ax, show=True)
    
    # Save plot
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def TimesFM_zeroshot_experiment(expiry, features_names, features_names_suffix):
    """
    Run TimesFM zero-shot experiment
    
    Args:
        expiry: Prediction horizon
        features_names: Feature column names
        features_names_suffix: Suffix for naming results
        
    Returns:
        Dictionary with evaluation metrics
    """
    window = expiry * 2
    
    print(f"\nRunning TimesFM Zero-shot experiment for {pred_value_to_char(expiry)} expiry")
    print(f"Features: {features_names}")
    print(f"Window size: {window}")
    
    # Preprocess data
    X_raw, y_raw, y_scaler = preprocess_data_for_timesfm(
        df, features_names, f'{pred_value_to_char(expiry)}_vol', window, expiry
    )
    
    # Split data (use last 20% for testing in zero-shot scenario)
    split_idx = int(0.8 * len(X_raw))
    X_train, X_test = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create TimesFM model
    input_size = X_raw.shape[-1]
    model = create_timesfm_model(
        input_size=input_size,
        hidden_size=64,   # Smaller for stability
        num_layers=2,     # Fewer layers for zero-shot
        num_heads=4,      # Fewer heads
        dropout=0.1,
        max_seq_length=1000
    ).to(device)
    
    print(f"Created TimesFM model with input_size={input_size}")
    
    # Evaluate in zero-shot mode (no training)
    metrics, predictions, y_test_inv = evaluate_timesfm_zeroshot(
        model, X_test, y_test, y_scaler, window, device
    )
    
    # Create and save prediction plot using line_plot
    plot_path = os.path.join(plots_dir, f"{expiry}_vol_vs_true_test_TimesFM_zeroshot_{features_names_suffix}.png")
    plot_predictions_using_line_plot(y_test_inv, predictions, expiry, features_names_suffix, plot_path)
    
    return metrics

def main():
    """Main function to run all zero-shot experiments"""
    print("Starting TimesFM Zero-shot Experiments")
    print("=" * 50)
    
    # Run experiments for different expiries
    for expiry in [5, 22, 66, 252]:
        names = ['log_returns', 'best_metric', 'best_5', 'best_10', 'best_20']
        best_features = find_n_best_features(expiry, 20)
        
        print(f"\n{'='*20} {pred_value_to_char(expiry)} Expiry {'='*20}")
        
        # 1. abs_log_returns
        metrics1 = TimesFM_zeroshot_experiment(
            expiry=expiry, 
            features_names='al_lme_prices_abs_log_returns', 
            features_names_suffix=f'TimesFM_zeroshot_{names[0]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # 2. best_metric
        metrics2 = TimesFM_zeroshot_experiment(
            expiry=expiry, 
            features_names=best_features[0], 
            features_names_suffix=f'TimesFM_zeroshot_{names[1]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # 3. best_5
        metrics3 = TimesFM_zeroshot_experiment(
            expiry=expiry, 
            features_names=best_features[:5], 
            features_names_suffix=f'TimesFM_zeroshot_{names[2]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # 4. best_10
        metrics4 = TimesFM_zeroshot_experiment(
            expiry=expiry, 
            features_names=best_features[:10], 
            features_names_suffix=f'TimesFM_zeroshot_{names[3]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # 5. best_20
        metrics5 = TimesFM_zeroshot_experiment(
            expiry=expiry, 
            features_names=best_features[:20], 
            features_names_suffix=f'TimesFM_zeroshot_{names[4]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # Combine all metrics for this expiry
        expiry_metrics = {
            'Feature_Selection': names,
            'MAPE': [metrics1['MAPE'], metrics2['MAPE'], metrics3['MAPE'], metrics4['MAPE'], metrics5['MAPE']],
            'MAE': [metrics1['MAE'], metrics2['MAE'], metrics3['MAE'], metrics4['MAE'], metrics5['MAE']],
            'RMSE': [metrics1['RMSE'], metrics2['RMSE'], metrics3['RMSE'], metrics4['RMSE'], metrics5['RMSE']],
            'MSE': [metrics1['MSE'], metrics2['MSE'], metrics3['MSE'], metrics4['MSE'], metrics5['MSE']],
            'MASE': [metrics1['MASE'], metrics2['MASE'], metrics3['MASE'], metrics4['MASE'], metrics5['MASE']]
        }
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(expiry_metrics)
        metrics_csv_path = os.path.join(metrics_dir, f'metrics_zeroshot_{pred_value_to_char(expiry)}.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Zero-shot metrics for {pred_value_to_char(expiry)} saved to: {metrics_csv_path}")
    
    print("\n" + "=" * 50)
    print("All TimesFM Zero-shot experiments completed!")

if __name__ == "__main__":
    main()
