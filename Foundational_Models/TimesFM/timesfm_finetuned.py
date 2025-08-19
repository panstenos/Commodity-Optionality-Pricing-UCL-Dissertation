import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
results_dir = os.path.join(os.path.dirname(__file__), 'finetuned_results')
plots_dir = os.path.join(results_dir, 'plots')
training_plots_dir = os.path.join(results_dir, 'training_plots')
metrics_dir = os.path.join(results_dir, 'metrics')

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(training_plots_dir, exist_ok=True)
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

def create_dataloaders(X, y, batch_size, train_ratio=0.8, val_ratio=0.1):
    """
    Create train, validation, and test dataloaders
    
    Args:
        X: Input features tensor
        y: Target values tensor
        batch_size: Batch size for training
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        
    Returns:
        train_loader, val_loader, test_loader, y_scaler
    """
    # Split data
    total_samples = len(X)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test

def train_timesfm_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    """
    Train TimesFM model
    
    Args:
        model: TimesFM model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        train_losses, val_losses
    """
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Take the last prediction for each sequence
            outputs = outputs[:, -1, :].squeeze(-1)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                outputs = outputs[:, -1, :].squeeze(-1)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Training Loss: {avg_train_loss:.3f}")
            print(f"Validation Loss: {avg_val_loss:.3f}")
    
    return train_losses, val_losses

def evaluate_timesfm_model(model, test_loader, y_scaler, device):
    """
    Evaluate TimesFM model
    
    Args:
        model: TimesFM model
        test_loader: Test data loader
        y_scaler: Scaler for inverse transformation
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            
            outputs = model(batch_X)
            outputs = outputs[:, -1, :].cpu().numpy().flatten()
            
            all_predictions.extend(outputs)
            all_targets.extend(batch_y.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Inverse transform predictions
    predictions_inv = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_inv = y_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mape_score = mape(targets_inv, predictions_inv)
    mae_score = mae(targets_inv, predictions_inv)
    rmse_score = rmse(targets_inv, predictions_inv)
    mse_score = mse(targets_inv, predictions_inv)
    mase_score = mase(targets_inv, predictions_inv)
    
    metrics = {
        'MAPE': mape_score,
        'MAE': mae_score,
        'RMSE': rmse_score,
        'MSE': mse_score,
        'MASE': mase_score
    }
    
    print(f"Fine-tuned Test Set Metrics - MAPE: {mape_score:.2f}, MAE: {mae_score:.2f}, "
          f"RMSE: {rmse_score:.2f}, MSE: {mse_score:.4f}, MASE: {mase_score:.2f}")
    
    return metrics, predictions_inv, targets_inv

def plot_training_history_using_line_plot(train_losses, val_losses, features_names_suffix, save_path):
    """
    Plot training history using line_plot function
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        features_names_suffix: Suffix for plot title
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Use line_plot function like in LSTM.py
    ax, training_fig = line_plot(epochs, train_losses, ylabel='train_loss', 
                                 graphtitle=f'Training History_{features_names_suffix}', 
                                 linecolor='red', show=False)
    _, _ = line_plot(epochs, val_losses, ylabel='test_loss', ax=ax, show=True)
    
    # Save training plot
    training_fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(training_fig)

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
                        graphtitle=f'TimesFM Fine-tuned: True vs Predicted {features_names_suffix}', 
                        linecolor='blue', show=False)
    # Plot true values in red
    _, _ = line_plot(time_index, y_true, ylabel='true_vol', linecolor='red', ax=ax, show=True)
    
    # Save plot
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def TimesFM_finetuned_experiment(expiry, features_names, features_names_suffix):
    """
    Run TimesFM fine-tuned experiment
    
    Args:
        expiry: Prediction horizon
        features_names: Feature column names
        features_names_suffix: Suffix for naming results
        
    Returns:
        Dictionary with evaluation metrics
    """
    window = expiry * 2
    batch_size = 64  # Smaller batch size for transformer
    num_epochs = 30
    
    print(f"\nRunning TimesFM Fine-tuned experiment for {pred_value_to_char(expiry)} expiry")
    print(f"Features: {features_names}")
    print(f"Window size: {window}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    
    # Preprocess data
    X_raw, y_raw, y_scaler = preprocess_data_for_timesfm(
        df, features_names, f'{pred_value_to_char(expiry)}_vol', window, expiry
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader, X_train, X_val, X_test, y_train, y_val, y_test = create_dataloaders(
        X_raw, y_raw, batch_size
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create TimesFM model
    input_size = X_raw.shape[-1]
    model = create_timesfm_model(
        input_size=input_size,
        hidden_size=64,   # Smaller for stability
        num_layers=2,     # Fewer layers for efficiency
        num_heads=4,      # Fewer heads
        dropout=0.1,
        max_seq_length=1000
    ).to(device)
    
    print(f"Created TimesFM model with input_size={input_size}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_timesfm_model(
        model, train_loader, val_loader, optimizer, criterion, num_epochs, device
    )
    
    # Create and save training history plot using line_plot
    training_plot_path = os.path.join(training_plots_dir, f'training_history_{features_names_suffix}.png')
    plot_training_history_using_line_plot(train_losses, val_losses, features_names_suffix, training_plot_path)
    
    # Evaluate model
    metrics, predictions, y_test_inv = evaluate_timesfm_model(
        model, test_loader, y_scaler, device
    )
    
    # Create and save prediction plot using line_plot
    plot_path = os.path.join(plots_dir, f"{expiry}_vol_vs_true_test_TimesFM_finetuned_{features_names_suffix}.png")
    plot_predictions_using_line_plot(y_test_inv, predictions, expiry, features_names_suffix, plot_path)
    
    return metrics

def main():
    """Main function to run all fine-tuned experiments"""
    print("Starting TimesFM Fine-tuned Experiments")
    print("=" * 50)
    
    # Run experiments for different expiries
    for expiry in [5, 22, 66, 252]:
        names = ['log_returns', 'best_metric', 'best_5', 'best_10', 'best_20']
        best_features = find_n_best_features(expiry, 20)
        
        print(f"\n{'='*20} {pred_value_to_char(expiry)} Expiry {'='*20}")
        
        # 1. abs_log_returns
        metrics1 = TimesFM_finetuned_experiment(
            expiry=expiry, 
            features_names='al_lme_prices_abs_log_returns', 
            features_names_suffix=f'TimesFM_finetuned_{names[0]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # 2. best_metric
        metrics2 = TimesFM_finetuned_experiment(
            expiry=expiry, 
            features_names=best_features[0], 
            features_names_suffix=f'TimesFM_finetuned_{names[1]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # 3. best_5
        metrics3 = TimesFM_finetuned_experiment(
            expiry=expiry, 
            features_names=best_features[:5], 
            features_names_suffix=f'TimesFM_finetuned_{names[2]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # 4. best_10
        metrics4 = TimesFM_finetuned_experiment(
            expiry=expiry, 
            features_names=best_features[:10], 
            features_names_suffix=f'TimesFM_finetuned_{names[3]}_{pred_value_to_char(expiry)}_exp'
        )
        
        # 5. best_20
        metrics5 = TimesFM_finetuned_experiment(
            expiry=expiry, 
            features_names=best_features[:20], 
            features_names_suffix=f'TimesFM_finetuned_{names[4]}_{pred_value_to_char(expiry)}_exp'
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
        metrics_csv_path = os.path.join(metrics_dir, f'metrics_finetuned_{pred_value_to_char(expiry)}.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Fine-tuned metrics for {pred_value_to_char(expiry)} saved to: {metrics_csv_path}")
    
    print("\n" + "=" * 50)
    print("All TimesFM Fine-tuned experiments completed!")

if __name__ == "__main__":
    main()
