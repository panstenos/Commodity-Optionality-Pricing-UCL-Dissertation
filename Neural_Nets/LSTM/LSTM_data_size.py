import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directories to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to root
neural_nets_dir = os.path.dirname(current_dir)  # Go up one level to Neural_Nets

# Add both directories to Python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if neural_nets_dir not in sys.path:
    sys.path.insert(0, neural_nets_dir)

from functions import pred_value_to_char, load_data, line_plot
import pandas as pd
from NN_functions import preprocess_dataframe, get_dataloaders, create_model, train_model, plot_train_test_predictions, evaluate_and_print_metrics, train_one_epoch, validate_one_epoch, TimeSeriesDataset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

# Load the data using the load_data function with proper path
data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
df = load_data(data_path)

# Create data_size_experiment_results directory structure
main_results_dir = os.path.join(os.path.dirname(__file__), 'data_size_experiment_results')
plots_dir = os.path.join(main_results_dir, 'plots')
training_plots_dir = os.path.join(main_results_dir, 'training_plots')
metrics_dir = os.path.join(main_results_dir, 'metrics')

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(training_plots_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

def find_n_best_features(expiry, n):
    corr_file = os.path.join(parent_dir, 'Feature_selection', 'absolute_feature_correlations.csv')
    best_features = load_data(corr_file, index_col=0)
    top_rows = best_features.sort_values(by=f'{pred_value_to_char(expiry)}_exp', ascending=False).head(n)
    best_features = top_rows.index.tolist()
    return best_features

def LSTM_data_size_experiment(expiry, features_names, features_names_suffix, train_size, test_size=200):
    """
    Run LSTM experiment with specified training data size
    
    Parameters:
    - expiry: expiry period (22 for 1 month)
    - features_names: features to use for training
    - features_names_suffix: suffix for naming results
    - train_size: number of rows to use for training (2000, 1000, 500, 200, 100)
    - test_size: number of rows to use for testing (default 200)
    """
    window = expiry * 2
    batch_size = 128

    hidden_size = 32
    number_layers = 2
    output_size = 1

    loss_function = nn.MSELoss()
    n_epochs = 30

    # Preprocess the full dataset first
    X_raw, y_raw, input_size = preprocess_dataframe(df[features_names], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)
    
    # Check if we have enough data
    total_available = len(X_raw)
    if train_size + test_size > total_available:
        # Adjust train_size to fit within available data
        train_size = max(100, total_available - test_size)
    
    # Split data based on train_size and test_size
    # Use first train_size rows for training, last test_size rows for testing
    X_train = X_raw[:train_size]
    y_train = y_raw[:train_size]
    
    # For testing, use the last test_size rows
    X_test = X_raw[-test_size:]
    y_test = y_raw[-test_size:]
    
    # Create custom dataloaders with the specified split
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train_scaled).float()
    X_test_tensor = torch.tensor(X_test).float()
    y_test_tensor = torch.tensor(y_test_scaled).float()

    
    train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
    test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and train model (custom training to avoid printing)
    model = create_model(input_size, hidden_size, number_layers, output_size, activation_fn=nn.ReLU())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, loss_function, trainloader, False)
        val_loss = validate_one_epoch(model, loss_function, testloader, False)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Create and save training history plot using line_plot
    ax, training_fig = line_plot(train_losses, train_losses, ylabel='train_loss', graphtitle=f'Training History_{features_names_suffix}_train{train_size}', linecolor='red', show=False)
    _, _ = line_plot(val_losses, val_losses, ylabel='test_loss', ax=ax, show=False)
    # Save training plot
    training_plot_path = os.path.join(training_plots_dir, f'training_history_{features_names_suffix}_train{train_size}.png')
    training_fig.savefig(training_plot_path, dpi=300, bbox_inches='tight')
    plt.close(training_fig)

    # Plot the train and test predictions
    fig1, fig2 = plot_train_test_predictions(model, X_train_tensor, y_train, X_test_tensor, y_test, y_scaler, window, device, f'{pred_value_to_char(expiry)}_vol', f'{features_names_suffix}_train{train_size}')
    
    # Save plots in the plots directory
    fig1.savefig(os.path.join(plots_dir, f"{expiry}_vol_vs_true_train_{features_names_suffix}_train{train_size}.png"))
    fig2.savefig(os.path.join(plots_dir, f"{expiry}_vol_vs_true_test_{features_names_suffix}_train{train_size}.png"))

    # Evaluate the model (custom function to avoid printing)
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test_tensor.to(device)).detach().cpu().numpy().flatten()

    dummies_pred = np.zeros((X_test_tensor.shape[0], window + 1))
    dummies_pred[:, 0] = test_predictions_scaled
    test_predictions = y_scaler.inverse_transform(dummies_pred)[:, 0]

    dummies_true = np.zeros((X_test_tensor.shape[0], window + 1))
    dummies_true[:, 0] = y_test.detach().cpu().numpy().flatten()
    test_true_vals = y_scaler.inverse_transform(dummies_true)[:, 0]

    from functions import mape, mae, rmse, mse, mase
    mape_val = mape(test_true_vals, test_predictions)
    mae_val = mae(test_true_vals, test_predictions)
    rmse_val = rmse(test_true_vals, test_predictions)
    mse_val = mse(test_true_vals, test_predictions)
    mase_val = mase(test_true_vals, test_predictions)

    metrics = {
        "MAPE": mape_val,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "MSE": mse_val,
        "MASE": mase_val
    }
    
    # Add training size information to metrics
    metrics['train_size'] = train_size
    metrics['test_size'] = test_size

    return metrics

def main():
    # Only run for 1-month expiry (22 days)
    expiry = 22
    
    # Define training data sizes to test
    train_sizes = [2000, 1000, 500, 200, 100]
    
    # Get best features for this expiry
    best_features = find_n_best_features(expiry, 20)
    
    # Define feature sets to test
    feature_sets = {
        'log_returns': 'al_lme_prices_abs_log_returns',
        'best_metric': best_features[0],
        'best_5': best_features[:5],
        'best_10': best_features[:10],
        'best_20': best_features[:20]
    }
    
    # Store all results
    all_results = []
    
    # Run experiments for each training size
    for train_size in train_sizes:
        # Store results for this training size
        train_size_results = []
        
        for feature_name, features in feature_sets.items():
            try:
                metrics = LSTM_data_size_experiment(
                    expiry=expiry,
                    features_names=features,
                    features_names_suffix=f'LSTM_{feature_name}_{pred_value_to_char(expiry)}_exp',
                    train_size=train_size
                )
                
                # Add feature set information
                metrics['feature_set'] = feature_name
                train_size_results.append(metrics)
                all_results.append(metrics)
                
            except Exception as e:
                continue
        
        # Save metrics for this training size to a separate CSV file
        if train_size_results:
            train_size_df = pd.DataFrame(train_size_results)
            # Reorder columns for better readability
            column_order = ['feature_set', 'train_size', 'test_size', 'MAPE', 'MAE', 'RMSE', 'MSE', 'MASE']
            available_columns = [col for col in column_order if col in train_size_df.columns]
            train_size_df = train_size_df[available_columns]
            
            # Save to CSV for this specific training size
            train_size_csv_path = os.path.join(metrics_dir, f'train_size_{train_size}_metrics_{pred_value_to_char(expiry)}.csv')
            train_size_df.to_csv(train_size_csv_path, index=False)
    
    # Create summary DataFrame and save overall results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = ['feature_set', 'train_size', 'test_size', 'MAPE', 'MAE', 'RMSE', 'MSE', 'MASE']
        available_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[available_columns]
        
        # Save overall results to CSV
        metrics_csv_path = os.path.join(metrics_dir, f'data_size_experiment_metrics_{pred_value_to_char(expiry)}.csv')
        results_df.to_csv(metrics_csv_path, index=False)

if __name__ == "__main__":
    main()
