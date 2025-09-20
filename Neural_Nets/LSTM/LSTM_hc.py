import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import random

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
from NN_functions import preprocess_dataframe, get_dataloaders, create_model, train_model, plot_train_test_predictions, evaluate_and_print_metrics, train_one_epoch, validate_one_epoch

# Load the data using the load_data function with proper path
data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
df = load_data(data_path)

# Create LSTM_hc_results directory structure
main_results_dir = os.path.join(os.path.dirname(__file__), 'LSTM_hc_results')
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

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def LSTM_hc_experiment(expiry, features_names, features_names_suffix, context_window, seed=42):
    """
    Run LSTM experiment with specified context window length
    
    Parameters:
    - expiry: expiry period (5, 22, 66, 252)
    - features_names: features to use for training
    - features_names_suffix: suffix for naming results
    - context_window: window size to use instead of expiry*2
    - seed: random seed for reproducibility
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    window = context_window  # Use specified context window instead of expiry*2
    batch_size = 128

    hidden_size = 32
    number_layers = 2
    output_size = 1

    loss_function = nn.MSELoss()
    n_epochs = 20

    X_raw, y_raw, input_size = preprocess_dataframe(df[features_names], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)
    trainloader, testloader, y_scaler, X_train, X_test, y_train, y_test = get_dataloaders(X_raw, y_raw, window, expiry, batch_size)
    model = create_model(input_size, hidden_size, number_layers, output_size, activation_fn=nn.ReLU())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = train_model(model, optimizer, loss_function, trainloader, testloader, n_epochs)

    # Create and save training history plot using line_plot
    ax, training_fig = line_plot(train_losses, train_losses, ylabel='train_loss', graphtitle=f'Training History_{features_names_suffix}', linecolor='red', show=False)
    _, _ = line_plot(val_losses, val_losses, ylabel='test_loss', ax=ax, show=True)
    # Save training plot
    training_plot_path = os.path.join(training_plots_dir, f'training_history_{features_names_suffix}.png')
    training_fig.savefig(training_plot_path, dpi=300, bbox_inches='tight')
    plt.close(training_fig)

    # Plot the train and test predictions
    fig1, fig2 = plot_train_test_predictions(model, X_train, y_train, X_test, y_test, y_scaler, window, device, f'{pred_value_to_char(expiry)}_vol', features_names_suffix)
    
    # Save plots in the plots directory
    fig1.savefig(os.path.join(plots_dir, f"{expiry}_vol_vs_true_train_{features_names_suffix}.png"))
    fig2.savefig(os.path.join(plots_dir, f"{expiry}_vol_vs_true_test_{features_names_suffix}.png"))

    # Evaluate the model
    metrics = evaluate_and_print_metrics(model, X_test, y_test, y_scaler, window, device)

    return metrics


# Define context windows to test
context_windows = [5, 10, 20, 50, 100, 200, 500, 1000]

# Run experiments for each expiry period
for expiry in [5, 22, 66, 252]:
    print(f"\n=== Testing Context Windows for Expiry {expiry} ({pred_value_to_char(expiry)}) ===")
    
    # Get top 5 best features for this expiry
    best_features = find_n_best_features(expiry, 20)
    top_5_features = best_features[:5]
    print(f"Top 5 features for {pred_value_to_char(expiry)}: {top_5_features}")
    
    # Store results for this expiry
    results = []
    
    # Test each context window
    for context_window in context_windows:
        print(f"\nTesting context window: {context_window}")
        
        # Test using top 5 features together as a single model
        print(f"  Testing top 5 features together with window {context_window}")
        try:
            metrics = LSTM_hc_experiment(
                expiry=expiry,
                features_names=top_5_features,  # Pass all 5 features together
                features_names_suffix=f'LSTM_hc_top5_{pred_value_to_char(expiry)}_window{context_window}',
                context_window=context_window,
                seed=42
            )
            
            # Store results
            result_metrics = {
                'Context_Window': context_window,
                'MAPE': metrics['MAPE'],
                'MAE': metrics['MAE'], 
                'RMSE': metrics['RMSE'],
                'MSE': metrics['MSE'],
                'MASE': metrics['MASE']
            }
            
            results.append(result_metrics)
            print(f"    Top 5 features MAPE: {metrics['MAPE']:.4f}")
            print(f"    Top 5 features RMSE: {metrics['RMSE']:.4f}")
            
        except Exception as e:
            print(f"    Error with top 5 features window {context_window}: {e}")
            continue
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(metrics_dir, f'context_window_results_{pred_value_to_char(expiry)}.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nContext window results for {pred_value_to_char(expiry)} saved to: {results_csv_path}")
        
        # Find best context window based on MAPE
        best_window_idx = results_df['MAPE'].idxmin()
        best_window = results_df.loc[best_window_idx, 'Context_Window']
        best_mape = results_df.loc[best_window_idx, 'MAPE']
        print(f"Best context window for {pred_value_to_char(expiry)}: {best_window} (MAPE: {best_mape:.4f})")
    else:
        print(f"No successful experiments for expiry {expiry}")

print("\n=== Context Window Analysis Complete ===")
print("Results saved in LSTM_hc_results/metrics/ directory")
print("Each CSV contains:")
print("- Performance metrics for top 5 covariates used together as a single model")
print("- Context windows tested: [5, 10, 20, 50, 100, 200, 500, 1000]")
print("\nTop 5 covariates are selected from the best 20 features based on correlation with volatility for each expiry period")
print("All 5 features are used together as input to a single LSTM model (not separately)")