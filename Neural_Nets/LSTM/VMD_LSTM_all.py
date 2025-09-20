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
from NN_functions import preprocess_dataframe, get_dataloaders, create_model, train_model, plot_train_test_predictions, evaluate_and_print_metrics, train_one_epoch, validate_one_epoch

# Load the data using the load_data function with proper path
data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
df = load_data(data_path)

# Create main_experiment_results directory structure
main_results_dir = os.path.join(os.path.dirname(__file__), 'VMD_LSTM_all_results')
plots_dir = os.path.join(main_results_dir, 'plots')
training_plots_dir = os.path.join(main_results_dir, 'training_plots')
metrics_dir = os.path.join(main_results_dir, 'metrics')
vmd_modes_dir = os.path.join(main_results_dir, 'vmd_modes_plots')

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(training_plots_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(vmd_modes_dir, exist_ok=True)

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_n_best_features(expiry, n):
    corr_file = os.path.join(parent_dir, 'feature_selection', 'absolute_feature_correlations.csv')
    best_features = load_data(corr_file, index_col=0)
    top_rows = best_features.sort_values(by=f'{pred_value_to_char(expiry)}_exp', ascending=False).head(n)
    best_features = top_rows.index.tolist()
    return best_features

def plot_vmd_modes(model, X_sample, y_sample, expiry, features_names_suffix, save_path):
    """
    Plot the decomposed VMD modes for a sample input
    """
    model.eval()
    with torch.no_grad():
        # Get a sample batch
        if isinstance(X_sample, np.ndarray):
            X_sample = torch.tensor(X_sample).float()
        
        # Ensure we have a batch dimension
        if X_sample.ndim == 2:
            X_sample = X_sample.unsqueeze(0)
        
        # Move to device
        X_sample = X_sample.to(device)
        
        # Get the decomposed modes from the model
        if hasattr(model, 'vmd_decompose'):
            imf_modes = model.vmd_decompose(X_sample)
            
            # Convert to numpy for plotting
            if isinstance(imf_modes, torch.Tensor):
                imf_modes = imf_modes.cpu().numpy()
            
            # Plot the original signal and decomposed modes
            fig, axes = plt.subplots(model.num_modes + 1, 1, figsize=(15, 3 * (model.num_modes + 1)))
            fig.suptitle(f'VMD Decomposition - {features_names_suffix}', fontsize=16)
            
            # Plot original signal
            original_signal = X_sample.cpu().numpy()[0, :, 0]  # First batch, all time steps, first feature
            axes[0].plot(original_signal, 'b-', linewidth=2, label='Original Signal')
            axes[0].set_title('Original Signal')
            axes[0].set_ylabel('Amplitude')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot each decomposed mode
            mode_names = ['Trend Mode', 'Seasonal Mode', 'Residual Mode']
            colors = ['r-', 'g-', 'm-']
            
            for i in range(model.num_modes):
                mode_signal = imf_modes[i, 0, :, 0]  # First batch, all time steps, first feature
                axes[i+1].plot(mode_signal, colors[i], linewidth=2, label=f'{mode_names[i]} (IMF {i+1})')
                axes[i+1].set_title(f'{mode_names[i]} (IMF {i+1})')
                axes[i+1].set_ylabel('Amplitude')
                axes[i+1].legend()
                axes[i+1].grid(True, alpha=0.3)
            
            # Add x-label to the last subplot
            axes[-1].set_xlabel('Time Steps')
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"VMD modes plot saved to: {save_path}")
            
            return imf_modes
        else:
            print("Model does not have vmd_decompose method")
            return None

def VMD_LSTM_all_main_experiment(expiry, features_names, features_names_suffix, seed=42):
    # Set seed for reproducibility
    set_seed(seed)
    
    window = 5  # Fixed context window of 5
    batch_size = 128
    hidden_size = 32
    number_layers = 2
    output_size = 1
    loss_function = nn.MSELoss()
    n_epochs = 20

    X_raw, y_raw, input_size = preprocess_dataframe(df[features_names], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)
    trainloader, testloader, y_scaler, X_train, X_test, y_train, y_test = get_dataloaders(X_raw, y_raw, window, expiry, batch_size)
    
    # Use VMD_LSTM model instead of regular LSTM
    model = create_model(input_size, hidden_size, number_layers, output_size, 
                        activation_fn=nn.ReLU(), model_type='VMD_LSTM')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = train_model(model, optimizer, loss_function, trainloader, testloader, n_epochs)

    # Create and save training history plot using line_plot
    ax, training_fig = line_plot(train_losses, train_losses, ylabel='train_loss', graphtitle=f'Training History_{features_names_suffix}', linecolor='red', show=False)
    _, _ = line_plot(val_losses, val_losses, ylabel='test_loss', ax=ax, show=True)
    # Save training plot
    training_plot_path = os.path.join(training_plots_dir, f'training_history_{features_names_suffix}.png')
    training_fig.savefig(training_plot_path, dpi=300, bbox_inches='tight')
    plt.close(training_fig)

    # Plot VMD decomposed modes for a sample from training data
    if len(X_train) > 0:
        sample_idx = 0  # Use first sample
        X_sample = X_train[sample_idx:sample_idx+1]  # Keep batch dimension
        vmd_modes_plot_path = os.path.join(vmd_modes_dir, f'vmd_modes_{features_names_suffix}.png')
        plot_vmd_modes(model, X_sample, y_train[sample_idx:sample_idx+1], expiry, features_names_suffix, vmd_modes_plot_path)

    # Plot the train and test predictions
    fig1, fig2 = plot_train_test_predictions(model, X_train, y_train, X_test, y_test, y_scaler, window, device, f'{pred_value_to_char(expiry)}_vol', features_names_suffix)
    
    # Save plots in the plots directory
    fig1.savefig(os.path.join(plots_dir, f"{expiry}_vol_vs_true_train_{features_names_suffix}.png"))
    fig2.savefig(os.path.join(plots_dir, f"{expiry}_vol_vs_true_test_{features_names_suffix}.png"))

    # Evaluate the model
    metrics = evaluate_and_print_metrics(model, X_test, y_test, y_scaler, window, device)

    return metrics


# Main experiment loop - same structure as LSTM_all.py
for expiry in [5, 22, 66, 252]:
    best_features = find_n_best_features(expiry, 20)
    vol_column = f'{pred_value_to_char(expiry)}_vol'
    
    # Get the name of the best metric for this expiry
    best_metric_name = best_features[0]
    
    # Define feature combinations and their names - same as LSTM_all.py
    feature_combinations = [
        ('al_lme_prices_abs_log_returns', 'abs_log_returns'),
        (vol_column, 'vol_only'),
        (best_metric_name, best_metric_name),  # Use actual name instead of 'best_metric'
        ([best_metric_name, vol_column], 'best_vol'),
        (best_features[:5], 'best5'),
        (best_features[:4] + [vol_column], 'best4_vol'),
        (best_features[:10], 'best10'),
        (best_features[:9] + [vol_column], 'best9_vol'),
        (best_features[:20], 'best20'),
        (best_features[:19] + [vol_column], 'best19_vol')
    ]
    
    # Store all metrics for this expiry across all runs
    all_runs_metrics = []
    feature_names = []
    
    # Define seeds for 3 different runs
    seeds = [42, 123, 456]
    
    # Run experiments for each feature combination
    for i, (features, name) in enumerate(feature_combinations):
        print(f"Running experiment {i+1}/10 for expiry {expiry}: {name}")
        
        # Store metrics for this configuration across 3 runs
        config_metrics = []
        
        # Run the same configuration 3 times with different seeds
        for run_idx, seed in enumerate(seeds):
            print(f"  Run {run_idx+1}/3 with seed {seed}")
            metrics = VMD_LSTM_all_main_experiment(
                expiry=expiry, 
                features_names=features, 
                features_names_suffix=f'VMD_LSTM_all_{name}_{pred_value_to_char(expiry)}_exp_run{run_idx+1}_seed{seed}',
                seed=seed
            )
            config_metrics.append(metrics)
        
        all_runs_metrics.append(config_metrics)
        feature_names.append(name)

    # Calculate mean and std for each configuration
    mean_metrics = {
        'Feature_Selection': feature_names,
        'MAPE': [], 'MAE': [], 'RMSE': [], 'MSE': [], 'MASE': []
    }
    std_metrics = {
        'Feature_Selection': feature_names,
        'MAPE': [], 'MAE': [], 'RMSE': [], 'MSE': [], 'MASE': []
    }
    
    for config_runs in all_runs_metrics:
        mape_values = [run['MAPE'] for run in config_runs]
        mae_values = [run['MAE'] for run in config_runs]
        rmse_values = [run['RMSE'] for run in config_runs]
        mse_values = [run['MSE'] for run in config_runs]
        mase_values = [run['MASE'] for run in config_runs]
        
        mean_metrics['MAPE'].append(np.mean(mape_values))
        mean_metrics['MAE'].append(np.mean(mae_values))
        mean_metrics['RMSE'].append(np.mean(rmse_values))
        mean_metrics['MSE'].append(np.mean(mse_values))
        mean_metrics['MASE'].append(np.mean(mase_values))
        
        std_metrics['MAPE'].append(np.std(mape_values, ddof=1))
        std_metrics['MAE'].append(np.std(mae_values, ddof=1))
        std_metrics['RMSE'].append(np.std(rmse_values, ddof=1))
        std_metrics['MSE'].append(np.std(mse_values, ddof=1))
        std_metrics['MASE'].append(np.std(mase_values, ddof=1))

    mean_metrics_df = pd.DataFrame(mean_metrics)
    mean_metrics_csv_path = os.path.join(metrics_dir, f'metrics_mean_{pred_value_to_char(expiry)}.csv')
    mean_metrics_df.to_csv(mean_metrics_csv_path, index=False)
    print(f"Mean metrics for {pred_value_to_char(expiry)} saved to: {mean_metrics_csv_path}")
    
    std_metrics_df = pd.DataFrame(std_metrics)
    std_metrics_csv_path = os.path.join(metrics_dir, f'metrics_std_{pred_value_to_char(expiry)}.csv')
    std_metrics_df.to_csv(std_metrics_csv_path, index=False)
    print(f"Standard deviation metrics for {pred_value_to_char(expiry)} saved to: {std_metrics_csv_path}")

print("\n=== VMD_LSTM_all Experiment Complete ===")
print("Results saved in VMD_LSTM_all_results/ directory")
print("Each CSV contains:")
print("- Mean and standard deviation of performance metrics")
print("- 10 feature combinations tested")
print("- 3 runs per configuration with different seeds")
print("- Fixed context window of 5")
print("- VMD-LSTM architecture with decomposition")
