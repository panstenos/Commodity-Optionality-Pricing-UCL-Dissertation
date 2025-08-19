import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import sys
import os
import matplotlib.pyplot as plt

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

# Create LSTM_with_vol_experiment_results directory structure
main_results_dir = os.path.join(os.path.dirname(__file__), 'LSTM_with_vol_experiment_results')
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

def LSTM_with_vol_main_experiment(expiry, features_names, features_names_suffix):
    window = expiry*2
    batch_size = 128

    hidden_size = 32
    number_layers = 2
    output_size = 1

    loss_function = nn.MSELoss()
    n_epochs = 30

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


for expiry in [5, 22, 66, 252]:
    names = ['vol_only', 'best_metric', 'vol_plus_4', 'vol_plus_9', 'vol_plus_19']
    best_features = find_n_best_features(expiry, 20)

    # 1. vol_only: Use only the corresponding volatility column instead of abs_log_returns
    vol_column = f'{pred_value_to_char(expiry)}_vol'
    metrics1 = LSTM_with_vol_main_experiment(expiry=expiry, features_names=vol_column, features_names_suffix=f'LSTM_with_vol_{names[0]}_{pred_value_to_char(expiry)}_exp')

    # 2. best_metric: Same as before - use the single best feature
    metrics2 = LSTM_with_vol_main_experiment(expiry=expiry, features_names=best_features[0], features_names_suffix=f'LSTM_with_vol_{names[1]}_{pred_value_to_char(expiry)}_exp')

    # 3. vol_plus_4: Use volatility + 4 best predictors (total 5 features)
    vol_plus_4_features = [vol_column] + best_features[:4]
    metrics3 = LSTM_with_vol_main_experiment(expiry=expiry, features_names=vol_plus_4_features, features_names_suffix=f'LSTM_with_vol_{names[2]}_{pred_value_to_char(expiry)}_exp')
    
    # 4. vol_plus_9: Use volatility + 9 best predictors (total 10 features)
    vol_plus_9_features = [vol_column] + best_features[:9]
    metrics4 = LSTM_with_vol_main_experiment(expiry=expiry, features_names=vol_plus_9_features, features_names_suffix=f'LSTM_with_vol_{names[3]}_{pred_value_to_char(expiry)}_exp')

    # 5. vol_plus_19: Use volatility + 19 best predictors (total 20 features)
    vol_plus_19_features = [vol_column] + best_features[:19]
    metrics5 = LSTM_with_vol_main_experiment(expiry=expiry, features_names=vol_plus_19_features, features_names_suffix=f'LSTM_with_vol_{names[4]}_{pred_value_to_char(expiry)}_exp')

    # Combine all metrics for this expiry
    expiry_metrics = {
        'Feature_Selection': names,
        'MAPE': [metrics1['MAPE'], metrics2['MAPE'], metrics3['MAPE'], metrics4['MAPE'], metrics5['MAPE']],
        'MAE': [metrics1['MAE'], metrics2['MAE'], metrics3['MAE'], metrics4['MAE'], metrics5['MAE']],
        'RMSE': [metrics1['RMSE'], metrics2['RMSE'], metrics3['RMSE'], metrics4['RMSE'], metrics5['RMSE']],
        'MSE': [metrics1['MSE'], metrics2['MSE'], metrics3['MSE'], metrics4['MSE'], metrics5['MSE']],
        'MASE': [metrics1['MASE'], metrics2['MASE'], metrics3['MASE'], metrics4['MASE'], metrics5['MASE']]
    }

    # Save metrics to CSV in the metrics directory
    metrics_df = pd.DataFrame(expiry_metrics)
    metrics_csv_path = os.path.join(metrics_dir, f'metrics_{pred_value_to_char(expiry)}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics for {pred_value_to_char(expiry)} saved to: {metrics_csv_path}")
