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
import numpy as np
from NN_functions import preprocess_dataframe, get_dataloaders, create_model, train_model, plot_train_test_predictions, evaluate_and_print_metrics

# Load the data using the load_data function with proper path
data_path = os.path.join(parent_dir, 'Data', 'aluminium_pre_inputs.csv')
df = load_data(data_path)

# Create walk_forward_results directory structure
main_results_dir = os.path.join(os.path.dirname(__file__), 'walk_forward_results')
plots_dir = os.path.join(main_results_dir, 'plots')
training_plots_dir = os.path.join(main_results_dir, 'training_plots')
metrics_dir = os.path.join(main_results_dir, 'metrics')
detailed_metrics_dir = os.path.join(main_results_dir, 'detailed_metrics')

# Create directories if they don't exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(training_plots_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(detailed_metrics_dir, exist_ok=True)

def find_n_best_features(expiry, n):
    """Find the n best features for a given expiry period"""
    corr_file = os.path.join(parent_dir, 'Feature_selection', 'absolute_feature_correlations.csv')
    best_features = load_data(corr_file, index_col=0)
    top_rows = best_features.sort_values(by=f'{pred_value_to_char(expiry)}_exp', ascending=False).head(n)
    best_features = top_rows.index.tolist()
    return best_features

def walk_forward_experiment(expiry, features_names, features_names_suffix, 
                           train_size=1000, test_size=200, step_size=100):
    """
    Perform walk-forward optimization experiment
    
    Parameters:
    - expiry: Expiry period (5, 22, 66, 252)
    - features_names: Features to use for prediction
    - features_names_suffix: Suffix for naming results
    - train_size: Size of training window
    - test_size: Size of test window
    - step_size: How much to advance the window each iteration
    """
    window = expiry * 2
    batch_size = 128
    
    hidden_size = 32
    number_layers = 2
    output_size = 1
    
    loss_function = nn.MSELoss()
    n_epochs = 30
    
    # Prepare data
    X_raw, y_raw, input_size = preprocess_dataframe(df[features_names], df[f'{pred_value_to_char(expiry)}_vol'], window, expiry)
    
    # Calculate number of iterations
    total_samples = len(X_raw)
    n_iterations = (total_samples - train_size - test_size) // step_size + 1
    
    print(f"Walk-forward experiment for {pred_value_to_char(expiry)} expiry")
    print(f"Total samples: {total_samples}, Training window: {train_size}, Test window: {test_size}")
    print(f"Step size: {step_size}, Number of iterations: {n_iterations}")
    print("=" * 60)
    
    # Store results for each iteration
    iteration_results = []
    all_predictions = []
    all_true_values = []
    
    for iteration in range(n_iterations):
        # Calculate window boundaries
        train_start = iteration * step_size
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        
        # Ensure we don't go beyond data bounds
        if test_end > total_samples:
            break
            
        print(f"Iteration {iteration + 1}/{n_iterations}: Training on samples {train_start}-{train_end}, Testing on {test_start}-{test_end}")
        
        # Extract data for this iteration
        X_train_iter = X_raw[train_start:train_end]
        y_train_iter = y_raw[train_start:train_end]
        X_test_iter = X_raw[test_start:test_end]
        y_test_iter = y_raw[test_start:test_end]
        
        # Create dataloaders for this iteration
        trainloader, testloader, y_scaler, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = get_dataloaders(
            X_train_iter, y_train_iter, window, expiry, batch_size=batch_size
        )
        
        # Create and train model
        model = create_model(input_size, hidden_size, number_layers, output_size, activation_fn=nn.ReLU())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_losses, val_losses = train_model(model, optimizer, loss_function, trainloader, testloader, n_epochs)
        
        # Save training history plot
        ax, training_fig = line_plot(train_losses, train_losses, ylabel='train_loss', 
                                   graphtitle=f'Training History_{features_names_suffix}_iter{iteration+1}', 
                                   linecolor='red', show=False)
        _, _ = line_plot(val_losses, val_losses, ylabel='test_loss', ax=ax, show=True)
        
        training_plot_path = os.path.join(training_plots_dir, f'training_history_{features_names_suffix}_iter{iteration+1}.png')
        training_fig.savefig(training_plot_path, dpi=300, bbox_inches='tight')
        plt.close(training_fig)
        
        # Evaluate model on test set
        metrics = evaluate_and_print_metrics(model, X_test_tensor, y_test_tensor, y_scaler, window, device)
        
        # Store results
        iteration_results.append({
            'iteration': iteration + 1,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_samples': train_size,
            'test_samples': test_size,
            'MAPE': metrics['MAPE'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MSE': metrics['MSE'],
            'MASE': metrics['MASE']
        })
        
        # Get predictions for plotting
        model.eval()
        with torch.no_grad():
            test_predictions_scaled = model(X_test_tensor.to(device)).detach().cpu().numpy().flatten()
        
        # Inverse transform predictions
        dummies_pred = np.zeros((X_test_tensor.shape[0], window + 1))
        dummies_pred[:, 0] = test_predictions_scaled
        test_predictions = y_scaler.inverse_transform(dummies_pred)[:, 0]
        
        # Inverse transform true values
        dummies_true = np.zeros((X_test_tensor.shape[0], window + 1))
        dummies_true[:, 0] = y_test_tensor.detach().cpu().numpy().flatten()
        test_true_vals = y_scaler.inverse_transform(dummies_true)[:, 0]
        
        all_predictions.extend(test_predictions)
        all_true_values.extend(test_true_vals)
        
        print(f"Iteration {iteration + 1} Results - MAPE: {metrics['MAPE']:.2f}, MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")
        print("-" * 40)
    
    # Create summary results
    summary_results = {
        'Feature_Selection': features_names_suffix,
        'Expiry': pred_value_to_char(expiry),
        'Total_Iterations': len(iteration_results),
        'Train_Window_Size': train_size,
        'Test_Window_Size': test_size,
        'Step_Size': step_size,
        'Avg_MAPE': np.mean([r['MAPE'] for r in iteration_results]),
        'Std_MAPE': np.std([r['MAPE'] for r in iteration_results]),
        'Avg_MAE': np.mean([r['MAE'] for r in iteration_results]),
        'Std_MAE': np.std([r['MAE'] for r in iteration_results]),
        'Avg_RMSE': np.mean([r['RMSE'] for r in iteration_results]),
        'Std_RMSE': np.std([r['RMSE'] for r in iteration_results]),
        'Avg_MSE': np.mean([r['MSE'] for r in iteration_results]),
        'Std_MSE': np.std([r['MSE'] for r in iteration_results]),
        'Avg_MASE': np.mean([r['MASE'] for r in iteration_results]),
        'Std_MASE': np.std([r['MASE'] for r in iteration_results])
    }
    
    # Save detailed results
    detailed_df = pd.DataFrame(iteration_results)
    detailed_path = os.path.join(detailed_metrics_dir, f'detailed_metrics_{features_names_suffix}.csv')
    detailed_df.to_csv(detailed_path, index=False)
    
    # Save summary results
    summary_df = pd.DataFrame([summary_results])
    summary_path = os.path.join(metrics_dir, f'summary_metrics_{features_names_suffix}.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Create and save walk-forward performance plot
    plt.figure(figsize=(12, 8))
    
    # Plot MAPE over iterations
    plt.subplot(2, 2, 1)
    iterations = [r['iteration'] for r in iteration_results]
    mapes = [r['MAPE'] for r in iteration_results]
    plt.plot(iterations, mapes, 'b-o', linewidth=2, markersize=6)
    plt.axhline(y=np.mean(mapes), color='r', linestyle='--', label=f'Mean: {np.mean(mapes):.2f}')
    plt.fill_between(iterations, 
                     [np.mean(mapes) - np.std(mapes)] * len(iterations),
                     [np.mean(mapes) + np.std(mapes)] * len(iterations),
                     alpha=0.3, color='r', label=f'±1 Std: {np.std(mapes):.2f}')
    plt.xlabel('Iteration')
    plt.ylabel('MAPE')
    plt.title(f'MAPE Performance Over Iterations - {features_names_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot RMSE over iterations
    plt.subplot(2, 2, 2)
    rmses = [r['RMSE'] for r in iteration_results]
    plt.plot(iterations, rmses, 'g-o', linewidth=2, markersize=6)
    plt.axhline(y=np.mean(rmses), color='r', linestyle='--', label=f'Mean: {np.mean(rmses):.2f}')
    plt.fill_between(iterations, 
                     [np.mean(rmses) - np.std(rmses)] * len(iterations),
                     [np.mean(rmses) + np.std(rmses)] * len(iterations),
                     alpha=0.3, color='r', label=f'±1 Std: {np.std(rmses):.2f}')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Performance Over Iterations - {features_names_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot predictions vs true values (last iteration)
    plt.subplot(2, 2, 3)
    plt.scatter(all_true_values, all_predictions, alpha=0.6, s=20)
    plt.plot([min(all_true_values), max(all_true_values)], 
             [min(all_true_values), max(all_true_values)], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values - {features_names_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot performance distribution
    plt.subplot(2, 2, 4)
    plt.hist(mapes, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(mapes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mapes):.2f}')
    plt.xlabel('MAPE')
    plt.ylabel('Frequency')
    plt.title(f'MAPE Distribution - {features_names_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'walk_forward_performance_{features_names_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nWalk-forward experiment completed for {features_names_suffix}")
    print(f"Average MAPE: {np.mean(mapes):.2f} ± {np.std(mapes):.2f}")
    print(f"Average RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    print(f"Results saved to: {main_results_dir}")
    
    return summary_results, iteration_results

# Run walk-forward experiments for different feature selections
def run_all_walk_forward_experiments():
    """Run walk-forward experiments for all expiry periods and feature selections"""
    
    # Define experiment parameters
    train_size = 1000  # Training window size
    test_size = 200    # Test window size
    step_size = 100    # Step size for advancing windows
    
    all_summary_results = []
    
    for expiry in [5, 22, 66, 252]:
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD EXPERIMENTS FOR {pred_value_to_char(expiry)} EXPIRY")
        print(f"{'='*80}")
        
        # Get best features for this expiry
        best_features = find_n_best_features(expiry, 20)
        
        # Define feature selections to test
        feature_configs = [
            ('vol_only', f'{pred_value_to_char(expiry)}_vol'),
            ('best_metric', best_features[0]),
            ('best_5', best_features[:5]),
            ('best_10', best_features[:10]),
            ('best_20', best_features[:20])
        ]
        
        for config_name, features in feature_configs:
            print(f"\nRunning {config_name} configuration...")
            
            try:
                summary, detailed = walk_forward_experiment(
                    expiry=expiry,
                    features_names=features,
                    features_names_suffix=f'LSTM_wf_{config_name}_{pred_value_to_char(expiry)}_exp',
                    train_size=train_size,
                    test_size=test_size,
                    step_size=step_size
                )
                
                all_summary_results.append(summary)
                
            except Exception as e:
                print(f"Error in {config_name} configuration: {e}")
                continue
    
    # Save combined summary results
    if all_summary_results:
        combined_summary = pd.DataFrame(all_summary_results)
        combined_path = os.path.join(metrics_dir, 'combined_walk_forward_summary.csv')
        combined_summary.to_csv(combined_path, index=False)
        print(f"\nCombined summary saved to: {combined_path}")
    
    print("\nAll walk-forward experiments completed!")
    return all_summary_results

if __name__ == "__main__":
    # Run all experiments
    results = run_all_walk_forward_experiments()