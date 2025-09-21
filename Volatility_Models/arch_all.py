"""
GARCH(1,1) Volatility Forecasting for All Horizons

This module implements GARCH(1,1) model with context window c=2h to predict volatility
for all horizons (1w, 1m, 3m, 1y) on the entire series.
"""

import numpy as np
from arch import arch_model
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import os
import warnings
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath('..'))
from functions import line_plot, mse, mae, rmse, mape, mase, pred_char_to_value, load_data, pred_value_to_char

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration variables
horizon_char = ['1w', '1m', '3m', '1y']
horizon_vals = [5, 22, 66, 252]

def garch_forecast_entire_series(df, horizon_pred, context_window=None):
    """
    Forecast volatility for the entire series using GARCH(1,1) with context window c=2h.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with price data
    horizon_pred : int
        Prediction horizon in trading days
    context_window : int, optional
        Context window size. If None, uses 2*horizon_pred
    
    Returns:
    --------
    vols_pred : numpy.array
        Array of predicted volatilities
    """
    if context_window is None:
        context_window = 2 * horizon_pred
    
    col_name = pred_value_to_char(horizon_pred)
    print(f"Forecasting {col_name} volatility with context window {context_window}")
    
    vols_pred = []
    
    # Walk through the entire series with rolling window
    for idx in tqdm(range(context_window, df.shape[0]), desc=f"GARCH(1,1) forecasting {col_name}"):
        # Get returns window for GARCH model
        returns_window = df['al_lme_prices_log_returns'][idx-context_window:idx] * 100
        
        # Fit GARCH(1,1) model
        model = arch_model(returns_window, vol='Garch', p=1, q=1, dist='normal', 
                          mean='constant', rescale=False)
        res = model.fit(disp='off')
        
        # Forecast volatility at the horizon
        forecast = res.forecast(horizon=horizon_pred)
        
        # Convert to annualized volatility
        next_volatility = np.sqrt(252) * np.sqrt(forecast.variance.values[0][-1]) / 100
        vols_pred.append(next_volatility)
    
    return np.array(vols_pred)

def save_predictions_to_json(predictions_dict, filename="arch_all_predictions.json"):
    """
    Save predictions dictionary to JSON file.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary with horizon keys and prediction arrays as values
    filename : str
        Output JSON filename
    """
    # Convert numpy arrays to lists for JSON serialization
    json_dict = {}
    for horizon, preds in predictions_dict.items():
        json_dict[horizon] = preds.tolist()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    print(f"Predictions saved to: {filename}")

def plot_true_vs_predicted(df, predictions_dict, save_dir="arch_all_results/plots"):
    """
    Create plots comparing true vs predicted volatility for each horizon.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with true volatility data
    predictions_dict : dict
        Dictionary with horizon keys and prediction arrays as values
    save_dir : str
        Directory to save plots
    """
    # Create plots directory
    os.makedirs(save_dir, exist_ok=True)
    
    for col_name, vols_pred in predictions_dict.items():
        # Get the horizon value from the column name
        horizon_pred = pred_char_to_value(col_name)
        context_window = 2 * horizon_pred
        
        # Get true volatility (aligned with predictions)
        true_vol = df[f'{col_name}_vol'][context_window:]
        
        # Create date index for plotting
        dates = df.index[context_window:]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot true and predicted volatility
        ax.plot(dates, true_vol, label=f'True {col_name} Volatility', 
               color='blue', linewidth=1.5, alpha=0.8)
        ax.plot(dates, vols_pred, label=f'Predicted {col_name} Volatility', 
               color='red', linewidth=1.5, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.set_title(f'GARCH(1,1) Volatility Forecasting - {col_name} (h={horizon_pred})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Save plot
        plot_filename = os.path.join(save_dir, f'garch11_true_vs_pred_{col_name}.png')
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_filename}")
        
        # Calculate and print metrics
        mape_val = mape(true_vol, vols_pred)
        mae_val = mae(true_vol, vols_pred)
        rmse_val = rmse(true_vol, vols_pred)
        mase_val = mase(true_vol, vols_pred)
        
        print(f"{col_name} Metrics:")
        print(f"  MAPE: {mape_val:.4f}")
        print(f"  MAE: {mae_val:.4f}")
        print(f"  RMSE: {rmse_val:.4f}")
        print(f"  MASE: {mase_val:.4f}")
        print()

def run_arch_all_analysis(data_path='../Data/aluminium_pre_inputs.csv'):
    """
    Run complete GARCH(1,1) analysis for all horizons.
    
    Parameters:
    -----------
    data_path : str
        Path to the input data file
    """
    print("=== GARCH(1,1) Volatility Forecasting Analysis ===")
    print("Context window: c = 2h (twice the prediction horizon)")
    print("Model: GARCH(1,1)")
    print()
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} features")
    print()
    
    # Create results directory
    results_dir = "arch_all_results"
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Store all predictions
    all_predictions = {}
    
    # Forecast for each horizon
    for horizon_pred in horizon_vals:
        col_name = pred_value_to_char(horizon_pred)
        print(f"=== Forecasting {col_name} volatility (h={horizon_pred}) ===")
        
        # Get predictions
        vols_pred = garch_forecast_entire_series(df, horizon_pred)
        all_predictions[col_name] = vols_pred
        
        print(f"Generated {len(vols_pred)} predictions for {col_name}")
        print(f"Prediction range: {vols_pred.min():.4f} - {vols_pred.max():.4f}")
        print()
    
    # Save all predictions to JSON
    predictions_file = os.path.join(results_dir, "garch11_predictions.json")
    save_predictions_to_json(all_predictions, predictions_file)
    
    # Create plots for each horizon
    print("=== Creating True vs Predicted Plots ===")
    plot_true_vs_predicted(df, all_predictions, plots_dir)
    
    # Calculate and save summary metrics
    print("=== Summary Metrics ===")
    metrics_summary = []
    
    for col_name, vols_pred in all_predictions.items():
        horizon_pred = pred_char_to_value(col_name)
        context_window = 2 * horizon_pred
        true_vol = df[f'{col_name}_vol'][context_window:]
        
        metrics_summary.append({
            'Horizon': col_name,
            'Horizon_Days': horizon_pred,
            'Context_Window': context_window,
            'MAPE': mape(true_vol, vols_pred),
            'MAE': mae(true_vol, vols_pred),
            'RMSE': rmse(true_vol, vols_pred),
            'MASE': mase(true_vol, vols_pred),
            'N_Predictions': len(vols_pred)
        })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_file = os.path.join(results_dir, "garch11_metrics_summary.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics summary saved to: {metrics_file}")
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved in: {results_dir}/")
    print("- Predictions: garch11_predictions.json")
    print("- Plots: plots/ directory")
    print("- Metrics: garch11_metrics_summary.csv")

if __name__ == "__main__":
    # Run the complete analysis
    run_arch_all_analysis()
