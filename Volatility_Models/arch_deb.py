"""
Autoregressive Conditional Heteroskedasticity (ARCH) Family Models

This module implements GARCH models for volatility forecasting.
- Periods of high volatility are followed by periods of even higher volatility
- Periods of low volatility are followed by periods of even lower volatility
- Volatility tends to cluster over time
- Error variance is thought to be autocorrelated over time
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
from functions import line_plot, mse, mae, rmse, mape, mase, pred_char_to_value

from functions import load_data, plot_squared_pacf
# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration variables
horizon_char = ['1w', '1m', '3m', '1y']
horizon_vals = [5, 22, 66, 252]

def plot_raw_data(X, y, title):
    """
    Plot the raw data.
    """
    # Create plots directory if it doesn't exist
    import os
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Generate the plot
    line_plot(X, y, title)
    
    # Save the plot
    filename = f"{plots_dir}/raw_data_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved as: {filename}")


def test_garch(p, q, df, horizon_char=horizon_char, horizon_vals=horizon_vals):
    """
    Test GARCH model with given p and q parameters across different horizons and context windows.
    
    Parameters:
    -----------
    p : int
        GARCH autoregressive parameter
    q : int  
        GARCH moving average parameter
    df : pandas.DataFrame
        Input dataframe with price data
    horizon_char : list
        Horizon character labels
    horizon_vals : list
        Horizon values in days
    
    Returns:
    --------
    None (saves results to CSV file)
    """
    hashMap = {i: [] for i in horizon_vals}
    context_window_vals = [5, 10, 20, 50, 100, 200, 500, 1000]
    
    for i, (horizon_pred, col_name) in enumerate(zip(horizon_vals, horizon_char)):
        losses = []
        for context_window in context_window_vals:
            print(f'{i+1}/{len(horizon_vals)}, cw = {context_window}, horizon = {horizon_vals[i]}')
            
            vols_pred = []
            for idx in tqdm(range(context_window, df.shape[0])):
                returns_window = df['al_lme_prices_log_returns'][idx-context_window:idx]*100
                model = arch_model(returns_window, vol='Garch', p=p, q=q, dist='normal', mean='constant', rescale=False)
                res = model.fit(disp='off')
                forecast = res.forecast(horizon=horizon_pred)
                
                next_volatility = np.sqrt(252)*np.sqrt(forecast.variance.values[0][-1])/100
                vols_pred.append(next_volatility)

            true_vol = df[f'{col_name}_vol'][context_window:]
            vols_pred = np.array(vols_pred)

            losses.append(mase(true_vol, vols_pred))
            print(context_window, '->', mase(true_vol, vols_pred))
        hashMap[horizon_pred] = losses

    cw_df = pd.DataFrame.from_dict(hashMap)
    cw_df.index = context_window_vals
    cw_df = cw_df.sort_index(axis=1)
    
    # Create directory for GARCH model comparison metrics
    os.makedirs('garch_model_comparison_metrics', exist_ok=True)
    cw_df.to_csv(f'garch_model_comparison_metrics/garch{p}{q}_metrics.csv', index=True)
    print(f'p = {p}, q = {q}\n', cw_df)


def run_garch_grid_search(df, values=[(2,2),(2,1),(2,0),(1,1)]):
    """
    Run grid search over different GARCH parameter combinations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with price data
    values : list of tuples
        List of (p, q) parameter combinations to test
    """
    for p, q in values:
        test_garch(p, q, df)


def generate_volatility_forecasts(df, bestp=2, bestq=1, horizon_char=horizon_char, horizon_vals=horizon_vals):
    """
    Generate volatility forecasts using the best GARCH parameters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with price data
    bestp : int
        Best GARCH autoregressive parameter
    bestq : int
        Best GARCH moving average parameter
    horizon_char : list
        Horizon character labels
    horizon_vals : list
        Horizon values in days
        
    Returns:
    --------
    tuple : (pred_volatilities, avg_pred_volatilities)
        Dictionaries containing predictions
    """
    pred_volatilities = {i:[] for i in horizon_char}
    avg_pred_volatilities = {i:[] for i in horizon_char}

    for i, (horizon_pred, col_name) in enumerate(zip(horizon_vals, horizon_char)):
        context_window = horizon_pred*2
        print(f'{i+1}/{len(horizon_vals)}, cw = {context_window}, horizon = {horizon_vals[i]}')
        
        vols_pred = []
        avg_vols_pred = []
        for idx in tqdm(range(context_window, df.shape[0])):
            returns_window = df['al_lme_prices_log_returns'][idx-context_window:idx]*100
            model = arch_model(returns_window, vol='Garch', p=bestp, q=bestq, dist='normal', mean='constant', rescale=False)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=horizon_pred)
            
            # Take the forecast at the horizon step (last forecast)
            next_volatility = np.sqrt(252)*np.sqrt(forecast.variance.values[0][-1])/100
            avg_next_volatility = np.sqrt(252)*np.sqrt(np.mean(forecast.variance.values[0]))/100
            vols_pred.append(next_volatility)
            avg_vols_pred.append(avg_next_volatility)

        vols_pred = np.array(vols_pred)
        pred_volatilities[col_name] = vols_pred
        avg_pred_volatilities[col_name] = avg_vols_pred

    # Create directory for prediction files
    os.makedirs('volatility_predictions', exist_ok=True)
    
    # Save predictions to JSON files
    with open("volatility_predictions/pred_volatilities.json", "w") as f:
        json.dump({k: list(map(float, v)) for k, v in pred_volatilities.items()}, f, indent=4)

    with open("volatility_predictions/avg_pred_volatilities.json", "w") as f:
        json.dump({k: list(map(float, v)) for k, v in avg_pred_volatilities.items()}, f, indent=4)
    
    return pred_volatilities, avg_pred_volatilities


def calculate_metrics(df, predictions_file="volatility_predictions/avg_pred_volatilities.json", output_file="garch_forecast_metrics.csv"):
    """
    Calculate forecast accuracy metrics for volatility predictions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with price data
    predictions_file : str
        Path to JSON file containing predictions
    output_file : str
        Path to save metrics CSV file
        
    Returns:
    --------
    pandas.DataFrame : Metrics dataframe
    """
    with open(predictions_file, "r") as f:
        avg_pred_volatilities_test = json.load(f)
    
    metric_names = ['MAPE', 'MAE', 'MSE', 'RMSE', 'MASE']
    metrics_df = pd.DataFrame(columns=['Expiry'] + metric_names)

    for expiry_date, vals in avg_pred_volatilities_test.items():
        metrics = []
        for metric in [mape, mae, mse, rmse, mase]:
            true_vol = df[f'{expiry_date}_vol'][pred_char_to_value(expiry_date) * 2:]
            assert len(true_vol) == len(vals), f"Length mismatch for {expiry_date}"
            res = metric(vals, true_vol)
            metrics.append(res)

        metrics_df.loc[len(metrics_df)] = [expiry_date] + metrics

    metrics_df.to_csv(output_file, index=False)
    return metrics_df


def calculate_fair_metrics(df, predictions_file="volatility_predictions/avg_pred_volatilities.json", output_file="garch_forecast_metrics_fair_250.csv"):
    """
    Calculate forecast accuracy metrics for the last 20% of data (fair evaluation).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with price data
    predictions_file : str
        Path to JSON file containing predictions
    output_file : str
        Path to save metrics CSV file
        
    Returns:
    --------
    pandas.DataFrame : Metrics dataframe
    """
    with open(predictions_file, "r") as f:
        avg_pred_volatilities_test = json.load(f)
    
    metric_names = ['MAPE', 'MAE', 'MSE', 'RMSE', 'MASE']
    metrics_df = pd.DataFrame(columns=['Expiry'] + metric_names)

    # Use last 488 observations for fair evaluation
    for expiry_date, vals in avg_pred_volatilities_test.items():
        metrics = []
        for metric in [mape, mae, mse, rmse, mase]:
            true_vol = df[f'{expiry_date}_vol'][-250:]
            vals_subset = vals[-250:]
            assert len(true_vol) == len(vals_subset), f"Length mismatch for {expiry_date}"
            res = metric(vals_subset, true_vol)
            metrics.append(res)

        metrics_df.loc[len(metrics_df)] = [expiry_date] + metrics

    metrics_df.to_csv(output_file, index=False)
    return metrics_df


def generate_volatility_plots(df, predictions_file="volatility_predictions/avg_pred_volatilities.json", plot_dir="volatility_plots"):
    """
    Generate and save volatility comparison plots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with price data
    predictions_file : str
        Path to JSON file containing predictions
    plot_dir : str
        Directory to save plots
    """
    with open(predictions_file, "r") as f:
        avg_pred_volatilities_test = json.load(f)
    
    os.makedirs(plot_dir, exist_ok=True)

    for expiry_date, vals in avg_pred_volatilities_test.items():
        # Plot predicted vs true volatility
        ax, fig = line_plot(
            df.index[pred_char_to_value(expiry_date) * 2:], 
            vals, 
            f'vol_pred_avg, horizon={pred_char_to_value(expiry_date)}', 
            graphtitle=f'{expiry_date}_vol', 
            show=False
        )
        _, _ = line_plot(
            df.index[pred_char_to_value(expiry_date) * 2:], 
            df[f'{expiry_date}_vol'][2*pred_char_to_value(expiry_date):], 
            'vol_true', 
            linecolor='red', 
            ax=ax, 
            show=False
        )
        fig.savefig(f'{plot_dir}/{expiry_date}_vol_vs_true_garch.png', dpi=300)
        plt.close(fig)
        
        # Plot volatility vs log returns
        ax, fig = line_plot(
            df.index[pred_char_to_value(expiry_date) * 2:], 
            df[f'al_lme_prices_log_returns'][2*pred_char_to_value(expiry_date):], 
            'log_returns', 
            graphtitle=f'{expiry_date}_vol', 
            linecolor='red', 
            show=False
        )
        _, _ = line_plot(
            df.index[pred_char_to_value(expiry_date) * 2:], 
            vals, 
            f'vol_pred_avg, horizon={pred_char_to_value(expiry_date)}', 
            ax=ax, 
            show=False
        )
        fig.savefig(f'{plot_dir}/{expiry_date}_vol_vs_log_returns_garch.png', dpi=300)
        plt.close(fig)


def generate_fair_volatility_plots(df, predictions_file="volatility_predictions/avg_pred_volatilities.json", plot_dir="volatility_plots"):
    """
    Generate and save volatility comparison plots for fair evaluation (last 20% of data).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with price data
    predictions_file : str
        Path to JSON file containing predictions
    plot_dir : str
        Directory to save plots
    """
    with open(predictions_file, "r") as f:
        avg_pred_volatilities_test = json.load(f)
    
    os.makedirs(plot_dir, exist_ok=True)

    for expiry_date, vals in avg_pred_volatilities_test.items():
        # Plot predicted vs true volatility (fair evaluation)
        ax, fig = line_plot(
            df.index[-250:], 
            vals[-250:], 
            f'vol_pred_avg, horizon={pred_char_to_value(expiry_date)}', 
            graphtitle=f'{expiry_date}_vol', 
            show=False
        )
        _, _ = line_plot(
            df.index[-250:], 
            df[f'{expiry_date}_vol'][-250:], 
            'vol_true', 
            linecolor='red', 
            ax=ax, 
            show=False
        )
        fig.savefig(f'{plot_dir}/{expiry_date}_vol_vs_true_garch_fair.png', dpi=300)
        plt.close(fig)
        
        # Plot volatility vs log returns (fair evaluation)
        ax, fig = line_plot(
            df.index[-250:], 
            df[f'al_lme_prices_log_returns'][-250:], 
            'log_returns', 
            graphtitle=f'{expiry_date}_vol', 
            linecolor='red', 
            show=False
        )
        _, _ = line_plot(
            df.index[-250:], 
            vals[-250:], 
            f'vol_pred_avg, horizon={pred_char_to_value(expiry_date)}', 
            ax=ax, 
            show=False
        )
        fig.savefig(f'{plot_dir}/{expiry_date}_vol_vs_log_returns_garch_fair.png', dpi=300)
        plt.close(fig)


def run_complete_garch_analysis(data_path='../Data/aluminium_pre_inputs.csv'):
    """
    Run the complete GARCH analysis pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to the input data file
    """
    # Load data
    df = load_data(data_path)

    # print("Plotting raw data...")
    # plot_raw_data(df.index, df['al_lme_prices_log_returns'], 'al_lme_prices_log_returns')

    # print("Plotting squared PACF...")
    # plot_squared_pacf(df, 'al_lme_prices_log_returns')
    
    # Run grid search for best parameters
    # print("Running GARCH grid search...")
    # run_garch_grid_search(df)
    
    # Generate forecasts with best parameters
    # print("Generating volatility forecasts...")
    # pred_volatilities, avg_pred_volatilities = generate_volatility_forecasts(df)
    
    # Calculate metrics
    print("Calculating metrics...")
    # metrics_df = calculate_metrics(df)
    fair_metrics_df = calculate_fair_metrics(df)
    
    # Generate plots
    print("Generating plots...")
    generate_volatility_plots(df)
    generate_fair_volatility_plots(df)
    
    print("Analysis complete!")
    return metrics_df, fair_metrics_df


if __name__ == "__main__":
    # Run the complete analysis
    metrics_df, fair_metrics_df = run_complete_garch_analysis()
    print("All GARCH metrics:")
    print(metrics_df)
    print("\nFair GARCH metrics:")
    print(fair_metrics_df)
