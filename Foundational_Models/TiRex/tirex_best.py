# # Only for Google Colab Notebook!

# import os

# # Clone TiRep Repo
# !git clone https://github.com/NX-AI/tirex

# # Install TiRex
# os.chdir('/content/tirex')
# !pip install .[gluonts]

# # Set Workin Dir to notebooks folder
# os.chdir('/content/tirex/examples')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
from pathlib import Path

import torch
from util_plot import plot_fc

from itertools import product
import json, os, math
from collections.abc import Mapping, Sequence

from tirex import ForecastModel, load_model

def line_plot(dates, values, ylabel, graphtitle='Time Series', linecolor='blue', ax=None, show=True, useDates=False):
    """
    Plots a time series line chart with dates on the x-axis.

    Parameters:
    - dates (array-like): Sequence of date values.
    - values (array-like): Corresponding values to plot.
    - ylabel (str): Label for the y-axis and legend.
    - graphtitle (str): Title of the plot.
    - linecolor (str): Line color (default is 'blue').
    - ax (matplotlib.axes.Axes): Optional existing axes to plot on.
    - show (bool): Whether to display the plot immediately.

    Returns:
    - ax: The matplotlib axes object.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
        new_plot = True
    else:
        new_plot = False
        fig = None

    if useDates: 
        dates = pd.to_datetime(dates)
    else:
        dates = np.arange(0, len(values))
    ax.plot(dates, values, label=ylabel, color=linecolor)

    if new_plot:
        ax.set_ylabel(ylabel)
        if useDates:
            ax.set_xlabel('Date')
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.set_xlabel('trading day')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.grid(True)
        ax.set_title(graphtitle)
    ax.legend()

    if show and new_plot:
        plt.tight_layout()
        plt.show()

    return ax, fig

def mase(y_true, y_pred):
    """
    Calculates Mean Absolute Scaled Error (MASE).

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - float: The MASE score.
    """
    assert len(y_true) == len(y_pred), "sequences have different lengths"

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)
    # Denominator: average absolute difference of true values (naïve forecast error)
    d = np.abs(np.diff(y_true)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d


def rmse(y_true, y_pred):
    """
    Calculates Root Mean Squared Error (RMSE).

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - float: RMSE score.
    """
    assert len(y_true) == len(y_pred), "sequences have different lengths"

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mse(y_true, y_pred):
    """
    Calculates Mean Squared Error (MSE).

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - float: MSE score.
    """
    assert len(y_true) == len(y_pred), "sequences have different lengths"

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    """
    Calculates Mean Absolute Error (MAE).

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - float: MAE score.
    """
    assert len(y_true) == len(y_pred), "sequences have different lengths"

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred): 
    """
    Calculates Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true (array-like): True values.
    - y_pred (array-like): Predicted values.

    Returns:
    - float: MAPE score.
    """
    assert len(y_true) == len(y_pred), "sequences have different lengths"

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compute_daily_volatility(prices: pd.Series, window: int = 20, method: str = 'rolling') -> pd.Series:
    """
    Estimate daily volatility for each calendar day using rolling or exponential method.

    Parameters:
    - prices (pd.Series): Price series indexed by datetime.
    - window (int): Size of the lookback window (e.g., 20 days).
    - method (str): 'rolling' for standard rolling std, 'ewm' for exponential weighting.

    Returns:
    - pd.Series: Daily volatility estimates aligned with calendar days.
    """

    if method == 'rolling':
        vol = prices.rolling(window=window).std()
    elif method == 'ewm':
        vol = prices.ewm(span=window, adjust=False).std()
    else:
        raise ValueError("Method must be either 'rolling' or 'ewm'.")

    return np.sqrt(252)*vol

def pred_char_to_value(s):
    hashMap = {'1w':5, '1m':22, '3m':66, '1y':252}
    return hashMap[s]

def pred_value_to_char(n):
    hashMap = {5:'1w', 22:'1m', 66:'3m', 252:'1y'}
    return hashMap[n]

def load_data(data_path='../Data/aluminium_pre_inputs.csv', index_col=None):
    """Load the aluminium price data."""
    return pd.read_csv(data_path, index_col=0)

def plot_squared_pacf(df, col_name):
    """
    Plot the Partial AutoCorrelation Function for the squared values of a given column in a dataframe.
    """
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Generate the plot
    plot_pacf(df[col_name]**2)
    
    # Save the plot
    filename = f"{plots_dir}/pacf_squared_{col_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Plot saved as: {filename}")

def find_n_best_features(expiry, n):
    """
    Find the n best features based on correlation with volatility for a given expiry.
    
    Parameters:
    - expiry (int): The expiry period (5, 22, 66, or 252)
    - n (int): Number of top features to return
    
    Returns:
    - list: List of feature names representing the top n features
    """
    # Get the absolute path to the feature selection directory from functions.py
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    feature_selection_dir = os.path.join(current_file_dir, 'Feature_selection')
    corr_file = os.path.join(feature_selection_dir, 'absolute_feature_correlations.csv')
    
    # Load the correlation data
    best_features = load_data(corr_file, index_col=0)
    
    # Sort by correlation with the specific expiry and get top n
    top_rows = best_features.sort_values(by=f'{pred_value_to_char(expiry)}_exp', ascending=False).head(n)
    best_features = top_rows.index.tolist()
    
    return best_features

def calculate_prediction_metrics(predictions, true_values):
    """
    Calculate common error metrics to evaluate prediction accuracy.

    Parameters
    ----------
    predictions : array-like
        The predicted values from a model.
    true_values : array-like
        The actual observed/true values.

    Returns
    -------
    dict
        A dictionary containing:
        - "MAPE": Mean Absolute Percentage Error
            Measures prediction accuracy as a percentage; lower is better.
        - "MAE": Mean Absolute Error
            Average of absolute differences between predictions and true values.
        - "RMSE": Root Mean Squared Error
            Square root of average squared differences; penalizes large errors.
        - "MSE": Mean Squared Error
            Average of squared differences; sensitive to outliers.
        - "MASE": Mean Absolute Scaled Error
            Scale-independent error metric useful for comparing across datasets.
    """

    # Coerce to numpy arrays
    y_pred = np.asarray(predictions)
    y_true = np.asarray(true_values)

    # Make them column vectors if 1D
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    # Align lengths (guard against off-by-one slicing)
    n = min(len(y_pred), len(y_true))
    if n == 0:
        raise ValueError("Empty inputs: predictions/true_values have no overlapping length.")
    y_pred = y_pred[:n]
    y_true = y_true[:n]


    mape_val = mape(true_values, predictions)
    mae_val = mae(true_values, predictions)
    rmse_val = rmse(true_values, predictions)
    mse_val = mse(true_values, predictions)
    mase_val = mase(true_values, predictions) 

    return {
        "MAPE": mape_val,
        "MAE": mae_val,
        "RMSE": rmse_val,
        "MSE": mse_val,
        "MASE": mase_val
    }

# Set seed for reproducibility
def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model: ForecastModel = load_model("NX-AI/TiRex")
df = pd.read_csv('/content/aluminium_pre_inputs.csv')

def get_data_tirex(horizon_len, df):
    data = df[f'{pred_value_to_char(horizon_len)}_vol'].dropna().values
    return data

def make_preds_tirex(data, horizon_len, context_len, seed=42):
    # Set seed for reproducibility
    set_seed(seed)
    
    LENGTH = len(data)
    print(f'total_data -> {LENGTH}')
    
    CONTEXT_END = LENGTH - horizon_len - context_len
    print(f'context_end -> {CONTEXT_END}')

    preds = []
    for start in range(CONTEXT_END-250, CONTEXT_END, 1):
        quantiles, mean = model.forecast(data[start:start+context_len], prediction_length=horizon_len)
        preds.append(mean.flatten()[-1])

    return preds, data[-250:]

def _is_tensor_like(x):
    return (
        (hasattr(x, "detach") and callable(getattr(x, "detach", None)) and
         hasattr(x, "cpu")    and callable(getattr(x, "cpu",    None)) and
         hasattr(x, "numpy")  and callable(getattr(x, "numpy",  None)))
        or hasattr(x, "tolist")
    )

def _to_list_from_tensor(x):
    try:
        if hasattr(x, "detach") and hasattr(x, "cpu"):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass
    try:
        if hasattr(x, "numpy"):
            return x.numpy().tolist()
    except Exception:
        pass
    if hasattr(x, "tolist"):
        return x.tolist()
    return str(x)

def make_json_safe(obj):
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import pandas as pd
    except Exception:
        pd = None

    # primitives
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    # tensors / arrays
    if _is_tensor_like(obj):
        return make_json_safe(_to_list_from_tensor(obj))

    if np is not None:
        if isinstance(obj, np.generic):
            v = obj.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return make_json_safe(obj.tolist())

    if pd is not None:
        if isinstance(obj, pd.Series):
            return make_json_safe(obj.to_list())
        if isinstance(obj, pd.DataFrame):
            return {str(c): make_json_safe(obj[c].to_list()) for c in obj.columns}

    # containers
    if isinstance(obj, Mapping):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, Path):
        return str(obj)

    return repr(obj)

# =========================================
# Paths & safe writes (Colab-friendly)
# =========================================
def setup_tirex_best_results(root=None):
    base = Path("/content") if Path("/content").exists() else Path.cwd()
    root = Path(root) if root else (base / "tirex_best_results")
    paths = {
        "root": root,
        "plots": root / "plots",
        "metrics": root / "metrics",
        "metadata": root / "metadata",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    print(f"[dirs] root: {paths['root'].resolve()}")
    return paths

def save_json_safely(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(make_json_safe(obj), fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())

def verify_file(p: Path, label: str):
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    print(f"[{label}] {p.resolve()}  exists={exists}  size={size}")

# =========================================
# Run experiments with best combinations only
# =========================================
dirs = setup_tirex_best_results()  # -> /content/tirex_best_results/...

# Best performing combinations: (horizon, context)
best_combinations = [
    (5, 512),
    (22, 64), 
    (66, 256),
    (252, 512)
]

# Seeds for 5 different runs
seeds = [42, 123, 456, 789, 999]

print(f"Testing {len(best_combinations)} best combinations with {len(seeds)} runs each")
print(f"Total experiments: {len(best_combinations) * len(seeds)}")

# Store all results for statistical analysis
all_results = []

c = 0
for h_len, c_len in best_combinations:
    print(f"\n=== Testing combination (h={h_len}, c={c_len}) ===")
    
    # Store metrics for this combination across all runs
    combo_metrics = []
    
    # Run 5 experiments with different seeds
    for run_idx, seed in enumerate(seeds):
        c += 1
        print(f"\n--- Run {run_idx+1}/5 with seed {seed} (experiment {c}/{len(best_combinations)*len(seeds)}) ---")
        
        # Your data + preds
        data = get_data_tirex(h_len, df)
        preds, true = make_preds_tirex(data, h_len, c_len, seed=seed)

        # Plot on same fig (only for first run to avoid too many plots)
        if run_idx == 0:
            ax, fig = line_plot(true, true, "true", graphtitle=f"h{h_len}_c{c_len}_best", linecolor="red", show=False)
            line_plot(preds, preds, "pred", ax=ax, show=False)

            # Save plot
            plot_path = dirs["plots"] / f"series_h{h_len}_c{c_len}_best.png"
            fig.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            verify_file(plot_path, "plot")

        # Metrics
        metrics = calculate_prediction_metrics(preds, true)
        combo_metrics.append(metrics)
        
        # Save individual run metrics
        metrics_path = dirs["metrics"] / f"metrics_h{h_len}_c{c_len}_run{run_idx+1}_seed{seed}.json"
        save_json_safely(metrics, metrics_path)
        
        print(f"    Run {run_idx+1} MAPE: {metrics['MAPE']:.4f}")
    
    # Calculate mean and std for this combination
    mean_metrics = {}
    std_metrics = {}
    
    for metric_name in ['MAPE', 'MAE', 'RMSE', 'MSE', 'MASE']:
        values = [run_metrics[metric_name] for run_metrics in combo_metrics]
        mean_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values, ddof=1)  # Sample std deviation
    
    # Add combination info
    combination_info = {
        'horizon': h_len,
        'context': c_len,
        'combination': f'h{h_len}_c{c_len}'
    }
    
    mean_metrics.update(combination_info)
    std_metrics.update(combination_info)
    
    all_results.append({
        'combination': combination_info,
        'mean': mean_metrics,
        'std': std_metrics
    })
    
    print(f"Combination (h={h_len}, c={c_len}) - Mean MAPE: {mean_metrics['MAPE']:.4f} ± {std_metrics['MAPE']:.4f}")

# Save summary results to CSV files
print("\n=== Saving Summary Results ===")

# Prepare data for CSV
mean_data = []
std_data = []

for result in all_results:
    mean_row = result['mean'].copy()
    std_row = result['std'].copy()
    
    mean_data.append(mean_row)
    std_data.append(std_row)

# Create DataFrames
mean_df = pd.DataFrame(mean_data)
std_df = pd.DataFrame(std_data)

# Save to CSV
mean_csv_path = dirs["metrics"] / "tirex_best_mean_results.csv"
std_csv_path = dirs["metrics"] / "tirex_best_std_results.csv"

mean_df.to_csv(mean_csv_path, index=False)
std_df.to_csv(std_csv_path, index=False)

print(f"Mean results saved to: {mean_csv_path}")
print(f"Std results saved to: {std_csv_path}")

# Print summary
print("\n=== Final Summary ===")
print("Best combinations tested:")
for result in all_results:
    combo = result['combination']
    mean_mape = result['mean']['MAPE']
    std_mape = result['std']['MAPE']
    print(f"  h={combo['horizon']}, c={combo['context']}: MAPE = {mean_mape:.4f} ± {std_mape:.4f}")

print(f"\nTotal experiments completed: {len(best_combinations) * len(seeds)}")
print("Results saved in tirex_best_results/ directory:")
print("- Individual run metrics in metrics/ folder")  
print("- Mean metrics in tirex_best_mean_results.csv")
print("- Standard deviation metrics in tirex_best_std_results.csv")
print("- Plots in plots/ folder")
