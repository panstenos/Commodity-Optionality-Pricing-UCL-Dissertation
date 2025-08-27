import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from itertools import product

from timesfm_functions import (
    TimesFMModel,
    load_aluminium_data,
    calculate_prediction_metrics
)

from functions import line_plot, pred_value_to_char


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
# load data once at import so it's available to run()
df = load_aluminium_data()


def run(expiry, window_size, normalize, freq_input):
    """
    Execute a single training/evaluation run for the given hyperparameters.

    Parameters
    ----------
    expiry : int
        Forecast horizon in steps (e.g., 5, 22, 66, 252).
    window_size : int
        Context length provided to the model (e.g., 32, 64, 128, ...).
    normalize : bool
        Whether to normalize the input inside model.predict (passed through).
    freq_input : list[int]
        Frequency conditioning input for the model (e.g., [0], [1], [2]).

    Returns
    -------
    list
        A single metrics row in the order:
        [expiry, window_size, int(normalize), freq_input[0],
         MAPE, MAE, RMSE, MSE, MASE]

    Side Effects
    ------------
    Saves a PNG plot comparing the first 300 predicted vs true points to
    timesfm_vanilla_results/plots/ with a filename that encodes the
    hyperparameters.
    """
    X = df[f'{pred_value_to_char(expiry)}_vol'].dropna()
    model = TimesFMModel(expiry=expiry, context_length=window_size)

    # rolling predictions (take the last-step forecast from each window)
    preds = []
    for i in tqdm(range(len(X) - window_size - expiry + 1)):
        mean_pred, quantiles = model.predict(
            forecast_input=X[i:i+window_size],
            frequency_input=freq_input,
            normalize=normalize
        )
        preds.append(mean_pred[-1])

    preds = np.asarray(preds)
    true = X[window_size + expiry - 1 : window_size + expiry - 1 + len(preds)].to_numpy()

    # plot & save (first 300 points)
    ax, fig = line_plot(range(len(preds[:300])), preds[:300], 'pred', show=False)
    _, _   = line_plot(range(len(true[:300])),  true[:300],  'true', linecolor='red', ax=ax, show=False)
    fig.savefig(os.path.join(plots_dir, f"pred_vs_true_{pred_value_to_char(expiry)}_window{window_size}_freq{freq_input[0]}_normal{normalize}.png"),
                dpi=150, bbox_inches="tight")

    # metrics
    metrics = calculate_prediction_metrics(preds, true)

    return [
        expiry,
        window_size,
        int(normalize),
        freq_input[0],
        metrics["MAPE"],
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MSE"],
        metrics["MASE"]
    ]


def main():
    """
    Run the grid per expiry and save a separate CSV for each expiry.

    - Ensures output directories exist
    - Iterates over the hyperparameter grid per expiry
    - Skips combos where expiry // 2 > window_size
    - Saves per-expiry metrics to metrics_expiry_{expiry}.csv
    """
    # make dirs to save results
    base_dir = "timesfm_vanilla_results"
    global plots_dir, metrics_dir  # used in run() for saving plots
    plots_dir = os.path.join(base_dir, "plots")
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    expiries      = [5, 22, 66, 252]
    window_sizes  = [32, 64, 128, 256, 512, 1024]
    normalizes    = [True, False]
    freq_inputs   = [[0], [1], [2]]

    for expiry in expiries:
        rows = []
        print(f"\n=== Running grid for expiry={expiry} ===")
        for window_size, normalize, freq_input in product(window_sizes, normalizes, freq_inputs):
            if expiry // 2 > window_size:
                continue
            print(f"expiry={expiry}, window_size={window_size}, normalize={normalize}, freq_input={freq_input}")
            row = run(expiry, window_size, normalize, freq_input)
            rows.append(row)

        # save per-expiry metrics
        metrics_df = pd.DataFrame(
            rows,
            columns=["expiry", "window_size", "normalize", "freq_input", "mape", "mae", "rmse", "mse", "mase"]
        )
        metrics_csv = os.path.join(metrics_dir, f"metrics_expiry_{expiry}.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics for expiry {expiry} to: {metrics_csv}")

    print(f"\nAll plots saved to: {plots_dir}")



if __name__ == "__main__":
    main()