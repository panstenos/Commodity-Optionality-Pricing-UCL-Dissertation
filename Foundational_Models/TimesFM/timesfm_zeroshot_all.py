import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from itertools import product
from collections import defaultdict

from timesfm_functions import (
    TimesFMModel,
    load_aluminium_data,
    calculate_prediction_metrics,
)

from functions import line_plot, pred_value_to_char, find_n_best_features

def get_batched_data_fn(df, batch_size, context_len, horizon_len):
    examples = defaultdict(list)
    sub_df = df.iloc[horizon_len:, :].reset_index(drop=True)
    N = len(sub_df)

    vol_col = f"{pred_value_to_char(horizon_len)}_vol"
    if vol_col not in sub_df.columns:
        raise KeyError(f"Column '{vol_col}' not found in sub_df")

    for start in range(0, N - (context_len + horizon_len) + 1, 1):
        context_end = start + context_len
        examples["inputs"].append(sub_df[vol_col].iloc[start:context_end].to_list())
    num_examples = len(examples["inputs"])

    def data_fn():
        for i in range(0, num_examples, batch_size):
            yield {k: v[i:i + batch_size] for k, v in examples.items()}

    return data_fn

def _last_step(x):
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Forecast must be 2D (batch, horizon), got {arr.shape}")
    return arr[:, -1].astype(float).tolist()

def _freq_list(freq, batch_size):
    if isinstance(freq, int):
        return [freq] * batch_size
    if isinstance(freq, (list, tuple, np.ndarray)):
        if len(freq) == 1:
            return [int(freq[0])] * batch_size
        if len(freq) == batch_size:
            return list(map(int, freq))
        raise ValueError(f"'freq' length {len(freq)} != batch size {batch_size}")
    raise TypeError(f"Unsupported freq type: {type(freq)}")


def run_inference(model, input_data, freq):
    """
    Pass context_len and horizon_len if you want dynamic-length checks; otherwise deduced from first batch.
    """

    vanilla_forecasts = []

    for example in input_data():
        bsz = len(example["inputs"])
        f_list = _freq_list(freq, bsz)

        raw_forecast, _ = model.predict(inputs=example["inputs"], freq=f_list)
        vanilla_forecasts.extend(_last_step(raw_forecast))

    return vanilla_forecasts

###########################################################################

def plotting_fn(df, vanilla_forecasts, horizon_len):
    pred_van = vanilla_forecasts[-250:]
    test_true = df[f'{pred_value_to_char(horizon_len)}_vol'][-250:].tolist()

    ax, fig = line_plot(pred_van, pred_van, 'pred_van', show=False)
    _, _ = line_plot(test_true, test_true, 'test', ax=ax, linecolor='red', show=False)

    return pred_van, test_true, fig

df = load_aluminium_data()

batch_size = 128
horizon_grid = [5, 22, 66, 252]
context_grid = [32, 64, 128, 256, 512, 1024]
freq_grid = [0, 1, 2]
pos_embed_grid = [True, False]

root_dir = "timesfm_zeroshot_all_results"
metrics_dir = os.path.join(root_dir, "metrics")
plots_dir = os.path.join(root_dir, "plots")
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

for expiry in horizon_grid:
    rows = []
    metrics_csv = os.path.join(metrics_dir, f"metrics_expiry_{expiry}.csv")

    for pos_emb, window_size, f in product(pos_embed_grid, context_grid, freq_grid):
        if window_size < expiry:
            continue

        print(f"[run] exp={expiry} win={window_size} freq={f} pos={int(pos_emb)}")

        model = TimesFMModel(expiry=expiry, context_length=window_size, positional_embedding=pos_emb)
        input_data = get_batched_data_fn(df, batch_size=batch_size, context_len=window_size, horizon_len=expiry)

        preds = []
        for example in input_data():
            bsz = len(example["inputs"])
            f_list = [f] * bsz
            raw_forecast, _ = model.predict(inputs=example["inputs"], freq=f_list)
            preds.extend(np.asarray(raw_forecast)[:, -1].astype(float).tolist())

        pred_van, test_true, fig = plotting_fn(df=df, vanilla_forecasts=preds, horizon_len=expiry)

        plot_path = os.path.join(
            plots_dir,
            f"pred_vs_true_{pred_value_to_char(expiry)}_window{window_size}_freq{f}_pos{int(pos_emb)}.png"
        )
        fig = fig or plt.gcf()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {plot_path}")

        m = calculate_prediction_metrics(pred_van, test_true)
        rows.append({
            "expiry": expiry,
            "window_size": window_size,
            "freq": f,
            "positional_embedding": int(pos_emb),
            "mape": float(m["MAPE"]),
            "mae":  float(m["MAE"]),
            "rmse": float(m["RMSE"]),
            "mse":  float(m["MSE"]),
            "mase": float(m["MASE"]),
        })
        print(f"[metrics] MAPE={float(m['MAPE']):.3f} MAE={float(m['MAE']):.4f} RMSE={float(m['RMSE']):.4f} MSE={float(m['MSE']):.6f} MASE={float(m['MASE']):.3f}")

    pd.DataFrame(rows, columns=[
        "expiry","window_size","freq","positional_embedding","mape","mae","rmse","mse","mase"
    ]).to_csv(metrics_csv, index=False)
    print(f"[saved] {metrics_csv}")