import timesfm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the Python path to import functions
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to root
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from functions import line_plot, mape, mae, rmse, mse, mase, pred_value_to_char, load_data

PATCH = 32
MAX_CTX = 2048

class TimesFMModel:
    """Base TimesFM model class with rolling window prediction capabilities."""
    
    def __init__(self, expiry, context_length, num_layers=50, positional_embedding=True, backend=None):
        ctx = max(PATCH, min(context_length, MAX_CTX))
        ctx_aligned = (ctx // PATCH) * PATCH
        if ctx_aligned != context_length:
            print(f"[TimesFM] context_length {context_length} -> aligned to {ctx_aligned} (multiple of {PATCH}).")
        self.context_len = ctx_aligned

        if backend is None:
            try:
                import torch
                backend = "gpu" if torch.cuda.is_available() else "cpu"
            except Exception:
                backend = "cpu"

        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=expiry,
                num_layers=num_layers,
                use_positional_embedding=positional_embedding,
                context_len=self.context_len,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

    def predict(self, inputs, freq=[0], normalize=False):
        """Make predictions using the TimesFM model."""

        point_forecast, experimental_quantile_forecast = self.model.forecast(
            inputs=inputs,
            freq=freq,
            forecast_context_len=self.context_len,
            normalize=normalize
        )
        return point_forecast, experimental_quantile_forecast
        
    def predict_with_covariates(
        self,
        inputs,
        freq,
        *,
        dynamic_numerical_covariates=None,
        dynamic_categorical_covariates=None,
        static_numerical_covariates=None,
        static_categorical_covariates=None,
        xreg_mode="xreg + timesfm",
        ridge=0.0,
        normalize_xreg_target_per_input=True,
        force_on_cpu=False,
    ):
        cov_forecast, ols_forecast = self.model.forecast_with_covariates(
            inputs=inputs,
            dynamic_numerical_covariates=dynamic_numerical_covariates,
            dynamic_categorical_covariates=dynamic_categorical_covariates,
            static_numerical_covariates=static_numerical_covariates,
            static_categorical_covariates=static_categorical_covariates,
            freq=freq,
            xreg_mode="xreg + timesfm",
            ridge=ridge,
            force_on_cpu=False,
            normalize_xreg_target_per_input=normalize_xreg_target_per_input,
        )

        return cov_forecast, ols_forecast


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


def load_aluminium_data(data_path=None):
    """
    Load aluminium data with automatic path detection.
    
    Parameters:
    - data_path: Optional custom path to data file
    
    Returns:
    - DataFrame with aluminium data
    """
    if data_path is None:
        # Try to load real data with automatic path detection
        try:
            df = load_data('../../Data/aluminium_pre_inputs.csv')
        except:
            try:
                df = load_data('../Data/aluminium_pre_inputs.csv')
            except:
                raise FileNotFoundError("Could not find aluminium data file")
    else:
        df = load_data(data_path)
    
    return df