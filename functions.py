import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

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

    return root_mean_squared_error(y_true, y_pred)


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

    return mean_squared_error(y_true, y_pred)


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

    return mean_absolute_error(y_true, y_pred)

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

    return vol

def pred_char_to_value(s):
    hashMap = {'1w':5, '1m':22, '3m':66, '1y':252}
    return hashMap[s]