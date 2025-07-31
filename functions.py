import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns

def line_plot(dates, values, ylabel, linecolor='blue', ax=None, show=True):
    # create new figure and axes only if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
        new_plot = True
    else:
        new_plot = False

    # plot the line
    ax.plot(dates, values, label=ylabel, color=linecolor)

    # configure the axes once
    if new_plot:
        ax.set_xlabel('Date')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    # update the title and legend
    ax.set_title("Time Series")
    ax.legend()

    if show and new_plot:
        plt.tight_layout()
        plt.show()

    return ax