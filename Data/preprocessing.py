# imports
import pandas as pd
import numpy as np

import sys
import os

sys.path.append(os.path.abspath('..'))
from functions import compute_daily_volatility

# load csv
df = pd.read_csv('aluminium_raw_inputs.csv')

# remove top 164 rows
df = df.iloc[164:].copy()
df.reset_index(drop=True, inplace=True)

# remove columns with the same values
def net_absolute_difference(column1, column2):
    curSum = 0
    for val in df[column1] - df[column2]:
        curSum += abs(val)
    return curSum

matching_columns = []
for i in range(1, df.shape[1]):
    for j in range(i+1, df.shape[1]):
        column1, column2 = df.columns[i], df.columns[j]
        if net_absolute_difference(column1, column2) == 0:
            matching_columns.append((column1, column2))

duplicates = []
for item in matching_columns:
    duplicates.append(item[0])

df = df.drop(duplicates, axis=1)

# change column names for convenience
col_map = dict()
for col in df.columns:
    if len(col.split(', ')) == 1: continue
    col_split = col.split("', '")
    feature_name, col_name = col_split[0][2:], col_split[1][:-2]
    df.rename(columns={col: col_name}, inplace=True)
    col_map[col_name] = feature_name

# convert date column to timeseries
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# create log-returns and daily returns
df.insert(1, 'al_lme_prices_log_returns', np.log(df['al_lme_prices'] / df['al_lme_prices'].shift(1)))
df.insert(2, 'al_lme_prices_daily_returns', (df['al_lme_prices'] - df['al_lme_prices'].shift(1)) / df['al_lme_prices'].shift(1))
df = df.dropna().reset_index(drop=True)
df.drop(columns=['al_lme_prices'], inplace=True)

# reverse spot prices 
inverse_currencies = ['euro_spot', 'australian_dollar_spot', 'uk_pound_spot']

for forex_ratio in inverse_currencies:
    df[forex_ratio] = 1 / df[forex_ratio]

# volatilities
for window_size, col_name in zip([5, 22, 66, 252], ['weekly', 'monthly', 'quarterly', 'yearly']):
    vol_window = compute_daily_volatility(df['al_lme_prices_log_returns'], window=window_size, method='rolling')
    df[f'{col_name}_vol'] = vol_window

# save new csv
df.to_csv('aluminium_pre_inputs.csv', index=False)
print('preprocessing complete and csv created')