import pandas as pd

# Read CSV files into separate dataframes
corn_df = pd.read_csv('Corn_data.csv')
soybeans_df = pd.read_csv('Soybeans_data.csv')
wheat_df = pd.read_csv('Wheat_data.csv')

# Select the 'adj_close' column from each dataframe
date = corn_df['Date']
corn_adj_close = corn_df['Adj Close']
soybeans_adj_close = soybeans_df['Adj Close']
wheat_adj_close = wheat_df['Adj Close']

# Concatenate selected columns into a single dataframe
agri_commod_df = pd.concat([date,corn_adj_close, soybeans_adj_close, wheat_adj_close], axis=1)
agri_commod_df.columns = ['Date','Corn', 'Soybeans', 'Wheat']  # Rename columns
# print(agri_commod_df)

# Read CSV file
fx_df = pd.read_csv('fx_rates.csv')
# print(fx_df)

# Merge on Date
agri_fx_merged_df= pd.merge(agri_commod_df, fx_df, on='Date')

### PREPARE DATA FOR COINTEGRATION ANALYSIS

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

# Extract commodity columns
commodity_cols = agri_fx_merged_df.columns[1:4]

# Extract FX rate columns
fx_cols = agri_fx_merged_df.columns[4:]

# Create a new DataFrame with only commodity and FX rate columns
df_selected = agri_fx_merged_df[commodity_cols.tolist() + fx_cols.tolist()]

# Replace infinite values with NaN
df_selected = df_selected.replace([np.inf, -np.inf], np.nan)

# Drop rows with any NaN values
df_selected = df_selected.dropna()

# Check for stationarity
for col in df_selected.columns:
    result = adfuller(df_selected[col])
    print(f"{col} ADF Statistic: {result[0]}")
    print(f"{col} p-value: {result[1]}")

# Perform cointegration tests
for commodity in commodity_cols:
    for fx in fx_cols:
        result = coint(df_selected[commodity], df_selected[fx])
        print(f"{commodity} and {fx} cointegration test p-value: {result[1]}")



