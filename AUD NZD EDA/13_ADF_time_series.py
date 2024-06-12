import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fetching data from Yahoo Finance
sgdusd_data = yf.download('SGDUSD=X', start='2016-01-01', end='2024-05-29')
gbpusd_data = yf.download('GBPUSD=X', start='2016-01-01', end='2024-05-29')

# Extract the adjusted close price
sgdusd_close = sgdusd_data['Adj Close']
gbpusd_close = gbpusd_data['Adj Close']

# Define window sizes
window_sizes = [12 * i for i in range(1, 8)]

# Start ADF test after accumulating 12 months of data
start_date = '2017-01-01'

for window_size in window_sizes:
    # Step 1: Fit a linear model
    residuals = gbpusd_close - sgdusd_close

    # Calculate rolling ADF statistic and p-value starting from the specified date
    rolling_adf_stat = residuals.loc[start_date:].rolling(window=window_size).apply(lambda x: adfuller(x)[0])

    # Drop NaN values from rolling statistics
    rolling_adf_stat = rolling_adf_stat.dropna()

    # Creating time variable for regression
    time = np.arange(len(rolling_adf_stat)).reshape(-1, 1)

    # Linear regression for ADF Statistic
    reg_adf = LinearRegression().fit(time, rolling_adf_stat)
    adf_trend = reg_adf.predict(time)

    # Calculate gradient of ADF Statistic Trend Line
    gradient = (adf_trend[-1] - adf_trend[0]) / len(adf_trend)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(top=0.9)  # Adjust top padding
    plt.subplots_adjust(bottom=0.3)  # Adjust bottom padding

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('ADF Statistic', color=color)
    ax1.plot(rolling_adf_stat.index, rolling_adf_stat, label=f'Rolling ADF Statistic ({window_size} Months)', color=color)
    ax1.plot(rolling_adf_stat.index, adf_trend, linestyle='--', color='green', linewidth=2, label='ADF Stat Trend')
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding annotations for ADF Statistic Trend Line Gradient
    start_adf = adf_trend[0]
    end_adf = adf_trend[-1]

    # Calculate middle x-coordinate of the plot's x-axis range
    middle_x = ax1.get_xlim()[0] + 0.5 * (ax1.get_xlim()[1] - ax1.get_xlim()[0])

    # Calculate fixed y-coordinate for text annotations
    y_coord = min(rolling_adf_stat.min(), start_adf) - 6

    # Get the bottom of the plot area
    bottom = ax1.get_ylim()[0]

    # Calculate the new y-coordinate for text annotations
    y_coord = bottom - 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])  # Adjust 0.05 based on your preference

    # Text annotations with adjusted y-coordinate
    ax1.text(middle_x, ax1.get_ylim()[0] * 1.4, f'Trend Line Starting ADF: {start_adf:.2f}', verticalalignment='top', horizontalalignment='center')
    ax1.text(middle_x, ax1.get_ylim()[0] * 1.5, f'Trend Line Final ADF: {end_adf:.2f}', verticalalignment='top', horizontalalignment='center')
    ax1.text(middle_x, ax1.get_ylim()[0] * 1.6, f'ADF Statistic Trend Line Gradient: {gradient:.6f}', verticalalignment='top', horizontalalignment='center', color='purple')

    plt.title(f'Rolling ADF Statistic with Trend Lines (Window Size: {window_size} Months)')
    plt.savefig(f'X13_Rolling_ADF_{window_size}_Months.png')
    # plt.show()



