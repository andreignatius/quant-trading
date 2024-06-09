import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Function to download and plot data on different axes
def plot_residuals_with_adf_test(forex_pair, dollar_index, start_date, end_date, interval='1h'):
    try:
        # Download the adjusted close prices for the given tickers with hourly interval
        forex_data = yf.Ticker(forex_pair).history(start=start_date, end=end_date, interval=interval)["Close"]
        index_data = yf.Ticker(dollar_index).history(start=start_date, end=end_date, interval=interval)["Close"]
        
        # Align the indices of the two data series
        data = forex_data.to_frame(name=forex_pair).join(index_data.to_frame(name=dollar_index), how='inner')
        
        # Standardize the scale of both datasets
        data_standardized = (data - data.mean()) / data.std()
        
        # Calculate residuals
        residuals = data_standardized[forex_pair] - data_standardized[dollar_index]
        
        # Plot the residuals
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(residuals.index, residuals, label='Residuals', color='tab:purple')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)  # Add horizontal line at y=0
        ax.set_xlabel('Date')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals Plot')
        ax.legend()
        
        # Apply ADF test to residuals
        adf_result = adfuller(residuals)
        print('ADF Statistic:', adf_result[0])
        print('p-value:', adf_result[1])
        print('Critical Values:', adf_result[4])
        print('Is the series stationary?', 'No' if adf_result[1] > 0.05 else 'Yes')
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing {forex_pair} and {dollar_index}: {e}")

# Usage
forex_pair = "USDSGD=X"  # US Dollar / Singapore Dollar
dollar_index = "DX-Y.NYB"
start_date = '2022-07-01'
end_date = '2023-12-31'

plot_residuals_with_adf_test(forex_pair, dollar_index, start_date, end_date)
