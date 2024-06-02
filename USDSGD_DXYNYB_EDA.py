import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_cointegration_matrix(forex_pairs, dollar_indexes, start_date, end_date, interval='1h', alpha=0.05):
    # Initialize a DataFrame to store the cointegration test scores and p-values
    cointegration_results = pd.DataFrame(index=forex_pairs, columns=dollar_indexes)
    p_values = pd.DataFrame(index=forex_pairs, columns=dollar_indexes)
    
    # Loop through each combination of forex pairs and dollar indexes
    for forex in forex_pairs:
        for index in dollar_indexes:
            try:
                # Download the adjusted close prices for the given tickers with hourly interval
                data1 = yf.Ticker(forex).history(start=start_date, end=end_date, interval=interval)["Close"]
                data2 = yf.Ticker(index).history(start=start_date, end=end_date, interval=interval)["Close"]
                
                # Align the indices of the two data series
                data = data1.to_frame(name=forex).join(data2.to_frame(name=index), how='inner')
                
                # Calculate the cointegration between the two tickers
                series1 = data[forex]
                series2 = data[index]
                
                # Perform the cointegration test
                score, p_value, _ = coint(series1, series2)
                
                # Store the results in the DataFrames
                cointegration_results.loc[forex, index] = score
                p_values.loc[forex, index] = p_value
            except Exception as e:
                # Handle any exceptions (e.g., due to missing data) and set the results to NaN
                cointegration_results.loc[forex, index] = np.nan
                p_values.loc[forex, index] = np.nan
                print(f"Error processing {forex} and {index}: {e}")
    
    return cointegration_results, p_values

# Usage
forex_pairs = [
    "USDJPY=X", 
    "EURUSD=X", 
    "USDGBP=X", 
    "USDAUD=X", 
    "USDCAD=X", 
    "USDCHF=X",
    "USDSGD=X",
    "USDNZD=X", 
    "USDMXN=X", 
    "USDZAR=X",
    "USDSEK=X",
]

dollar_indexes = ["DX-Y.NYB"]
start_date = '2022-07-01'
end_date = '2023-12-31'
cointegration_matrix, p_value_matrix = generate_cointegration_matrix(forex_pairs, dollar_indexes, start_date, end_date)

print("Cointegration Test Scores:")
print(cointegration_matrix)
print("\nP-Values:")
print(p_value_matrix)

# Replace NaN values with a large number (e.g., 1) for p-values and a neutral number (e.g., 0) for scores
cointegration_matrix = cointegration_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
p_value_matrix = p_value_matrix.apply(pd.to_numeric, errors='coerce').fillna(1)

# Generate Heatmaps
plt.figure(figsize=(10, 8))

# Cointegration scores heatmap
plt.subplot(2, 1, 1)
sns.heatmap(cointegration_matrix, annot=True, cmap='RdYlGn_r', center=0,\
            cbar_kws={'label': 'Cointegration Score'})
plt.title('Cointegration Test Scores')

# P-values heatmap
plt.subplot(2, 1, 2)
sns.heatmap(p_value_matrix, annot=True, cmap='RdYlGn_r', center=1,\
            cbar_kws={'label': 'P-Value'})
plt.title('P-Values')

plt.tight_layout()
plt.show()
