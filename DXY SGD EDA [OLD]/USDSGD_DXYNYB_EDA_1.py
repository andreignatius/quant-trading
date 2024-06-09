import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_correlation_matrix(forex_pairs, dollar_indexes, start_date, end_date, interval='1h'):
    # Initialize a DataFrame to store the correlation values
    correlation_results = pd.DataFrame(index=forex_pairs, columns=dollar_indexes)
    
    # Loop through each combination of forex pairs and dollar indexes
    for forex in forex_pairs:
        for index in dollar_indexes:
            try:
                # Download the adjusted close prices for the given tickers with hourly interval
                data1 = yf.Ticker(forex).history(start=start_date, end=end_date, interval=interval)["Close"]
                data2 = yf.Ticker(index).history(start=start_date, end=end_date, interval=interval)["Close"]
                
                # Align the indices of the two data series
                data = data1.to_frame(name=forex).join(data2.to_frame(name=index), how='inner')
                
                # Calculate the correlation between the two tickers
                correlation = data[forex].corr(data[index])
                
                # Store the results in the DataFrame
                correlation_results.loc[forex, index] = correlation
            except Exception as e:
                # Handle any exceptions (e.g., due to missing data) and set the results to NaN
                correlation_results.loc[forex, index] = np.nan
                print(f"Error processing {forex} and {index}: {e}")
    
    return correlation_results

# Usage
forex_pairs = [
    "USDJPY=X",  # US Dollar / Japanese Yen
    "USDEUR=X",  # Euro / US Dollar
    "USDGBP=X",  # US Dollar / British Pound
    "USDAUD=X",  # US Dollar / Australian Dollar
    "USDCAD=X",  # US Dollar / Canadian Dollar
    "USDCHF=X",  # US Dollar / Swiss Franc
    "USDSGD=X",  # US Dollar / Singapore Dollar
    "USDNZD=X",  # US Dollar / New Zealand Dollar
    "USDMXN=X",  # US Dollar / Mexican Peso
    "USDZAR=X",  # US Dollar / South African Rand
    "USDSEK=X",  # US Dollar / Swedish Krona
    "USDHKD=X",  # US Dollar / Hong Kong Dollar
    "USDNOK=X",  # US Dollar / Norwegian Krone
    "USDTRY=X",  # US Dollar / Turkish Lira
    "USDRUB=X",  # US Dollar / Russian Ruble
    "USDINR=X",  # US Dollar / Indian Rupee
    "USDKRW=X",  # US Dollar / South Korean Won
    "USDPHP=X",  # US Dollar / Philippine Peso
    "USDTHB=X",  # US Dollar / Thai Baht
    "USDMYR=X",  # US Dollar / Malaysian Ringgit
    "USDIDR=X",  # US Dollar / Indonesian Rupiah
]


dollar_indexes = ["DX-Y.NYB"]
start_date = '2022-07-01'
end_date = '2023-12-31'
correlation_matrix = generate_correlation_matrix(forex_pairs, dollar_indexes, start_date, end_date)

print("Correlation Values:")
print(correlation_matrix)

# Replace NaN values with a neutral number (e.g., 0) for correlation values
correlation_matrix = correlation_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

# Generate Heatmap
plt.figure(figsize=(8, 10))

# Correlation values heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0,\
            cbar_kws={'label': 'Correlation Value'})
plt.title('Correlation Values')

plt.tight_layout()
plt.show()
