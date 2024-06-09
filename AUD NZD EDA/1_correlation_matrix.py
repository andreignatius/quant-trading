import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_correlation_matrix(forex_pairs, start_date, end_date, interval='1d'):
    # Create a DataFrame to store the adjusted close prices of all forex pairs
    data = pd.DataFrame()

    # Download the adjusted close prices for each forex pair
    for forex in forex_pairs:
        try:
            data[forex] = yf.Ticker(forex).history(start=start_date, end=end_date, interval=interval)["Close"]
        except Exception as e:
            print(f"Error downloading data for {forex}: {e}")
            data[forex] = pd.Series(dtype='float64')

    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    return correlation_matrix

# Usage
forex_pairs = [
    "USDJPY=X",  # US Dollar / Japanese Yen
    "USDEUR=X",  # US DOllar / Euro
    "USDGBP=X",  # US Dollar / British Pound
    "USDAUD=X",  # US Dollar / Australian Dollar
    "USDNZD=X",  # US Dollar / New Zealand Dollar
    "USDCAD=X",  # US Dollar / Canadian Dollar
    "USDCHF=X",  # US Dollar / Swiss Franc
    "USDSGD=X",  # US Dollar / Singapore Dollar
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

start_date = '2016-01-01'
end_date = '2023-12-31'
correlation_matrix = generate_correlation_matrix(forex_pairs, start_date, end_date)

print("Correlation Values:")
print(correlation_matrix)

# Replace NaN values with 0
correlation_matrix.fillna(0, inplace=True)

# Create a mask to display only the lower triangle of the matrix
# Mask values above the diagonal
mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

# Optional: Replace diagonal values with NaN
np.fill_diagonal(correlation_matrix.values, np.nan)

# Generate Heatmap
plt.figure(figsize=(13, 12))  # Adjust figure size

# Correlation values heatmap
heatmap = sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlGn', fmt=".2f", center=0, cbar_kws={'label': 'Correlation Value'}, annot_kws={"size": 8})  # Adjust annotation font size

# Rotate the y-axis labels for better visibility and adjust layout
plt.yticks(rotation=0)
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

plt.title('Correlation Values Between Forex Pairs')

plt.savefig('X1_correlation_matrix.png')
plt.tight_layout()
plt.show()
