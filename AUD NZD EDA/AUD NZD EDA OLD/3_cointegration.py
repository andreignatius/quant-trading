import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import coint

def generate_cointegration_matrix(forex_pairs, start_date, end_date, interval='1d'):
    # Create a DataFrame to store the adjusted close prices of all forex pairs
    data = pd.DataFrame()

    # Download the adjusted close prices for each forex pair
    for forex in forex_pairs:
        try:
            data[forex] = yf.Ticker(forex).history(start=start_date, end=end_date, interval=interval)["Close"]
        except Exception as e:
            print(f"Error downloading data for {forex}: {e}")
            data[forex] = pd.Series(dtype='float64')

    # Replace infinite or missing values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Calculate cointegration and p-values
    cointegration_matrix = pd.DataFrame(index=forex_pairs, columns=forex_pairs)
    p_values = pd.DataFrame(index=forex_pairs, columns=forex_pairs)
    
    for i in range(len(forex_pairs)):
        for j in range(i+1, len(forex_pairs)):
            result = coint(data[forex_pairs[i]], data[forex_pairs[j]])
            cointegration_matrix.loc[forex_pairs[i], forex_pairs[j]] = result[0]
            p_values.loc[forex_pairs[i], forex_pairs[j]] = result[1]

    return cointegration_matrix, p_values


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
cointegration_matrix, p_values = generate_cointegration_matrix(forex_pairs, start_date, end_date)

# Generate Heatmap for cointegration values
plt.figure(figsize=(13, 12))  # Adjust figure size
plt.title('Cointegration Values Between Forex Pairs')
heatmap_coint = sns.heatmap(cointegration_matrix.astype(float), annot=True, cmap='RdYlGn', fmt=".2f", center=0, cbar_kws={'label': 'Cointegration Value'}, annot_kws={"size": 8})
plt.tight_layout()
plt.savefig('X3_cointegration_matrix.png')
plt.show()

# Generate Heatmap for p-values
plt.figure(figsize=(13, 12))  # Adjust figure size
plt.title('P-Values Between Forex Pairs')
heatmap_p_values = sns.heatmap(p_values.astype(float), annot=True, cmap='RdYlGn_r', fmt=".2f", center=0.05, cbar_kws={'label': 'P-Value'}, annot_kws={"size": 8})
plt.tight_layout()
plt.savefig('X3_p_values_heatmap.png')
plt.show()
