import yfinance as yf
import pandas as pd
import math
import itertools
import statsmodels.tsa.stattools as ts
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Additional forex pairs
forex_pairs = [
    "JPYUSD=X",  # US Dollar / Japanese Yen
    "EURUSD=X",  # US Dollar / Euro
    "GBPUSD=X",  # US Dollar / British Pound
    "AUDUSD=X",  # US Dollar / Australian Dollar
    "NZDUSD=X",  # US Dollar / New Zealand Dollar
    "CADUSD=X",  # US Dollar / Canadian Dollar
    "CHFUSD=X",  # US Dollar / Swiss Franc
    "SGDUSD=X",  # US Dollar / Singapore Dollar
    "MXNUSD=X",  # US Dollar / Mexican Peso
    "ZARUSD=X",  # US Dollar / South African Rand
    "SEKUSD=X",  # US Dollar / Swedish Krona
    "HKDUSD=X",  # US Dollar / Hong Kong Dollar
    "NOKUSD=X",  # US Dollar / Norwegian Krone
    "TRYUSD=X",  # US Dollar / Turkish Lira
    "INRUSD=X",  # US Dollar / Indian Rupee
    "KRWUSD=X",  # US Dollar / South Korean Won
    "PHPUSD=X",  # US Dollar / Philippine Peso
    "THBUSD=X",  # US Dollar / Thai Baht
    "IDRUSD=X",  # US Dollar / Indonesian Rupiah
]

# Parameters
start_date = "2016-01-01"
end_date = "2024-05-31"
train_ratio = 0.7
interval = "1d"

# Fetch data
currency_data = {}
for pair in forex_pairs:
    data = yf.Ticker(pair).history(start=start_date, end=end_date, interval=interval)["Close"]
    currency_data[pair] = data

# Align indices and remove NaN values
df_together = pd.concat(currency_data.values(), axis=1, keys=currency_data.keys()).dropna()

# Prepare train and out-of-sample data
train_end = math.floor(train_ratio * len(df_together))
training = df_together.iloc[:train_end]

# Initialize DataFrame to store cointegration test results
cointegration_results = pd.DataFrame(index=forex_pairs, columns=forex_pairs)

# Cointegration test
for pair1, pair2 in itertools.combinations(forex_pairs, 2):
    coint_result = ts.coint(training[pair1], training[pair2])
    cointegration_results.loc[pair1, pair2] = coint_result[0]
    cointegration_results.loc[pair2, pair1] = coint_result[0]

# Generate a mask for the bottom triangle
mask = np.tril(np.ones_like(cointegration_results, dtype=bool))

# Plot and save heatmap for cointegration test results
plt.figure(figsize=(12, 8))
sns.heatmap(cointegration_results.astype(float), mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Cointegration Test Results")
plt.xlabel("Currency Pair 1")
plt.ylabel("Currency Pair 2")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('X3_cointegration_results.png')
plt.show()

# Initialize DataFrame to store p-values
p_values = pd.DataFrame(index=forex_pairs, columns=forex_pairs)

# Cointegration test for p-values
for pair1, pair2 in itertools.combinations(forex_pairs, 2):
    coint_result = ts.coint(training[pair1], training[pair2])
    p_values.loc[pair1, pair2] = coint_result[1]  # Storing p-values
    p_values.loc[pair2, pair1] = coint_result[1]  # Same value for symmetry

# Generate a mask for the bottom triangle
mask = np.tril(np.ones_like(p_values, dtype=bool))

# Plot and save heatmap for p-values
plt.figure(figsize=(12, 8))
sns.heatmap(p_values.astype(float), mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Cointegration Test P-Values")
plt.xlabel("Currency Pair 1")
plt.ylabel("Currency Pair 2")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.savefig('X3_cointegration_pvalues.png')
plt.show()
