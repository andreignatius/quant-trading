import yfinance as yf
import pandas as pd
import math
import itertools
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt

# Forex pairs
forex_pairs = [
    "JPYUSD=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X", 
    "CADUSD=X", "CHFUSD=X", "SGDUSD=X", "MXNUSD=X", "ZARUSD=X", 
    "SEKUSD=X", "HKDUSD=X", "NOKUSD=X", "TRYUSD=X", "INRUSD=X", 
    "KRWUSD=X", "PHPUSD=X", "THBUSD=X", "IDRUSD=X"
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

# Prepare train data
train_end = math.floor(train_ratio * len(df_together))
training = df_together.iloc[:train_end]

# Initialize DataFrames to store cointegration test results and p-values
cointegration_results = pd.DataFrame(index=forex_pairs, columns=forex_pairs)
p_values = pd.DataFrame(index=forex_pairs, columns=forex_pairs)

# Cointegration test
for pair1, pair2 in itertools.combinations(forex_pairs, 2):
    coint_result = ts.coint(training[pair1], training[pair2])
    cointegration_results.loc[pair1, pair2] = coint_result[0]
    cointegration_results.loc[pair2, pair1] = coint_result[0]
    p_values.loc[pair1, pair2] = coint_result[1]
    p_values.loc[pair2, pair1] = coint_result[1]

# Flatten the DataFrame and sort by p-values
flattened_results = p_values.stack().reset_index()
flattened_results.columns = ['Pair1', 'Pair2', 'P-Value']
flattened_results['Cointegration Score'] = flattened_results.apply(
    lambda row: cointegration_results.loc[row['Pair1'], row['Pair2']], axis=1
)

# Remove duplicates by ensuring Pair1 < Pair2
flattened_results['Ordered Pair'] = flattened_results.apply(
    lambda row: tuple(sorted([row['Pair1'], row['Pair2']])), axis=1
)
flattened_results = flattened_results.drop_duplicates(subset=['Ordered Pair']).drop(columns=['Ordered Pair'])

# Round p-values and cointegration scores
flattened_results['P-Value'] = flattened_results['P-Value'].astype(float).round(5)
flattened_results['Cointegration Score'] = flattened_results['Cointegration Score'].astype(float).round(3)

# Get top 3 results by p-value
top_3_results = flattened_results.sort_values('P-Value').head(3)

# Extract the top 3 pairs
top_pairs = top_3_results[['Pair1', 'Pair2']].values.flatten()
top_pairs = pd.unique(top_pairs)

# Standard scale the FX rates
standard_scaled_data = df_together[top_pairs].apply(lambda x: (x - x.mean()) / x.std())

# Plot the standard scaled FX data for the top pairs
plt.figure(figsize=(14, 7))

for pair in top_pairs:
    plt.plot(standard_scaled_data[pair], label=pair)

plt.xlabel('Date')
plt.ylabel('Standard Scaled Exchange Rate')
plt.legend()
plt.savefig('X6_fx_eda_standard_scaled.png')
plt.show()

