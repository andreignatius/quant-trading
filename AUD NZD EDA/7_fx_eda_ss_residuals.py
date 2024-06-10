import yfinance as yf
import pandas as pd
import math
import itertools
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# Split data into train and test sets
train_end = math.floor(train_ratio * len(df_together))
train_data = df_together.iloc[:train_end]
test_data = df_together.iloc[train_end:]

# Standard scale the train and test data
scaler = StandardScaler()
scaled_train = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns, index=train_data.index)
scaled_test = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns, index=test_data.index)

# Initialize DataFrames to store cointegration test results and p-values
p_values = pd.DataFrame(index=forex_pairs, columns=forex_pairs)

# Cointegration test using train data
for pair1, pair2 in itertools.combinations(forex_pairs, 2):
    coint_result = ts.coint(train_data[pair1], train_data[pair2])
    p_values.loc[pair1, pair2] = coint_result[1]
    p_values.loc[pair2, pair1] = coint_result[1]

# Flatten the DataFrame and sort by p-values
flattened_results = p_values.stack().reset_index()
flattened_results.columns = ['Pair1', 'Pair2', 'P-Value']

# Get top 3 results by p-value
top_3_results = flattened_results.sort_values('P-Value').head(3)

# Calculate differences for the top 3 pairs using scaled data
differences = {}
means = {}
adf_scores = {}
for idx, row in top_3_results.iterrows():
    pair1, pair2 = row['Pair1'], row['Pair2']
    difference = scaled_train[pair1] - scaled_train[pair2]
    differences[f"{pair1} - {pair2}"] = difference
    means[pair1] = difference.mean()
    means[pair2] = 0
    adf_result = ts.adfuller(difference.dropna())
    adf_scores[f"{pair1} - {pair2}"] = adf_result[0]

# Plot the differences for the top 3 pairs
plt.figure(figsize=(12, 8))
for i, (pair1, pair2) in enumerate(top_3_results[['Pair1', 'Pair2']].values):
    ax = plt.subplot(3, 1, i + 1)
    difference = scaled_train[pair1] - scaled_train[pair2]
    plt.plot(difference, label=f"{pair1} - {pair2}", color='blue')
    plt.axhline(y=means[pair1], color='red', linestyle='--', label='Mean')
    plt.title(f"Difference between {pair1} and {pair2}")
    plt.xlabel("Date")
    plt.ylabel("Difference")
    
    # Annotate ADF score and p-value in the middle of the plot
    plt.text(0.5, 0.5, f"ADF Score: {adf_scores[f'{pair1} - {pair2}']:.2f}\n p-value: {p_values.loc[pair1, pair2]:.4f}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.legend()

plt.tight_layout()
plt.savefig('X7_fx_eda_ss_residuals.png')
plt.show()
