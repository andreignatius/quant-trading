import yfinance as yf
import pandas as pd
import math
import itertools
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import numpy as np

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
flattened_results['P-Value'] = flattened_results['P-Value'].map(lambda x: f"{float(x):.5f}")
flattened_results['Cointegration Score'] = flattened_results['Cointegration Score'].map(lambda x: f"{float(x):.3f}")

# Get top 10 results by p-value
top_10_results = flattened_results.sort_values('P-Value').head(10)

# Create table of top 10 pairs
plt.figure(figsize=(7, 6))
table = plt.table(cellText=top_10_results.values,
                  colLabels=top_10_results.columns,
                  loc='center',
                  cellLoc='center',
                  colColours=['darkblue']*len(top_10_results.columns),
                  cellColours=[['white']*len(top_10_results.columns)]*len(top_10_results),
                  )
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_text_props(color='white')
plt.axis('off')

# Align text vertically to center
for key, cell in table._cells.items():
    cell._text.set_verticalalignment('center')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

# Save the table as an image
plt.savefig('X4_Cointegration_table.png')
plt.show()
