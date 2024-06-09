import yfinance as yf
import pandas as pd
import math
import itertools
import statsmodels.tsa.stattools as ts

# Parameters
start_date = "2016-01-01"
end_date = "2024-05-31"
currency_pairs = ["AUDUSD=X", "NZDUSD=X", "SGDUSD=X"]
train_ratio = 0.7
interval = "1d"

# Fetch data
currency_data = {}
for pair in currency_pairs:
    data = yf.Ticker(pair).history(start=start_date, end=end_date, interval=interval)["Close"]
    currency_data[pair] = data

# Align indices and remove NaN values
df_together = pd.concat(currency_data.values(), axis=1, keys=currency_data.keys()).dropna()

# Prepare train and out-of-sample data
train_end = math.floor(train_ratio * len(df_together))
training = df_together.iloc[:train_end]

# Cointegration test
print("************ COINTEGRATION TEST RESULTS ************")
for pair1, pair2 in itertools.combinations(currency_pairs, 2):
    coint_result = ts.coint(training[pair1], training[pair2])
    print(f"Pair: {pair1} - {pair2}")
    print(f"Test Statistic: {coint_result[0]}")
    print(f"Critical Values (1%, 5%, 10%): {coint_result[2]}")
    print("-----------------------------------------------------")
print("************ END OF COINTEGRATION TEST ************")
