import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read CSV file
df = pd.read_csv('fx_rates.csv')

# Drop columns as needed, if comment out all means no columns dropped
df.drop(columns=[
    # 'ARSUSD=X'
    # 'BRLUSD=X',
    # 'CNYUSD=X',
    # 'RUBUSD=X',
    # 'INRUSD=X',
    ],inplace=True)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Plotting
plt.figure(figsize=(10, 6))
for currency in df.columns[1:]:
    plt.plot(df['Date'], df[currency], label=currency[:-3])  # Removing '=X' from currency name

# Format x-axis to show only years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.title('FX Rates')
plt.xlabel('Year')
plt.ylabel('Exchange Rate (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# print(df) # Print FX df if needed
