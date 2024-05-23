import pandas as pd
import matplotlib.pyplot as plt

# Agriculture commodities and their corresponding tickers
commodities = {
    'Corn': 'CORN', # Teucrim Corn Fund
    'Soybeans': 'SOYB', # Teucrim Soybean Fund
    'Wheat': 'WEAT', # Teucrim Wheat Fund\

    # THESE GOT DELISTED
    # 'Sugar': 'SGG', # iPath Series B Bloomberg Sugar Subindex Total Return ETN
    # 'Coffee': 'JO', # Ipatha.B Coffee Subindex TR ETN
    # 'Cocoa': 'NIB', # iPath Bloomberg Cocoa Subindex Total Return(SM) ETN (NIB)
    # 'Cotton': 'BAL' # iPath Series B Bloomberg Cotton Subindex Total Return ETN
}

# Function to plot adjusted close prices for each commodity
def plot_adj_close(commodities):
    plt.figure(figsize=(12, 6))
    for commodity, ticker in commodities.items():
        filename = f"{commodity}_data.csv"
        data = pd.read_csv(filename, index_col='Date', parse_dates=True)
        plt.plot(data.index, data['Adj Close'], label=commodity)

    plt.title('Adjusted Close Prices of Commodities')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot adjusted close prices
plot_adj_close(commodities)
