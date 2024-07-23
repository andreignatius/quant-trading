import yfinance as yf
import pandas as pd

# Agriculture commodities and their corresponding tickers
commodities = {
    'Corn': 'CORN', # Teucrim Corn Fund
    'Soybeans': 'SOYB', # Teucrim Soybean Fund
    'Wheat': 'WEAT', # Teucrim Wheat Fund
    
    # THESE GOT DELISTED
    # 'Sugar': 'SGG', # iPath Series B Bloomberg Sugar Subindex Total Return ETN
    # 'Coffee': 'JO', # Ipatha.B Coffee Subindex TR ETN
    # 'Cocoa': 'NIB', # iPath Bloomberg Cocoa Subindex Total Return(SM) ETN (NIB)
    # 'Cotton': 'BAL' # iPath Series B Bloomberg Cotton Subindex Total Return ETN
}

# Function to fetch data for a ticker and save to CSV
def fetch_and_save_data(ticker, filename):
    data = yf.download(ticker, start="2019-05-22", end="2024-05-22")
    data.to_csv(filename)

# Loop through commodities dictionary and fetch data for each ticker
for commodity, ticker in commodities.items():
    filename = f"{commodity}_data.csv"
    fetch_and_save_data(ticker, filename)
    print(f"Data for {commodity} saved to {filename}")
