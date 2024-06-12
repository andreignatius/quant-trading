import yfinance as yf
import matplotlib.pyplot as plt

# Function to download and plot data on different axes
def plot_forex_vs_index(forex_pair, dollar_index, start_date, end_date, interval='1h'):
    try:
        # Download the adjusted close prices for the given tickers with hourly interval
        forex_data = yf.Ticker(forex_pair).history(start=start_date, end=end_date, interval=interval)["Close"]
        index_data = yf.Ticker(dollar_index).history(start=start_date, end=end_date, interval=interval)["Close"]
        
        # Align the indices of the two data series
        data = forex_data.to_frame(name=forex_pair).join(index_data.to_frame(name=dollar_index), how='inner')
        
        # Create a figure and a set of subplots
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot the forex data on the primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(forex_pair, color=color)
        ax1.plot(data.index, data[forex_pair], label=forex_pair, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Instantiate a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(dollar_index, color=color)
        ax2.plot(data.index, data[dollar_index], label=dollar_index, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Add a title
        plt.title(f'{forex_pair} vs {dollar_index}')
        
        # Show the plot
        fig.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error processing {forex_pair} and {dollar_index}: {e}")

# Usage
forex_pair = "USDSGD=X"  # US Dollar / Singapore Dollar
dollar_index = "DX-Y.NYB"
start_date = '2022-07-01'
end_date = '2023-12-31'

plot_forex_vs_index(forex_pair, dollar_index, start_date, end_date)
