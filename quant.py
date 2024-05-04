import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from gtda.homology import VietorisRipsPersistence  # Corrected import
from gtda.plotting import plot_diagram  # Corrected import

# Fetch data from Yahoo Finance
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Close']

# Normalize data and plot with annotations in the legend
def plot_data(data):
    plt.figure(figsize=(14, 7))
    max_values = data.max()  # Get maximum value for each series to normalize
    for column in data.columns:
        normalized_series = data[column] / max_values[column]
        plt.plot(data.index, normalized_series, label=f"{column} (scaled by 1/{max_values[column]:.2f})")
    plt.title('Normalized Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend(title='Legend')
    plt.grid(True)
    plt.show()

# Seasonal Analysis
def seasonal_analysis(data):
    monthly_data = data.resample('M').mean()
    sns.lineplot(x=monthly_data.index, y=monthly_data['CL=F'])  # Example for WTI Crude
    plt.title('Seasonal Trends in WTI Crude Prices')
    plt.show()

# Harmonic Analysis
def harmonic_analysis(data):
    peaks, _ = find_peaks(data['CL=F'], height=0)
    plt.figure(figsize=(10, 4))
    plt.plot(data['CL=F'])
    plt.plot(peaks, data['CL=F'][peaks], "x")
    plt.title('Peak Pricing Events in WTI Crude Prices')
    plt.show()

# Topological Data Analysis
def topological_data_analysis(data):
    tda = VietorisRipsPersistence()
    point_cloud = data[['CL=F']].to_numpy()
    diagram = tda.fit_transform(point_cloud.reshape(1, -1, 1))
    plt.figure(figsize=(5, 5))
    plot_diagram(diagram[0])
    plt.title('Persistence Diagram for WTI Crude')
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Define the tickers and time period
    tickers = ['BZ=F', 'CL=F', 'USDCAD=X', 'USDNOK=X']
    start_date = '2013-01-01'
    end_date = '2023-01-01'

    # Fetch and plot data
    data = fetch_data(tickers, start_date, end_date)
    plot_data(data)

    # Run analyses
    seasonal_analysis(data)
    harmonic_analysis(data)
    topological_data_analysis(data)
