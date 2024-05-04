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

# # Harmonic Analysis
# def harmonic_analysis(data):
#     peaks, _ = find_peaks(data['CL=F'], height=0)
#     plt.figure(figsize=(10, 4))
#     plt.plot(data['CL=F'])
#     plt.plot(peaks, data['CL=F'][peaks], "x")
#     plt.title('Peak Pricing Events in WTI Crude Prices')
#     plt.show()

# def harmonic_analysis(data):
#     data['CL=F'].dropna(inplace=True)  # Handle missing data
#     mean_price = data['CL=F'].mean()
#     std_price = data['CL=F'].std()
#     min_height = mean_price + std_price  # Dynamic height based on data statistics

#     peaks, _ = find_peaks(data['CL=F'], height=min_height, distance=20, prominence=1)
#     plt.figure(figsize=(10, 4))
#     plt.plot(data['CL=F'], label='WTI Crude Prices')
#     plt.plot(peaks, data['CL=F'][peaks], "x", label='Peaks')
#     plt.title('Peak Pricing Events in WTI Crude Prices')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# Harmonic Analysis
def harmonic_analysis(data):
    # Ensure data doesn't have missing values which can disrupt peak analysis
    data = data.dropna(subset=['CL=F'])
    
    # Calculate dynamic height if necessary, or use a static value
    # For illustration, we use a static height here; adjust as needed
    height = 0
    
    # Find peaks
    peaks, _ = find_peaks(data['CL=F'], height=height)

    # Plot the data
    plt.figure(figsize=(10, 4))
    plt.plot(data['CL=F'], label='WTI Crude Prices')  # Plot the crude prices

    # Plot the peaks: convert indices to datetime index for correct plotting
    plt.plot(data.index[peaks], data['CL=F'].iloc[peaks], "x", label='Peaks')  # Corrected peak plotting

    # Adding plot title and labels
    plt.title('Peak Pricing Events in WTI Crude Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def topological_data_analysis(data):
    clean_data = data[['CL=F']].dropna()
    if clean_data.empty:
        print("No data available for analysis after removing NaN values.")
        return

    tda = VietorisRipsPersistence(homology_dimensions=[0, 1], max_edge_length=2)
    point_cloud = clean_data.to_numpy()
    point_cloud_reshaped = point_cloud.reshape(1, -1, 1)
    diagram = tda.fit_transform(point_cloud_reshaped)

    print(diagram)  # To see if there are any non-trivial topological features

    plt.figure(figsize=(5, 5))
    plot_diagram(diagram[0])
    plt.title('Persistence Diagram for WTI Crude')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.xlim([0, 2])  # Set limits based on your data
    plt.ylim([0, 2])
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
