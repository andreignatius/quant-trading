import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.signal import find_peaks
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
import numpy as np

def fetch_data(tickers, start_date, end_date):
    """Fetches closing prices for given tickers from Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        return data['Close']
    except Exception as e:
        print(f"Error fetching data for {tickers}: {e}")
        return pd.DataFrame()

def plot_data(data):
    """Plots normalized price trends."""
    plt.figure(figsize=(14, 7))
    max_values = data.max()
    for column in data.columns:
        normalized_series = data[column] / max_values[column]
        plt.plot(data.index, normalized_series, label=f"{column} (scaled by 1/{max_values[column]:.2f})")
    plt.title('Normalized Price Trends')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend(title='Legend')
    plt.grid(True)
    plt.show()

def seasonal_analysis(data, ticker):
    """Performs and plots seasonal analysis on selected ticker data."""
    if ticker in data.columns:
        monthly_data = data[ticker].resample('M').mean()
        sns.lineplot(x=monthly_data.index, y=monthly_data)
        plt.title(f'Seasonal Trends in {ticker} Prices')
        plt.show()

def harmonic_analysis(data, ticker):
    """Identifies and plots peak pricing events for the specified ticker."""
    if ticker in data.columns:
        series_data = data[ticker].dropna()
        height = np.std(series_data) * 0.75  # Adjusted peak height
        peaks, _ = find_peaks(series_data, height=height)
        plt.figure(figsize=(10, 4))
        plt.plot(series_data, label=f'{ticker} Prices')
        plt.plot(series_data.index[peaks], series_data.iloc[peaks], "x", label='Peaks')
        plt.title(f'Peak Pricing Events in {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

def topological_data_analysis(data, ticker):
    """Computes and plots a persistence diagram for specified ticker."""
    if ticker in data.columns:
        clean_data = data[ticker].dropna()
        if clean_data.empty:
            print("No data available for analysis after removing NaN values.")
            return
        tda = VietorisRipsPersistence(homology_dimensions=[0, 1], max_edge_length=2)
        point_cloud = clean_data.to_numpy()
        point_cloud_reshaped = point_cloud.reshape(1, -1, 1)
        diagram = tda.fit_transform(point_cloud_reshaped)
        plt.figure(figsize=(5, 5))
        plot_diagram(diagram[0])
        plt.title('Persistence Diagram for WTI Crude')
        plt.xlabel('Birth')
        plt.ylabel('Death')
        plt.xlim([0, 2])
        plt.ylim([0, 2])
        plt.show()

def correlation_analysis(data):
    """Analyzes and plots correlation between commodities and currencies."""
    plt.figure(figsize=(10, 6))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Commodities and Currencies')
    plt.show()

def cointegration_analysis(data, asset1, asset2):
    """Tests for cointegration between two assets using the Engle-Granger method."""
    try:
        x = data[asset1].dropna()  # Clean asset1 data
        y = data.loc[x.index, asset2].dropna()  # Align and clean asset2 data

        x, y = x.align(y, join='inner')  # Ensuring both series have the same dates

        x = sm.add_constant(x)
        result = sm.OLS(y, x).fit()
        residuals = result.resid
        adf_test = adfuller(residuals)
        print(f'ADF Statistic: {adf_test[0]}')
        print(f'p-value: {adf_test[1]}')
        if adf_test[1] < 0.05:
            print(f"{asset1} and {asset2} are cointegrated.")
        else:
            print(f"{asset1} and {asset2} are not cointegrated.")
    except Exception as e:
        print(f"Error in cointegration analysis between {asset1} and {asset2}: {e}")

if __name__ == "__main__":
    # tickers = ['BZ=F', 'CL=F', 'USDCAD=X', 'USDNOK=X']
    tickers = [ 'BZ=F', 'CL=F', 'GC=F', 'SI=F', 'NG=F', \
                'USDCAD=X', 'USDNOK=X', 'AUDUSD=X', 'NZDUSD=X', 'USDAUD=X', 'USDZAR=X', 'USDBRL=X']
    start_date = '2013-01-01'
    end_date = '2023-01-01'
    data = fetch_data(tickers, start_date, end_date)
    if not data.empty:
        plot_data(data)
        seasonal_analysis(data, 'CL=F')
        harmonic_analysis(data, 'CL=F')
        topological_data_analysis(data, 'CL=F')
        correlation_analysis(data)
        cointegration_analysis(data, 'BZ=F', 'CL=F')  # Example cointegration test
    else:
        print("Data retrieval was unsuccessful.")
