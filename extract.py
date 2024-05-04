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
from base_model import BaseModel  # Assuming the BaseModel class is in a separate file named 'base_model.py'

def fetch_data(tickers, start_date, end_date):
    """Fetches closing prices for given tickers from Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data for {tickers}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    tickers = ['BZ=F', 'CL=F', 'GC=F', 'SI=F', 'NG=F', 'USDCAD=X', 'USDNOK=X', 'AUDUSD=X', 'NZDUSD=X', 'USDAUD=X', 'USDZAR=X', 'USDBRL=X']
    start_date = '2013-01-01'
    end_date = '2023-01-01'
    raw_data = fetch_data(tickers, start_date, end_date)

    if not raw_data.empty:
        # Assume file_path is where you save the fetched data temporarily
        raw_data.to_csv('temp_data.csv')
        
        # Initialize and use the BaseModel for advanced analysis
        model = BaseModel(file_path='temp_data.csv', train_start='2013-01-01', train_end='2018-01-01', test_start='2018-01-01', test_end='2023-01-01')
        model.load_preprocess_data()  # Load and preprocess the data
        model.train_test_split_time_series()  # Split data into training and testing
        model.train()  # Placeholder for training method
        test_set = model.retrieve_test_set()

        # # Use the processed data for further analysis
        # plot_data(model.data)  # Plot the data with additional features
        
        # # Additional analyses could be done here, such as correlation or cointegration on enhanced dataset
        # correlation_analysis(model.data[['Close'] + [col for col in model.data.columns if 'Kalman' in col or 'RSI' in col]])
    else:
        print("Data retrieval was unsuccessful.")
