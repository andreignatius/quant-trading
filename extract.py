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
# from base_model import BaseModel  # Assuming the BaseModel class is in a separate file named 'base_model.py'
from logreg_model import LogRegModel
from trading_strategy import TradingStrategy

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
    interest_costs_total = []
    transaction_costs_total = []
    final_portfolio_values = []
    trade_logs = []

    if not raw_data.empty:
        # Assume file_path is where you save the fetched data temporarily
        raw_data.to_csv('temp_data.csv')
        
        # Initialize and use the BaseModel for advanced analysis
        # model = BaseModel(file_path='temp_data.csv', train_start='2013-01-01', train_end='2018-01-01', test_start='2018-01-01', test_end='2023-01-01')
        model = LogRegModel(file_path='temp_data.csv', train_start='2013-01-01', train_end='2018-01-01', test_start='2018-01-01', test_end='2023-01-01')
        model.load_preprocess_data()  # Load and preprocess the data
        model.train_test_split_time_series()  # Split data into training and testing
        model.train()  # Placeholder for training method

        data = model.retrieve_test_set()
        # predicted_categories = model.predict()

        # Perform backtesting, log trades, calculate final portfolio value
        # ... [Your backtesting logic here] ...
        # Backtesting with stop-loss and take-profit
        # Instantiate the TradingStrategy class
        # trading_strategy = TradingStrategy(model, data, leverage_factor=leverage_factor, annual_interest_rate=annual_interest_rate)
        trading_strategy = TradingStrategy(model, data)
        # Run the trading strategy with the model's predictions
        trading_strategy.execute_trades()

        # Retrieve results and output
        trading_results = trading_strategy.evaluate_performance()

        trade_log = trading_results['Trade Log']
        final_portfolio_value = trading_results['Final Portfolio Value']
        pnl_per_trade = trading_results['Profit/Loss per Trade']
        interest_costs = sum(trading_results['Interest Costs'])
        transaction_costs = trading_results['Transaction Costs']
        print("interest_costs111: ", interest_costs)
        print("transaction_costs111: ", transaction_costs)

        interest_costs_total.append( interest_costs )
        transaction_costs_total.append( transaction_costs )

        # Output
        print(trade_log)
        print("num trades: ", len(trade_log))
        print(f"Final Portfolio Value Before Cost: {final_portfolio_value}")
        final_portfolio_value = final_portfolio_value - ( interest_costs + transaction_costs )
        print(f"Final Portfolio Value After Cost: {final_portfolio_value}")

        # pnl_per_trade = ( final_portfolio_value - starting_cash ) / len(trade_log)
        print("PnL per trade: ", pnl_per_trade)

        # Collect results
        trade_logs.append(trade_log)
        final_portfolio_values.append(final_portfolio_value)

        # # Use the processed data for further analysis
        # plot_data(model.data)  # Plot the data with additional features
        
        # # Additional analyses could be done here, such as correlation or cointegration on enhanced dataset
        # correlation_analysis(model.data[['Close'] + [col for col in model.data.columns if 'Kalman' in col or 'RSI' in col]])
    else:
        print("Data retrieval was unsuccessful.")
