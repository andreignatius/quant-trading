import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
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

def rolling_window_train_predict(data, start_date, end_date, train_duration_months, test_duration_months):
    trade_logs = []
    final_portfolio_values = []
    interest_costs_total = []
    transaction_costs_total = []

    # Convert 'Date' column to datetime if it's not already
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)  # Sort by Date

    start_date = pd.to_datetime(start_date)
    current_date = start_date

    while current_date < pd.to_datetime(end_date):
        print("Current period starting:", current_date)

        train_start = current_date
        train_end = train_start + pd.DateOffset(months=train_duration_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_duration_months)

        if test_end > pd.to_datetime(end_date):
            break

        # Filter data for current training and testing periods
        train_data = data[(data.index >= train_start) & (data.index < train_end)]
        test_data = data[(data.index >= test_start) & (data.index < test_end)]

        if train_data.empty or test_data.empty:
            print("No data available for training or testing for the period starting:", current_date)
            current_date += pd.DateOffset(months=test_duration_months)
            continue

        # Initialize and use the LogRegModel for this window
        print("train_start: ", train_start, "train_end: ", train_end, "test_start: ", test_start, "test_end: ", test_end)
        model = LogRegModel('temp_data.csv', train_start, train_end, test_start, test_end)
        model.load_preprocess_data()  # Load and preprocess the data
        model.train_test_split_time_series()  # Split data into training and testing
        model.train()
        test_data = model.retrieve_test_set()

        # Instantiate the TradingStrategy class
        trading_strategy = TradingStrategy(model, test_data)
        trading_strategy.execute_trades()
        trading_results = trading_strategy.evaluate_performance()

        # Collect results
        trade_logs.append(trading_results['Trade Log'])
        final_portfolio_values.append(trading_results['Final Portfolio Value'])
        interest_costs_total.append(sum(trading_results['Interest Costs']))
        transaction_costs_total.append(trading_results['Transaction Costs'])

        # Move to the next window
        current_date += pd.DateOffset(months=test_duration_months)

    return trade_logs, final_portfolio_values, interest_costs_total, transaction_costs_total

if __name__ == "__main__":
    tickers = ['BZ=F', 'CL=F', 'GC=F', 'SI=F', 'NG=F', 'USDCAD=X', 'USDNOK=X', 'AUDUSD=X', 'NZDUSD=X', 'USDAUD=X', 'USDZAR=X', 'USDBRL=X']
    start_date = '2013-01-01'
    end_date = '2023-01-01'
    raw_data = fetch_data(tickers, start_date, end_date)

    if not raw_data.empty:
        trade_logs, final_values, interest_costs_total, transaction_costs_total = rolling_window_train_predict(
            raw_data, start_date, end_date, 12, 6  # 12 months training, 6 months testing
        )

        print("Final trade logs:", trade_logs)
        print("Final portfolio values:", final_values)
        print("Interest costs:", interest_costs_total)
        print("Transaction costs:", transaction_costs_total)
    else:
        print("Data retrieval was unsuccessful.")
