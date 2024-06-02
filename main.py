import pandas as pd
import yfinance as yf
import csv

from training.logreg_model import LogRegModel
from trading.trading_strategy import TradingStrategy


def fetch_and_format_data(tickers, start_date, end_date):
    """Fetches and formats closing prices for given tickers from Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data fetched for {tickers}. Please check your tickers or date range.")

        # Assuming the data might have multi-level columns, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            for col in data.columns.values:
                print("col: ", col)
            data.columns = ['_'.join(col_tuple[0].split()) + '_' + col_tuple[1] for col_tuple in data.columns.values]
            print("data.columns: ", data.columns)

        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)  # Sort by Date
        print("111Data loaded: ", data.head())
        return data
    except Exception as e:
        print(f"Error fetching and formatting data for {tickers}: {e}")
        raise e


def rolling_window_train_predict(
    data, start_date, end_date, train_duration_months, test_duration_months, trading_instrument
):
    trade_logs = []
    final_portfolio_values = []
    interest_costs_total = []
    transaction_costs_total = []

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

        train_data = data[(data.index >= train_start) & (data.index < train_end)]
        test_data = data[(data.index >= test_start) & (data.index < test_end)]

        # Initialize and use the LogRegModel for this window
        print(
            "train_start: ",
            train_start,
            "train_end: ",
            train_end,
            "test_start: ",
            test_start,
            "test_end: ",
            test_end,
        )
        model = LogRegModel(
            "inputs/temp_data.csv", train_start, train_end, test_start, test_end, trading_instrument
        )
        model.load_preprocess_data()
        model.train_test_split_time_series()
        model.train()
        test_data = model.retrieve_test_set()

        # Instantiate the TradingStrategy class
        trading_strategy = TradingStrategy(model, test_data, trading_instrument)
        trading_strategy.execute_trades()
        trading_results = trading_strategy.evaluate_performance()

        trade_logs.append(trading_results["Trade Log"])
        final_portfolio_values.append(trading_results["Final Portfolio Value"])
        interest_costs_total.append(sum(trading_results["Interest Costs"]))
        transaction_costs_total.append(trading_results["Transaction Costs"])

        current_date += pd.DateOffset(months=test_duration_months)

    return (
        trade_logs,
        final_portfolio_values,
        interest_costs_total,
        transaction_costs_total,
    )


if __name__ == "__main__":
    tickers = [
        "BZ=F",
        "CL=F",
        "GC=F",
        "SI=F",
        "NG=F",
        "USDCAD=X",
        "USDNOK=X",
        "AUDUSD=X",
        "NZDUSD=X",
        "USDAUD=X",
        "USDZAR=X",
        "USDBRL=X",
    ]
    start_date = "2013-01-01"
    end_date = "2023-01-01"
    raw_data = fetch_and_format_data(tickers, start_date, end_date)

    raw_data.to_csv('inputs/temp_data.csv', index=True)

    trading_instrument = "USDBRL=X"

    trade_logs, final_values, interest_costs_total, transaction_costs_total = (
        rolling_window_train_predict(
            raw_data,
            start_date,
            end_date,
            12,
            6,  # 12 months training, 6 months testing
            trading_instrument, # USDBRL testing
        )
    )
    
    # Specify the CSV file name
    filename = "trade_log.csv"

    # Open the file in write mode
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow('trade_type,lcy_bought,lcy,rate,date,leverage,comment'.split(","))
        # Write each row of data
        for trade_period in trade_logs:
            for trade in trade_period:
                print("trade: ", trade)
                writer.writerow(trade.split(','))

    print("Final trade logs:", trade_logs)
    print("Final portfolio values:", final_values)
    print("Interest costs:", interest_costs_total)
    print("Transaction costs:", transaction_costs_total)
