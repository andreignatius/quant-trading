import pandas as pd
import yfinance as yf
import csv
import math
from training.logreg_model import LogRegModel
from trading.trading_strategy import TradingStrategy


def fetch_and_format_data(tickers, start_date, end_date):
    """Fetches and formats closing prices for given tickers from Yahoo Finance."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(
                f"No data fetched for {tickers}. Please check your tickers or date range."
            )

        # Assuming the data might have multi-level columns, flatten them
        if isinstance(data.columns, pd.MultiIndex):
            for col in data.columns.values:
                print("col: ", col)
            data.columns = [
                "_".join(col_tuple[0].split()) + "_" + col_tuple[1]
                for col_tuple in data.columns.values
            ]
            print("data.columns: ", data.columns)

        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)  # Sort by Date
        print("111Data loaded: ", data.head())
        return data
    except Exception as e:
        print(f"Error fetching and formatting data for {tickers}: {e}")
        raise e


def rolling_window_train_predict(
    data,
    start_date,
    end_date,
    train_ratio,
    trading_instrument,
):
    trade_logs = []
    final_portfolio_values = []
    interest_costs_total = []
    transaction_costs_total = []

    start_date = pd.to_datetime(start_date)
    current_date = start_date

    while current_date < pd.to_datetime(end_date):
        print("Current period starting:", current_date)

        # prepare data for train and out of sample
        train_start = 0
        train_end = math.floor(train_ratio * len(data))
        oos_start = train_end + 1
        oos_end = len(data) - 1

        # yea these are it
        _splitidx = data.iloc[train_end].name
        _training = data[train_start:train_end]
        _oos = data[oos_start:oos_end]

        # Initialize and use the LogRegModel for this window
        print(
            "train_start: ",
            train_start,
            "train_end: ",
            train_end,
            "test_start: ",
            oos_start,
            "test_end: ",
            oos_end,
        )
        model = LogRegModel(
            "rawr.csv",
            train_start,
            train_end,
            oos_start,
            oos_end,
            trading_instrument,
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

        # current_date += pd.DateOffset(months=test_duration_months)

    return (
        trade_logs,
        final_portfolio_values,
        interest_costs_total,
        transaction_costs_total,
    )


if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2024-05-31"
    _item1 = "USDSGD=X"
    _item2 = "DX-Y.NYB"
    interval = "1h"
    train_ratio = 0.8

    raw_data = pd.read_csv("traderesult.csv")
    raw_data = raw_data[["Datetime", "spread"]]
    raw_data.columns = ["Date", "spread"]
    print(raw_data)  # datetime not as index because
    raw_data.to_csv("rawr.csv", index=False)

    trade_logs, final_values, interest_costs_total, transaction_costs_total = (
        rolling_window_train_predict(
            raw_data,
            start_date,
            end_date,
            train_ratio,  # 12 months training, 6 months testing
            "xoxox",  # USDBRL testing
        )
    )

    # Specify the CSV file name
    filename = "trade_log.csv"

    # Open the file in write mode
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            "trade_type,lcy_bought,lcy,rate,date,leverage,comment".split(",")
        )
        # Write each row of data
        for trade_period in trade_logs:
            for trade in trade_period:
                print("trade: ", trade)
                writer.writerow(trade.split(","))

    print("Final trade logs:", trade_logs)
    print("Final portfolio values:", final_values)
    print("Interest costs:", interest_costs_total)
    print("Transaction costs:", transaction_costs_total)
