import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import math


class PairsTrader:
    def __init__(
        self, start_date, end_date, item1, item2, interval, train_ratio
    ) -> None:
        self._start_date = start_date
        self._end_date = end_date
        self._item1 = item1
        self._item2 = item2
        self._normport = [0, 0]
        # get the good stuff from yahoo finance
        self._item1data = yf.Ticker(item1).history(
            start=start_date, end=end_date, interval=interval
        )["Close"]
        self._item2data = yf.Ticker(item2).history(
            start=start_date, end=end_date, interval=interval
        )["Close"]

        # align indices
        if len(self._item2data) > len(self._item1data):
            myindex = self._item1data.index
        else:
            myindex = self._item2data.index

        # zip together data frame
        self._df_together = pd.DataFrame(
            zip(self._item1data, self._item2data),
            index=myindex,
            columns=[self._item1, self._item2],
        )

        # prepare return frame
        self._return_df = (self._df_together / self._df_together.shift(1)) - 1

        # prepare data for train and out of sample
        self.train_start = 0
        self.train_end = math.floor(train_ratio * len(self._df_together))
        self.oos_start = self.train_end + 1
        self.oos_end = len(self._df_together) - 1

        # yea these are it
        self._splitidx = self._df_together.iloc[self.train_end].name
        self._training = self._df_together[self.train_start : self.train_end]
        self._oos = self._df_together[self.oos_start : self.oos_end]

    def plotOrigData(self):
        fig, ax1 = plt.subplots(figsize=(10, 7))
        color = "tab:red"
        ax1.set_ylabel(self._item1, color=color)
        ax1.plot(self._item1data, color=color)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = "tab:blue"
        ax2.set_ylabel(self._item2, color=color)
        ax2.plot(self._item2data, color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def cointTest(self):
        # lets do it
        coint_result = ts.coint(
            self._training[self._item1], self._training[self._item2]
        )
        print("************ COINT TEST ON TRAINING SUBSET ************")
        print(f"test statistic = {coint_result[0]}")
        print(
            f"crit values (from left to right 1%, 5%, 10% significance level)= {coint_result[2]}"
        )
        print("************ END COINT TEST ************")

    def cointRegression(self):
        y = self._training[self._item1]  # usdsgd
        x = self._training[self._item2]  # dxy
        result = sm.OLS(y.to_numpy(), sm.tools.add_constant(x.to_numpy())).fit()

        self.mu = result.params[0]
        self.gamma = result.params[1]
        # print model vs actual
        self._training["spread"] = y - self.gamma * x
        model = self.gamma * x + self.mu

        # spread, idk why prof ben coded it like this
        y_all = self._df_together[self._item1]  # usdsgd
        x_all = self._df_together[self._item2]  # dxy

        result_adf = ts.adfuller(self._training["spread"])
        print("************ ADF TEST ON TRAINING SUBSET ************")
        print(f"test statistic {result_adf[0]}")
        print(f"p value {result_adf[1]}")
        print(f"critical values for test statistic {result_adf[4]}")
        print("************ END ADF TEST ************")

        self._df_together["spread"] = y_all - self.gamma * x_all
        plt.axhline(np.mean(y - self.gamma * x), color="green")
        plt.plot(self._df_together["spread"])
        plt.axvline(self._splitidx, color="red")
        plt.show()

    def constructPortfolio(self):
        """
        Refer to this !
        https://palomar.home.ece.ust.hk/MAFS5310_lectures/Rsession_pairs_trading_with_R.html

        thresh_list is z score threshold to short/long
        """
        self._normport = [1, -self.gamma] / (1 + self.gamma)
        self._df_together_weighted = (
            self._normport * self._df_together[[self._item1, self._item2]]
        )
        self._df_together_weighted["spread"] = (
            self._df_together_weighted[self._item1]
            + self._df_together_weighted[self._item2]
        )

        self.weight_train_mean = np.mean(
            self._df_together_weighted[self.train_start : self.train_end]["spread"]
        )
        self.weight_train_std = np.std(
            self._df_together_weighted[self.train_start : self.train_end]["spread"],
            ddof=1,
        )

    def generateSignal(self, thresh_list):

        thresh_short = thresh_list[1]
        thresh_long = thresh_list[0]
        print(f"short thresh {thresh_short} ; long thresh {thresh_long}")

        # construct Z score to build port
        df_z = (
            self._df_together_weighted["spread"] - self.weight_train_mean
        ) / self.weight_train_std
        df_z = df_z.to_frame()
        df_z.reset_index(inplace=True)

        plt.axhline(0, color="green")
        plt.axhline(thresh_short, color="pink", linestyle="dashed")
        plt.axhline(thresh_long, color="pink", linestyle="dashed")

        # plt.plot(df_z["Datetime"], df_z["spread"])

        plt.axvline(self._splitidx, color="red")

        # now actually signal
        df_z["position"] = np.nan
        df_z.head(10)
        # initial position
        df_z["position"].iloc[0] = 0
        if df_z["spread"].iloc[0] < thresh_long:
            df_z["position"].iloc[0] = 1
        elif df_z["spread"].iloc[0] >= thresh_short:
            df_z["position"].iloc[0] = -1

        for idx, row in df_z.iterrows():
            if idx == 0:
                print("first row do nothing")
            else:
                if df_z["position"].loc[idx - 1] == 0:  # no position
                    if df_z["spread"].loc[idx] < thresh_long:
                        df_z["position"].loc[idx] = 1
                    elif df_z["spread"].loc[idx] > thresh_short:
                        df_z["position"].loc[idx] = -1
                    else:
                        df_z["position"].loc[idx] = df_z["position"].loc[idx - 1]
                elif df_z["position"].loc[idx - 1] == 1:  # ure long
                    if df_z["spread"].loc[idx] > 0:
                        df_z["position"].loc[idx] = 0  # close it
                    else:
                        df_z["position"].loc[idx] = df_z["position"].loc[idx - 1]
                elif df_z["position"].loc[idx - 1] == -1:  # ure shorte
                    if df_z["spread"].loc[idx] < 0:
                        df_z["position"].loc[idx] = 0  # close it
                    else:
                        df_z["position"].loc[idx] = df_z["position"].loc[idx - 1]

        df_z["position"] = df_z["position"].shift(1)
        plt.plot(df_z["Datetime"], df_z["spread"])
        plt.plot(df_z["Datetime"], df_z["position"], color="orange")
        plt.grid(alpha=0.5)
        plt.title("Generated signal -1 = SHORT pos, 1 = LONG pos, 0 = NO pos")
        plt.show()

        # PnL calcs
        self._df_together_weighted["return"] = self._df_together_weighted[
            "spread"
        ].diff()
        self._df_together_weighted["position"] = df_z["position"].to_numpy()
        self._df_together_weighted["traded_return"] = self._df_together_weighted[
            "return"
        ] * self._df_together_weighted["position"].shift(1)
        self._df_together_weighted["traded_return_cumsum"] = self._df_together_weighted[
            "traded_return"
        ].cumsum()
        self.index_for_pnl = df_z["Datetime"]

        self._df_together_weighted[f"{self._item1}_ratio"] = (
            self._df_together_weighted["position"] * self._normport[0]
        )
        self._df_together_weighted[f"{self._item2}_ratio"] = (
            self._df_together_weighted["position"] * self._normport[1]
        )

        # this returns the final DataFrame that also contains trade logs
        return self._df_together_weighted

    def plotCumulativePnL(self):
        plt.title(f"Cumulative PnL from {self._item1} & {self._item2} Pairs Trade")
        plt.grid(alpha=0.5)
        plt.plot(self.index_for_pnl, self._df_together_weighted["traded_return_cumsum"])
        plt.axvline(self._splitidx, color="red")
        plt.xticks(rotation=45)
        plt.show()


if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2024-05-31"

    # 1. initialize the strat
    # this is example using USDSGD and DXY USD index !
    # interval is 1h
    # train ratio is 80% of data from start date is for training, remaining 20% for out of sample
    myPairsTrader = PairsTrader(
        start_date,
        end_date,
        item1="USDSGD=X",
        item2="DX-Y.NYB",
        interval="1h",
        train_ratio=0.8,
    )

    # 2. plot the original data to do some qualitative check
    myPairsTrader.plotOrigData()

    # 3. test the pairs you provided during init, are they coint?
    myPairsTrader.cointTest()

    # 4. run the regression to find the spread
    myPairsTrader.cointRegression()

    # 5. spread from actual and regression result should be stationary
    myPairsTrader.constructPortfolio()

    # 6. if step 3 and 5 ok, then we can use it to start generating signals
    # this threshold is used when to enter Short / Long position w.r.t to normalized z
    # e.g. [-1,1].
    #       a) The strategy will enter SHORT if the Z value of the spread is +1 std dev from mean = 0
    #       b) The strategy will enter LONG if the Z value of the spread is -1 std dev from mean = 0
    #       c) If the spread Z value crosses 0, it will close any long / short positions
    #       d) note you can only do either a long, a short or nothing at any given time.
    #
    # result of trades are stored in myTradeResult,
    # PLEASE USE self._splitidx  to get the index where you split the training and test subset
    myZThreshold = [-1, 1]
    myTradeResult = myPairsTrader.generateSignal(myZThreshold)

    # 7. plot cumul PnL
    myPairsTrader.plotCumulativePnL()

    # NOTE THAT THIS IS POSITION SIGNAL, NOT BUY / SELL SIGNAL !
    print(myTradeResult)

    myTradeResult.to_csv("traderesult.csv")
