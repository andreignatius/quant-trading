from pairs import PairsTrader

if __name__ == "__main__":
    start_date = "2023-01-01"
    end_date = "2024-05-31"

    # 1. initialize the strat
    # this is example for AUDUSD and NZDUSD pairs trading !
    # interval is 1h
    # train ratio is 80% of data from start date is for training, remaining 20% for out of sample
    myPairsTrader = PairsTrader(
        start_date,
        end_date,
        item1="AUDUSD=X",
        item2="NZDUSD=X",
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
