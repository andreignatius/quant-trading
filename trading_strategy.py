
class TradingStrategy:

    def __init__(self, model, data, start_cash=10000, trading_lot=7500, stop_loss_threshold=0.05, leverage_factor=4, margin_call_threshold=0.5, annual_interest_rate=0.03):
        self.model = model
        self.data = data
        self.cash = start_cash
        self.margin_requirement = start_cash * margin_call_threshold
        self.starting_cash = start_cash
        self.trading_lot = trading_lot
        self.stop_loss_threshold = stop_loss_threshold
        self.leverage_factor = leverage_factor
        self.trade_log = []
        self.buy_price = None
        self.brl_inventory = 0
        self.annual_interest_rate = annual_interest_rate
        self.daily_return_factors = []
        self.interest_costs = []

    # def execute_trades(self):
    #     predicted_categories = self.model.predict()

    #     print("check predicted_categories: ", predicted_categories)

    #     for index, (row, prediction) in enumerate(zip(self.data.iterrows(), predicted_categories)):
    #         # print("row: ", row)
    #         print("prediction: ", prediction)
    #         usd_brl_spot_rate = row[1]['Adj Close']['USDBRL=X']
    #         print("usd_brl_spot_rate: ", usd_brl_spot_rate)
    #         current_date = row[0]
    #         print("current_date: ", current_date)
    #         # daily_change_percentage = row[1]['Daily_Change_Open_to_Close']
            
    #         # if self.jpy_inventory > 0:
    #         #     self.daily_return_factors.append(1 + (daily_change_percentage * self.leverage_factor))

    #         is_stop_loss_triggered = self._check_stop_loss(usd_brl_spot_rate, current_date)

    #         if is_stop_loss_triggered:
    #             continue

    #         # if prediction == 'Sell' and self.cash >= self.trading_lot:
    #         if prediction == 'Sell' and self.cash >= self.trading_lot and ( self.buy_price is None or (self.buy_price is not None and ( usd_brl_spot_rate < self.buy_price * 0.99 or usd_brl_spot_rate > self.buy_price * 1.01) ) ):
    #             self._buy_brl(usd_brl_spot_rate, current_date)
    #             print("buying USD BRL at : ", usd_brl_spot_rate, current_date)
    #         elif prediction == 'Buy' and self.brl_inventory > 0:
    #             self._sell_brl(usd_brl_spot_rate, current_date)
    #             print("selling USD BRL at : ", usd_brl_spot_rate, current_date)

    #         # if self._check_margin_call(usd_jpy_spot_rate):
    #         #     print("MARGIN CALL!!! this should not happen!")
    #         #     self._sell_jpy(usd_jpy_spot_rate, current_date)

    def execute_trades(self):
        previous_prediction = None  # Initialize with no previous prediction

        predicted_categories = self.model.predict()

        for index, (row, prediction) in enumerate(zip(self.data.iterrows(), predicted_categories)):
            if previous_prediction:  # Checking if there's a prediction from the previous day
                # Using current day's data for trading based on the previous day's prediction
                usd_brl_spot_rate = row[1]['Open']['USDBRL=X']
                current_date = row[0]

                print("Executing trade for date: ", current_date)
                print("Previous day prediction: ", previous_prediction)
                print("usd_brl_spot_rate: ", usd_brl_spot_rate)

                is_stop_loss_triggered = self._check_stop_loss(usd_brl_spot_rate, current_date)
                if is_stop_loss_triggered:
                    previous_prediction = prediction  # Update prediction for the next loop iteration
                    continue

                if previous_prediction == 'Sell' and self.cash >= self.trading_lot and (self.buy_price is None or (self.buy_price is not None and (usd_brl_spot_rate < self.buy_price * 0.99 or usd_brl_spot_rate > self.buy_price * 1.01))):
                    self._buy_brl(usd_brl_spot_rate, current_date)
                    print("buying USD BRL at: ", usd_brl_spot_rate, current_date)
                elif previous_prediction == 'Buy' and self.brl_inventory > 0:
                    self._sell_brl(usd_brl_spot_rate, current_date)
                    print("selling USD BRL at: ", usd_brl_spot_rate, current_date)

            # Update the previous prediction for the next day's trading action
            previous_prediction = prediction


    def execute_trades_perfect_future_knowledge(self):
        for index, row in self.data.iterrows():
            usd_jpy_spot_rate = row['Open']
            current_date = row['Date']
            daily_change_percentage = row['Daily_Change_Open_to_Close']

            if self.jpy_inventory > 0:
                self.daily_return_factors.append(1 + (daily_change_percentage * self.leverage_factor))

            is_stop_loss_triggered = self._check_stop_loss(usd_jpy_spot_rate, current_date)

            if is_stop_loss_triggered:
                continue

            if row['Label'] == 'Sell' and self.cash >= self.trading_lot and ( self.buy_price is None or (self.buy_price is not None and ( usd_jpy_spot_rate < self.buy_price * 0.99 or usd_jpy_spot_rate > self.buy_price * 1.01) ) ):
                self._buy_jpy(usd_jpy_spot_rate, current_date)
            elif row['Label'] == 'Buy' and self.jpy_inventory > 0:
                self._sell_jpy(usd_jpy_spot_rate, current_date)

            if self._check_margin_call(usd_jpy_spot_rate):
                print("MARGIN CALL!!! this should not happen!")
                self._sell_jpy(usd_jpy_spot_rate, current_date)

    def _buy_brl(self, rate, date):
        brl_bought = int(self.trading_lot * self.leverage_factor * rate)
        self.brl_inventory += brl_bought
        self.cash -= self.trading_lot
        self.buy_price = rate
        self.trade_log.append(f"111Buy {brl_bought} BRL at {rate} on {date}")

    def _sell_brl(self, rate, date, forced=False):
        if self.brl_inventory <= 0:
            return

        # jpy_convert_to_usd = ( self.jpy_inventory / rate ) / self.leverage_factor
        # self.cash += jpy_convert_to_usd
        self.cash = self._compute_mtm(rate)
        sell_reason = "Model predicted sell" if not forced else "Margin call / stop-loss triggered"
        self.trade_log.append(f"111Sell {self.brl_inventory} BRL at {rate} on {date} ({sell_reason})")

        self._apply_interest_charge(rate)

        self.brl_inventory = 0
        self.daily_return_factors = []

    def _compute_mtm(self, usd_brl_spot_rate):
        # print("usd_jpy_spot_rate: ", usd_jpy_spot_rate)
        # print("buy_price: ", self.buy_price)
        # sell_quantum = ( self.jpy_inventory / usd_jpy_spot_rate )
        # buy_quantum = ( self.jpy_inventory / self.buy_price )
        # print("sell_quantum: ", sell_quantum)
        # print("buy_quantum: ", buy_quantum)
        # return self.cash + self.trading_lot + ( sell_quantum  - buy_quantum )
        if self.brl_inventory <= 0:
            return self.cash

        # Calculate the current value of the JPY inventory at the current spot rate
        current_value = self.brl_inventory / usd_brl_spot_rate
        print("current_value: ", current_value)
        # Calculate the invested amount (in USD) for the JPY inventory
        invested_amount = (self.brl_inventory / self.buy_price)
        print("invested_amount: ", invested_amount)
        pnl = current_value - invested_amount
        principal = self.trading_lot
        # MTM is the current value minus the invested amount, adjusted for the cash
        mtm = self.cash + principal + pnl

        print("mtm: ", mtm)
        # # Subtracting total interest charges from the MTM
        # total_interest = sum(self.interest_costs)
        # return mtm - total_interest
        return mtm

    def _check_stop_loss(self, usd_brl_spot_rate, date):
        if self.brl_inventory > 0:
            change_percentage = (usd_brl_spot_rate - self.buy_price) / self.buy_price
            if change_percentage * self.leverage_factor > self.stop_loss_threshold:
                self._sell_brl(usd_brl_spot_rate, date, forced=True)
                print("!!!STOP LOSS TRIGGERED!!!")
                return True
        return False

    def _check_margin_call(self, usd_jpy_spot_rate):
        if self.jpy_inventory > 0:
            if self._compute_mtm(usd_jpy_spot_rate) < self.margin_requirement:
                return True
        return False

    def _apply_interest_charge(self, rate):
        days_held = len(self.daily_return_factors)
        daily_interest_rate = (1 + self.annual_interest_rate) ** (1/365) - 1
        # interest_charge = ( self.jpy_inventory / rate ) * daily_interest_rate * days_held
        borrowed_quantum = self.brl_inventory - ( self.brl_inventory / self.leverage_factor )
        interest_charge = ( borrowed_quantum / rate ) * daily_interest_rate * days_held
        self.interest_costs.append( interest_charge )

    def evaluate_performance(self):
        final_usd_brl_spot_rate = self.data.iloc[-1]['Adj Close']['USDBRL=X']
        # final_portfolio_value = self.cash + (self.jpy_inventory / final_usd_jpy_spot_rate)
        final_portfolio_value = self._compute_mtm(final_usd_brl_spot_rate)
        print("final_portfolio_value000: ", final_portfolio_value)
        print("shares: ", self.brl_inventory)
        pnl_per_trade = (final_portfolio_value - self.starting_cash) / len(self.trade_log) if self.trade_log else 0
        print("111 pnl_per_trade: ", pnl_per_trade)
        print("check interest_costs: ", self.interest_costs)
        return {
            'Final Portfolio Value': final_portfolio_value,
            'Number of Trades': len(self.trade_log),
            'Profit/Loss per Trade': pnl_per_trade,
            'Trade Log': self.trade_log,
            'Interest Costs': self.interest_costs,
            'Transaction Costs': len(self.trade_log),
        }
