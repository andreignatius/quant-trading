import numpy as np
import pandas as pd
import talib
from gtda.diagrams import BettiCurve
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SlidingWindow
from hurst import compute_Hc
from pandas import DateOffset, to_datetime
from pykalman import KalmanFilter
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


class BaseModel:
    def __init__(self, file_path, train_start, train_end, test_start, test_end, trading_instrument, naming="Andre"):
        self.file_path = file_path
        # self.model_type = model_type
        self.data = None

        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

        self.model = None
        self.scaler = StandardScaler()

        self.fft_features = None

        self.naming = naming

        self.instruments = [
            "AUDUSD=X",
            "BZ=F",
            "CL=F",
            "GC=F",
            "NG=F",
            "NZDUSD=X",
            "SI=F",
            "USDAUD=X",
            "USDBRL=X",
            "USDCAD=X",
            "USDNOK=X",
            "USDZAR=X",
        ]

        self.trading_instrument = "USDBRL=X"

    def load_preprocess_data(self):
        # Load the data

        # self.data = pd.read_csv(self.file_path)
        self.data = pd.read_csv(self.file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)

        # Optional: Reinforce that the index is a datetime index (usually not needed)
        # self.data.index = pd.to_datetime(self.data.index)
        # Convert data types to reduce memory usage
        for dtype in ["float64", "int64"]:
            selected_dtype = "float32" if dtype == "float64" else "int32"
            self.data.loc[
                :, self.data.select_dtypes(include=[dtype]).columns
            ] = self.data.select_dtypes(include=[dtype]).astype(selected_dtype)

        # Forward fill missing data
        self.data.ffill(inplace=True)

        # Print the first few rows to check if loaded correctly
        print(self.data.head())

        # self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.calculate_daily_percentage_change()

        self.perform_fourier_transform_analysis()
        # self.calculate_stochastic_oscillator()
        # self.calculate_slow_stochastic_oscillator()
        self.construct_kalman_filter()
        self.detect_rolling_peaks_and_troughs()

        # self.calculate_moving_averages_and_rsi()
        # # self.calculate_bollinger_bands()
        # # self.calculate_bollinger_bandwidth()
        # # self.calculate_bollinger_percent_b()

        # self.estimate_hurst_exponent()
        self.calculate_days_since_peaks_and_troughs()
        self.detect_fourier_signals()
        # self.calculate_betti_curves()
        self.calculate_moving_averages_and_rsi()
        self.calculate_first_second_order_derivatives()
        
        self.ut_bot()
        self.WAE()
        self.BPMI()
        self.kijun_sen()
        self.T3MA_ATR()
        self.calculate_aroon()
        self.ssl_channel()

        self.nnfx_rule()
        self.ssl_rule()
        self.final_rachel()

        # self.integrate_tbill_data()
        # self.integrate_currency_account_data()

        # self.preprocess_data()

    def nnfx_rule(self):
        ut_buy = (self.data[f'ut_buy_{self.trading_instrument}'] == 1) | (self.data[f'ut_buy_{self.trading_instrument}'] == 1)
        ut_sell = (self.data[f'ut_sell_{self.trading_instrument}'] == 1) | (self.data[f'ut_sell_{self.trading_instrument}'] == 1)
        wae_buy_trend = (self.data[f'WAE_trend_up_{self.trading_instrument}'] > 0)
        wae_sell_trend = (self.data[f'WAE_trend_up_{self.trading_instrument}'] > 0)
        wae_buy_el = (self.data[f'WAE_trend_up_{self.trading_instrument}'] > self.data[f'WAE_el_{self.trading_instrument}'])
        wae_sell_el = (self.data[f'WAE_trend_down_{self.trading_instrument}'] > self.data[f'WAE_el_{self.trading_instrument}'])
        baseline_buy = (self.data[f'Close_{self.trading_instrument}'] > self.data[f't3_{self.trading_instrument}'])
        baseline_sell = (self.data[f'Close_{self.trading_instrument}'] < self.data[f't3_{self.trading_instrument}'])
        conflict_signal = ((self.data[f'ut_buy_{self.trading_instrument}'] == 1) & baseline_sell) | ((self.data[f'ut_sell_{self.trading_instrument}'] == 1) & baseline_buy)
        cross_up = (self.data[f'AROON_up_{self.trading_instrument}'].shift(1) < self.data[f'AROON_down_{self.trading_instrument}'].shift(1)) & \
        (self.data[f'AROON_up_{self.trading_instrument}'] > self.data[f'AROON_down_{self.trading_instrument}'])

        cross_down = (self.data[f'AROON_down_{self.trading_instrument}'].shift(1) < self.data[f'AROON_up_{self.trading_instrument}'].shift(1)) & \
        (self.data[f'AROON_down_{self.trading_instrument}'] > self.data[f'AROON_up_{self.trading_instrument}'])

        buy_signal = ut_buy & wae_buy_trend & wae_buy_el & baseline_buy
        exit_buy_signal = cross_up | conflict_signal
        sell_signal = ut_sell & wae_sell_trend & wae_sell_el & baseline_sell
        exit_sell_signal = cross_down | conflict_signal
        combined_signal = np.zeros(len(self.data))
        current_position = None
        ###################### Need to modify to buy hold sell ###############

        ''' based on Racheld_indicator1 attach buy / sell signal to self.data '''
        self.data["Trade1"] = rachel_signal

    def ssl_rule(self):
        cross_up_ssl = (self.data[f'sslUp_{self.trading_instrument}'].shift(1) < self.data[f'sslDown_{self.trading_instrument}'].shift(1)) & \
        (self.data[f'sslUp_{self.trading_instrument}'] > self.data[f'sslDown_{self.trading_instrument}'])
        
        cross_down_ssl = (self.data[f'sslDown_{self.trading_instrument}'].shift(1) < self.data[f'sslUp_{self.trading_instrument}'].shift(1)) & \
        (self.data[f'sslDown_{self.trading_instrument}'] > self.data[f'sslUp_{self.trading_instrument}'])

        kijun_buy = (self.data[f'Close_{self.trading_instrument}'] > self.data[f'kijun_sen_baseline_{self.trading_instrument}'])
        kijun_sell = (self.data[f'Close_{self.trading_instrument}'] < self.data[f'kijun_sen_baseline_{self.trading_instrument}'])

        wae_buy = (self.data[f'WAE_trend_up_{self.trading_instrument}'] > 0) & \
            (self.data[f'WAE_trend_up_{self.trading_instrument}'] > self.data[f'WAE_el_{self.trading_instrument}']) & \
            (self.data[f'WAE_trend_up_{self.trading_instrument}'] > self.data[f'WAE_dead_zone_{self.trading_instrument}'])
        
        wae_sell = (self.data[f'WAE_trend_down_{self.trading_instrument}'] > 0) & \
            (self.data[f'WAE_trend_down_{self.trading_instrument}'] > self.data[f'WAE_el_{self.trading_instrument}']) & \
            (self.data[f'WAE_trend_down_{self.trading_instrument}'] > self.data[f'WAE_dead_zone_{self.trading_instrument}'])

        buy_signal = cross_up_ssl & kijun_buy & wae_buy
        sell_signal = cross_down_ssl & kijun_sell & wae_sell
        exit_buy_signal = cross_down_ssl
        exit_sell_signal = cross_up_ssl
        ###################### Need to modify to buy hold sell ###############
        ''' based on Racheld_indicator1 attach buy / sell signal to self.data '''
        self.data["Trade2"] = rachel_signal

    def final_rachel(self):
        if self.data["Trade1"] and self.data["Trade2"]:
            self.data["Decision"] = ["Buy", "Sell", "Sell", "Buy", "Hold"]
        return self.data["Decision"]

    def calculate_daily_percentage_change(self):
        # Loop through each instrument's 'Close' column
        for instrument in [
            "AUDUSD=X",
            "BZ=F",
            "CL=F",
            "GC=F",
            "NG=F",
            "NZDUSD=X",
            "SI=F",
            "USDAUD=X",
            "USDBRL=X",
            "USDCAD=X",
            "USDNOK=X",
            "USDZAR=X",
        ]:
            adj_close_key = f"Adj_Close_{instrument}"
            open_key = f"Open_{instrument}"
            # Calculate daily percentage change for each 'Close' column
            if adj_close_key in self.data.columns:
                self.data[f"Daily_Change_{instrument}"] = self.data[adj_close_key].pct_change() * 100

            # Calculate the daily change from open to previous close
            if open_key in self.data.columns and adj_close_key in self.data.columns:
                self.data[f"Daily_Change_Open_to_Close_{instrument}"] = (
                    (self.data[open_key] - self.data[adj_close_key].shift(1)) / self.data[adj_close_key].shift(1) * 100
                )

    def perform_fourier_transform_analysis(self):
        # Fourier Transform Analysis
        d1 = self.train_end - pd.DateOffset(months=12)
        d2 = self.train_end

        data_window = self.data[(self.data.index >= d1) & (self.data.index < d2)].copy()

        close_prices = data_window[f"Adj_Close_{self.trading_instrument}"].to_numpy()

        # Compute the mean of non-NaN values
        mean_value = np.nanmean(close_prices)

        # Replace NaN values with the mean
        close_prices[np.isnan(close_prices)] = mean_value

        # Convert the series to numpy array for FFT
        # close_prices = data_window.to_numpy()
        N = len(close_prices)
        T = 1.0  # Assuming daily data, so the period is 1 day
        close_fft = fft(close_prices)
        fft_freq = np.fft.fftfreq(N, T)
        positive_frequencies = fft_freq[: N // 2]
        positive_fft_values = 2.0 / N * np.abs(close_fft[0 : N // 2])
        amplitude_threshold = 0.1  # This can be adjusted based on your data scale
        significant_peaks, _ = find_peaks(
            positive_fft_values, height=amplitude_threshold
        )
        significant_frequencies = positive_frequencies[significant_peaks]
        significant_amplitudes = positive_fft_values[significant_peaks]
        days_per_cycle = 1 / significant_frequencies

        # Store the results in a DataFrame
        self.fft_features = pd.DataFrame(
            {
                "Frequency": significant_frequencies,
                "Amplitude": significant_amplitudes,
                "DaysPerCycle": days_per_cycle,
            }
        )

    def calculate_stochastic_oscillator(
        self, k_window=14, d_window=3, slow_k_window=3
    ):
        """
        Calculate the Stochastic Oscillator.
        %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K

        Where:
        - Lowest Low = lowest low for the look-back period
        - Highest High = highest high for the look-back period
        - %K is multiplied by 100 to move the decimal point two places

        The result is two time series: %K and %D
        """
        # Calculate %K
        # Use MultiIndex to specify levels if DataFrame has multi-level columns
        # Adjust instrument as needed or generalize for multiple instruments
        # Calculate %K
        low_min = self.data[f"Adj_Close_{self.trading_instrument}"].rolling(window=k_window).min()
        high_max = self.data[f"Adj_Close_{self.trading_instrument}"].rolling(window=k_window).max()
        current_close = self.data[f"Adj_Close_{self.trading_instrument}"]

        self.data[f"%K_{self.trading_instrument}"] = (
            100 * (current_close - low_min) / (high_max - low_min)
        )

        # Calculate %D as the moving average of %K
        self.data[f"%D_{self.trading_instrument}"] = (
            self.data[f"%K_{self.trading_instrument}"].rolling(window=d_window).mean()
        )

        # Handle any NaN values that may have been created
        self.data[f"%K_{self.trading_instrument}"].bfill(inplace=True)
        self.data[f"%D_{self.trading_instrument}"].bfill(inplace=True)

    def calculate_slow_stochastic_oscillator(self, d_window=3, slow_k_window=3):
        """
        Calculate the Slow Stochastic Oscillator.
        Slow %K = 3-period SMA of %K
        Slow %D = 3-day SMA of Slow %K
        """
        # Calculate Slow %K, which is the moving average of %K
        self.data["Slow %K"] = self.data["%K"].rolling(window=slow_k_window).mean()

        # Calculate Slow %D, which is the moving average of Slow %K
        self.data["Slow %D"] = self.data["Slow %K"].rolling(window=d_window).mean()

        # Handle any NaN values
        self.data["Slow %K"].bfill(inplace=True)
        self.data["Slow %D"].bfill(inplace=True)

    def detect_fourier_signals(self):
        # Initialize Fourier signal columns for each instrument
        for instrument in self.instruments:
            self.data[f"FourierSignalSell_{instrument}"] = False
            self.data[f"FourierSignalBuy_{instrument}"] = False

        # Process each instrument
        for instrument in self.instruments:
            # Get dominant periods, filter, and sort
            dominant_periods = sorted(
                set(
                    (self.fft_features.loc[:10, "DaysPerCycle"].values / 2).astype(int)
                ),
                reverse=True,
            )
            dominant_periods = [
                p for p in dominant_periods if p < 30
            ]  # Filter out long periods
            dominant_periods = dominant_periods[:5]  # Take top 5
            print("Dominant Period Lengths for ", instrument, ": ", dominant_periods)
            # Mark the Fourier signals based on the dominant periods
            self.data[f"FourierSignalSell_{instrument}"] = self.data[
                f"DaysSinceTrough_{instrument}"
            ].isin(dominant_periods)
            self.data[f"FourierSignalBuy_{instrument}"] = self.data[
                f"DaysSincePeak_{instrument}"
            ].isin(dominant_periods)

    def detect_rolling_peaks_and_troughs(self, window_size=5):
        # Iterate through each instrument
        for instrument in self.instruments:
            close_key = f"Close_{instrument}" # Adjust this if your 'Close' data is stored differently

            # Initialize columns to store results for each instrument
            self.data[f"isLocalPeak_{instrument}"] = False
            self.data[f"isLocalTrough_{instrument}"] = False

            # Retrieve close data for the instrument
            close_series = self.data[close_key]

            # Iterate through the DataFrame using a rolling window
            for end_idx in range(window_size, len(close_series)):
                start_idx = max(0, end_idx - window_size)
                window_data = close_series[start_idx:end_idx]

                # Find peaks
                peaks, _ = find_peaks(window_data)
                peaks_global_indices = [
                    close_series.index[start_idx + p] for p in peaks
                ]
                self.data.loc[peaks_global_indices, f"isLocalPeak_{instrument}"] = True

                # Find troughs by inverting the data
                troughs, _ = find_peaks(-window_data)
                troughs_global_indices = [
                    close_series.index[start_idx + t] for t in troughs
                ]
                self.data.loc[
                    troughs_global_indices, f"isLocalTrough_{instrument}"
                ] = True

            # Assign labels based on peaks and troughs
            self.data[f"Label_{instrument}"] = "Hold"  # Default label
            self.data.loc[
                self.data[f"isLocalPeak_{instrument}"].shift(-1).fillna(False), f"Label_{instrument}"
            ] = "Sell"
            self.data.loc[
                self.data[f"isLocalTrough_{instrument}"].shift(-1).fillna(False), f"Label_{instrument}"
            ] = "Buy"

        # Optional: Handle data cleaning if required
        self.data.ffill(inplace=True)

    # Calculating Moving Averages and RSI manually
    def calculate_rsi(self, instrument, window=14):
        """Calculate the Relative Strength Index (RSI) for a given dataset and window"""
        delta = self.data[f"Close_{instrument}"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_moving_averages_and_rsi(self):
        # Initialize Fourier signal columns for each instrument
        for instrument in self.instruments:
            short_window = 5
            long_window = 20
            rsi_period = 14
            self.data[f"Short_Moving_Avg_{instrument}"] = (
                self.data[f"Close_{instrument}"].rolling(window=short_window).mean()
            )
            self.data[f"Long_Moving_Avg_{instrument}"] = (
                self.data[f"Close_{instrument}"].rolling(window=long_window).mean()
            )
            self.data[f"RSI_{instrument}"] = self.calculate_rsi(
                instrument, window=rsi_period
            )

    # Bollinger Bands
    def calculate_bollinger_bands(self):
        (
            self.data["BBand_Upper"],
            self.data["BBand_Middle"],
            self.data["BBand_Lower"],
        ) = talib.BBANDS(self.data["Close"], timeperiod=20)

    # Bollinger Bandwidth
    def calculate_bollinger_bandwidth(self):
        self.data["Bollinger_Bandwidth"] = (
            self.data["BBand_Upper"] - self.data["BBand_Lower"]
        ) / self.data["BBand_Middle"]

    # Bollinger %B
    def calculate_bollinger_percent_b(self):
        self.data["Bollinger_PercentB"] = (
            self.data["Close"] - self.data["BBand_Lower"]
        ) / (self.data["BBand_Upper"] - self.data["BBand_Lower"])

    def construct_kalman_filter(self):
        for instrument in self.instruments:
            close_prices = self.data[f"Close_{instrument}"]
            # Construct a Kalman Filter
            kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

            # Use the observed data (close prices) to estimate the state
            state_means, _ = kf.filter(close_prices.values)

            # Convert state means to a Pandas Series for easy plotting
            kalman_estimates = pd.Series(state_means.flatten(), index=self.data.index)

            # Combine the original close prices and Kalman Filter estimates
            kalman_estimates = pd.DataFrame(
                {f"KalmanFilterEst_{instrument}": kalman_estimates}
            )
            self.data = self.data.join(kalman_estimates)

            # print('KalmanFilterEst: ', self.data['KalmanFilterEst'])  # Display the first few rows of the dataframe

    # Function to estimate Hurst exponent over multiple sliding windows
    def estimate_hurst_exponent(self, window_size=100, step_size=1):
        """
        Calculates the Hurst exponent over sliding windows and appends the results to the DataFrame.

        :param window_size: Size of each sliding window.
        :param step_size: Step size for moving the window.
        """
        # Prepare an empty list to store Hurst exponent results and corresponding dates
        hurst_exponents = []

        for i in range(0, len(self.data) - window_size + 1, step_size):
            # Extract the window
            window = self.data["Close"].iloc[i : i + window_size]

            # Calculate the Hurst exponent
            H, _, _ = compute_Hc(window, kind="price", simplified=True)

            # Store the result along with the starting date of the window
            hurst_exponents.append(
                {"Date": self.data["Date"].iloc[i], "HurstExponent": H}
            )

        # Convert the results list into a DataFrame
        hurst_exponents = pd.DataFrame(hurst_exponents)

        # Combine the original DataFrame and Hurst exponent estimates
        self.data = self.data.merge(hurst_exponents, on="Date", how="left")

    def calculate_days_since_peaks_and_troughs(self):
        # Prepare columns for each instrument
        for instrument in self.instruments:
            self.data[f"DaysSincePeak_{instrument}"] = 0
            self.data[f"DaysSinceTrough_{instrument}"] = 0
            self.data[f"PriceChangeSincePeak_{instrument}"] = 0
            self.data[f"PriceChangeSinceTrough_{instrument}"] = 0

        # Iterate over each instrument
        for instrument in self.instruments:
            checkpoint_date_bottom = None
            checkpoint_date_top = None
            checkpoint_price_bottom = None
            checkpoint_price_top = None

            # Iterate over rows
            for today_date, row in self.data.iterrows():
                current_price = row[f"Close_{instrument}"]

                # Check for buy/sell signals and update checkpoints
                if row[f"Label_{instrument}"] == "Buy":
                    checkpoint_date_bottom = today_date
                    checkpoint_price_bottom = current_price
                if row[f"Label_{instrument}"] == "Sell":
                    checkpoint_date_top = today_date
                    checkpoint_price_top = current_price

                # Calculate days since events
                days_since_bottom = (
                    (today_date - checkpoint_date_bottom).days
                    if checkpoint_date_bottom
                    else 0
                )
                days_since_peak = (
                    (today_date - checkpoint_date_top).days
                    if checkpoint_date_top
                    else 0
                )

                # Calculate price changes since events
                price_change_since_bottom = (
                    current_price - checkpoint_price_bottom
                    if checkpoint_price_bottom is not None
                    else 0
                )
                price_change_since_peak = (
                    current_price - checkpoint_price_top
                    if checkpoint_price_top is not None
                    else 0
                )

                # Update DataFrame
                self.data.at[
                    today_date, f"DaysSincePeak_{instrument}"
                ] = days_since_peak
                self.data.at[
                    today_date, f"DaysSinceTrough_{instrument}"
                ] = days_since_bottom
                self.data.at[
                    today_date, f"PriceChangeSincePeak_{instrument}"
                ] = price_change_since_peak
                self.data.at[
                    today_date, f"PriceChangeSinceTrough_{instrument}"
                ] = price_change_since_bottom

    def compute_persistence_norms(self, persistence_diagram):
        """
        Compute L1 and L2 persistence norms from a persistence diagram.

        Parameters:
        persistence_diagram (np.ndarray): Array of (birth, death) pairs.

        Returns:
        tuple: (L1_norm, L2_norm)
        """
        lifetimes = persistence_diagram[:, 1] - persistence_diagram[:, 0]
        L1_norm = np.sum(np.abs(lifetimes))
        L2_norm = np.sqrt(np.sum(lifetimes**2))
        return L1_norm, L2_norm

    def calculate_betti_curves(self, K=10, d=1, w=5):
        """
        Calculate Betti Curves and persistence norms for the given data.

        :param K: Number of top nodes to consider.
        :param d: Maximum Betti dimension.
        :param w: Window size for rolling persistence diagram.
        """
        instruments = ["CL=F", "GC=F", "SI=F"]  # Adjust based on your data

        betti_curve_dfs = []
        norm_dfs = []

        for instrument in instruments:
            print(f"Calculating Betti curves and persistence norms for {instrument}")

            # Extract data for the instrument
            data_series = self.data[f"Adj_Close_{instrument}"].dropna()

            # Create sliding windows
            sliding_window = SlidingWindow(size=w, stride=1)
            windows = sliding_window.fit_transform(data_series.values.reshape(-1, 1))

            # Compute Vietoris-Rips persistence diagrams
            VR_persistence = VietorisRipsPersistence(
                homology_dimensions=list(range(d + 1))
            )
            diagrams = VR_persistence.fit_transform(windows)

            # Compute Betti curves
            betti_curves = BettiCurve(n_bins=w)
            betti_curves_values = betti_curves.fit_transform(diagrams)

            # Compute and store Betti curves
            for dim in range(d + 1):
                col_name = f"BettiCurve_{dim}_{instrument}"
                betti_curve_series = pd.Series(
                    betti_curves_values[:, dim, :].mean(axis=1),
                    index=data_series.index[w - 1 :],
                )
                betti_curve_df = pd.DataFrame(betti_curve_series, columns=[col_name])
                betti_curve_dfs.append(betti_curve_df)

            # Compute and store persistence norms
            L1_norms = []
            L2_norms = []
            for diagram in diagrams:
                L1, L2 = self.compute_persistence_norms(diagram)
                L1_norms.append(L1)
                L2_norms.append(L2)

            norm_df = pd.DataFrame(
                {
                    f"L1_Persistence_{instrument}": L1_norms,
                    f"L2_Persistence_{instrument}": L2_norms,
                },
                index=data_series.index[w - 1 :],
            )
            norm_dfs.append(norm_df)

        # Concatenate all Betti curve DataFrames and norm DataFrames
        betti_curve_combined_df = pd.concat(betti_curve_dfs + norm_dfs, axis=1)

        # Flatten the MultiIndex in self.data to a single level
        self.data.columns = ["_".join(col).strip() for col in self.data.columns.values]

        # Join the Betti curves and norm DataFrame with the main data
        self.data = self.data.join(betti_curve_combined_df, how="left")

        # Restore the original MultiIndex structure if needed
        self.data.columns = pd.MultiIndex.from_tuples(
            [
                (col.split("_", 1)[0], col.split("_", 1)[1] if "_" in col else "")
                for col in self.data.columns
            ]
        )

        print("check betti data: ", self.data)


    def calculate_first_second_order_derivatives(self):
        # Calculate first and second order derivatives for selected features
        for feature in ["KalmanFilterEst", "Short_Moving_Avg", "Long_Moving_Avg"]:
            for instrument in self.instruments:
                self.data[f"{feature}_1st_Deriv_{instrument}"] = (
                    self.data[f"{feature}_{instrument}"].diff() * 100
                )
                self.data[f"{feature}_2nd_Deriv_{instrument}"] = (
                    self.data[f"{feature}_1st_Deriv_{instrument}"].diff() * 100
                )

        # Fill NaN values that were generated by diff()
        self.data.bfill(inplace=True)

    def ut_bot(self, key_value=2, atr_period=10):
        # Calculate EMA with a span of 1
        def pine_ema(src, length):
            alpha = 2 / (length + 1)
            sum_series = np.zeros_like(src)
            sum_series[0] = src[0]
            for i in range(1, len(src)):
                sum_series[i] = alpha * src[i] + (1 - alpha) * sum_series[i - 1]
            return sum_series
        
        # Define the crossover function
        def crossover(series1, series2):
            return (series1 > series2) & (series1.shift() < series2.shift())

        instrument = self.trading_instrument
        close_col = f'Close_{instrument}'
        high_col = f'High_{instrument}'
        low_col = f'Low_{instrument}'
        print(close_col, high_col, low_col)
        src = self.data[close_col]
        high = self.data[high_col]
        low = self.data[low_col]

        # Calculate ATR
        atr = talib.ATR(high, low, src, timeperiod=atr_period)
        n_loss = key_value * atr

        # Initialize ATR Trailing Stop
        xatr_trailing_stop = np.zeros(len(src))

        for i in range(1, len(src)):
            prev_stop = xatr_trailing_stop[i - 1]
            if src[i] > prev_stop and src[i - 1] > prev_stop:
                xatr_trailing_stop[i] = max(prev_stop, src[i] - n_loss[i])
            elif src[i] < prev_stop and src[i - 1] < prev_stop:
                xatr_trailing_stop[i] = min(prev_stop, src[i] + n_loss[i])
            elif src[i] > prev_stop:
                xatr_trailing_stop[i] = src[i] - n_loss[i]
            else:
                xatr_trailing_stop[i] = src[i] + n_loss[i]

        # Determine positions
        position = np.zeros(len(src))
        for i in range(1, len(src)):
            if src[i - 1] < xatr_trailing_stop[i - 1] and src[i] > xatr_trailing_stop[i - 1]:
                position[i] = 1
            elif src[i - 1] > xatr_trailing_stop[i - 1] and src[i] < xatr_trailing_stop[i - 1]:
                position[i] = -1
            else:
                position[i] = position[i - 1]

        ema = pine_ema(src, length=1)

        # Calculate crossovers
        above = crossover(pd.Series(ema), pd.Series(xatr_trailing_stop))
        below = crossover(pd.Series(xatr_trailing_stop), pd.Series(ema))

        # Define buy and sell conditions
        buy = np.where((src.values > xatr_trailing_stop) & above, 1, 0)
        sell = np.where((src.values < xatr_trailing_stop) & below, 1, 0)

        # Add the buy and sell columns to the dataframe
        self.data[f'ut_buy_{instrument}'] = buy
        self.data[f'ut_sell_{instrument}'] = sell

    def BPMI(self, length=15, smoothing=6):
        instrument = self.trading_instrument
        close_col = f'Close_{instrument}'

        # Calculate Polychromatic Momentum Indicator
        mom = self.data[close_col] - self.data[close_col].shift(length)
        pmi = mom - (mom - mom.shift(length)) / length

        # Smoothed PMI
        sm_pmi = pmi.rolling(window=smoothing).mean()

        # Add the smoothed PMI column to the dataframe
        self.data[f'sm_pmi_{instrument}'] = sm_pmi

    def WAE(self, sensitivity=80, fast_length=22, slow_length=45, channel_length=25, mult=2.0):
        def rma(source, length):
            alpha = 1 / length
            sum_value = np.zeros_like(source, dtype=np.float64)
            
            for i in range(len(source)):
                if i == 0:
                    sum_value[i] = np.nan  # Set the first value as NaN since it's not available
                elif np.isnan(sum_value[i - 1]):
                    sum_value[i] = np.mean(source[:length])  # Calculate SMA for the first available value
                else:
                    sum_value[i] = alpha * source[i] + (1 - alpha) * sum_value[i - 1]  # Calculate RMA
                    
            return sum_value

        def calculate_macd(source, fast_length, slow_length):
            fast_ma = talib.EMA(source, timeperiod=fast_length)
            slow_ma = talib.EMA(source, timeperiod=slow_length)
            return fast_ma - slow_ma

        def calculate_bb_upper(source, length, mult):
            basis = talib.SMA(source, timeperiod=length)
            dev = mult * talib.STDDEV(source, timeperiod=length)
            return basis + dev

        def calculate_bb_lower(source, length, mult):
            basis = talib.SMA(source, timeperiod=length)
            dev = mult * talib.STDDEV(source, timeperiod=length)
            return basis - dev

        instrument = self.trading_instrument
        high_col = f'High_{instrument}'
        low_col = f'Low_{instrument}'
        close_col = f'Close_{instrument}'

        data = self.data[[high_col, low_col, close_col]].copy()

        # Calculate ATR and Dead Zone
        data['tr'] = talib.TRANGE(data[high_col], data[low_col], data[close_col])
        data['rma_tr'] = rma(data['tr'], length=100)
        data['dead_zone'] = data['rma_tr'] * 3.7

        # Calculate MACD values
        macd_current = calculate_macd(data[close_col], fast_length, slow_length)
        macd_previous = calculate_macd(data[close_col].shift(1), fast_length, slow_length)
        data['t1'] = (macd_current - macd_previous) * sensitivity

        # Calculate Bollinger Bands width
        data['bb_upper'] = calculate_bb_upper(data[close_col], channel_length, mult)
        data['bb_lower'] = calculate_bb_lower(data[close_col], channel_length, mult)
        data['e1'] = data['bb_upper'] - data['bb_lower']

        # Determine trend direction
        data['trend_up'] = np.where(data['t1'] >= 0, data['t1'], 0)
        data['trend_down'] = np.where(data['t1'] < 0, -data['t1'], 0)

        # Add the final columns to the main dataframe
        self.data[f'WAE_trend_up_{instrument}'] = data['trend_up']
        self.data[f'WAE_trend_down_{instrument}'] = data['trend_down']
        self.data[f'WAE_e1_{instrument}'] = data['e1']
        self.data[f'WAE_dead_zone_{instrument}'] = data['dead_zone']

    def T3MA_ATR(self, length=20, factor=0.4):
        def rma(source, length):
            alpha = 1 / length
            sum_value = np.zeros_like(source, dtype=np.float64)
            
            for i in range(len(source)):
                if i == 0:
                    sum_value[i] = np.nan  # Set the first value as NaN since it's not available
                elif np.isnan(sum_value[i - 1]):
                    sum_value[i] = np.mean(source[:length])  # Calculate SMA for the first available value
                else:
                    sum_value[i] = alpha * source[i] + (1 - alpha) * sum_value[i - 1]  # Calculate RMA
                    
            return sum_value
    
        instrument = self.trading_instrument
        close_col = f'Close_{instrument}'
        high_col = f'High_{instrument}'
        low_col = f'Low_{instrument}'

        self.data[f't3_{instrument}'] = talib.T3(self.data[close_col], length, factor)
        self.data[f'tr_{instrument}'] = talib.TRANGE(self.data[high_col], self.data[low_col], self.data[close_col])
        self.data[f'atr_{instrument}'] = rma(self.data[f'tr_{instrument}'], 14)
        self.data[f'atr_upper_{instrument}'] = self.data[f't3_{instrument}'] + self.data[f'atr_{instrument}']
        self.data[f'atr_lower_{instrument}'] = self.data[f't3_{instrument}'] - self.data[f'atr_{instrument}']

        # Drop intermediate columns
        # self.data.drop([f'tr_{instrument}', f'atr_{instrument}'], axis=1, inplace=True)        
    def calculate_aroon(self, length = 10):
        high_col = f'High_{self.trading_instrument}'
        low_col = f'Low_{self.trading_instrument}'
        self.data[f'AROON_down_{self.trading_instrument}'], self.data[f'AROON_up_{self.trading_instrument}'] = \
        talib.AROON(self.data[high_col], self.data[low_col], length)


    def ssl_channel(self, length=10):
        instrument = self.trading_instrument
        high_col = f'High_{instrument}'
        low_col = f'Low_{instrument}'
        close_col = f'Close_{instrument}'

        # Calculate SMAs of high and low prices
        self.data[f'smaHigh_{instrument}'] = self.data[high_col].rolling(window=length).mean()
        self.data[f'smaLow_{instrument}'] = self.data[low_col].rolling(window=length).mean()

        # Initialize Hlv (Horizontal Line Value)
        self.data[f'Hlv_{instrument}'] = np.nan

        # Determine Hlv values
        for i in range(len(self.data)):
            if pd.notnull(self.data[close_col].iloc[i]):
                if self.data[close_col].iloc[i] > self.data[f'smaHigh_{instrument}'].iloc[i]:
                    self.data[f'Hlv_{instrument}'].iloc[i] = 1
                elif self.data[close_col].iloc[i] < self.data[f'smaLow_{instrument}'].iloc[i]:
                    self.data[f'Hlv_{instrument}'].iloc[i] = -1
                else:
                    self.data[f'Hlv_{instrument}'].iloc[i] = self.data[f'Hlv_{instrument}'].iloc[i - 1]

        # Calculate sslDown and sslUp
        self.data[f'sslDown_{instrument}'] = np.where(self.data[f'Hlv_{instrument}'] < 0, self.data[f'smaHigh_{instrument}'], self.data[f'smaLow_{instrument}'])
        self.data[f'sslUp_{instrument}'] = np.where(self.data[f'Hlv_{instrument}'] < 0, self.data[f'smaLow_{instrument}'], self.data[f'smaHigh_{instrument}'])

        # Drop intermediate columns
        self.data.drop([f'smaHigh_{instrument}', f'smaLow_{instrument}', f'Hlv_{instrument}'], axis=1, inplace=True)
            
    def kijun_sen(self, period=26):
        # Define function to calculate Donchian channel
        def donchian_channel(data, length):
            lowest_low = data[low_col].rolling(window=length).min()
            highest_high = data[high_col].rolling(window=length).max()
            return (lowest_low + highest_high) / 2
        
        instrument = self.trading_instrument
        low_col = f'Low_{instrument}'
        high_col = f'High_{instrument}'
        # Calculate Kijun-Sen (base line)
        self.data[f'kijun_sen_baseline_{instrument}'] = donchian_channel(self.data, period)

    def integrate_tbill_data(self):
        file_path_JPYTBill = "data/JPY_1Y_TBill_GJTB12MO.csv"
        file_path_USTBill = "data/USD_1Y_TBill_H15T1Y.csv"

        TBill_data_JP = pd.read_csv(file_path_JPYTBill)
        TBill_data_US = pd.read_csv(file_path_USTBill)

        # get TBill data
        TBill_data = TBill_data_US.merge(TBill_data_JP, on="Date")
        TBill_data["Interest_Rate_Difference"] = (
            TBill_data["PX_LAST_y"] - TBill_data["PX_LAST_x"]
        )
        TBill_data["Interest_Rate_Difference_Change"] = (
            TBill_data["Interest_Rate_Difference"].diff() * 10
        )
        TBill_data = TBill_data[
            ["Date", "Interest_Rate_Difference", "Interest_Rate_Difference_Change"]
        ]
        TBill_data["Date"] = pd.to_datetime(TBill_data["Date"], dayfirst=True)
        print("check TBill_data: ", TBill_data)
        self.data = self.data.merge(TBill_data, on="Date")

    def integrate_currency_account_data(self):
        file_path_CurrencyAccount_JP = "data/Japan_Currency_Account.xlsx"
        file_path_CurrencyAccount_US = "data/US_Currency_Account.xlsx"

        CurrencyAccount_JP = pd.read_excel(
            file_path_CurrencyAccount_JP, parse_dates=True
        )
        CurrencyAccount_US = pd.read_excel(
            file_path_CurrencyAccount_US, parse_dates=True
        )

        # get Currency Account data
        CurrencyAccount_data = CurrencyAccount_US.merge(CurrencyAccount_JP, on="Date")
        CurrencyAccount_Date = CurrencyAccount_data["Date"].values
        CurrencyAccount_data = CurrencyAccount_data.set_index("Date")
        CurrencyAccount_data = CurrencyAccount_data[["PX_LAST_x", "PX_LAST_y"]]
        CurrencyAccount_data_scaled = self.scaler.fit_transform(CurrencyAccount_data)
        CurrencyAccount_data = pd.DataFrame(
            CurrencyAccount_data_scaled, index=CurrencyAccount_Date
        )
        CurrencyAccount_data["Currency_Account_difference"] = (
            CurrencyAccount_data.iloc[:, 1] - CurrencyAccount_data.iloc[:, 0]
        )
        CurrencyAccount_data = CurrencyAccount_data[["Currency_Account_difference"]]
        # Linear Interpolate the data
        CurrencyAccount_data = CurrencyAccount_data.resample("D").interpolate()
        CurrencyAccount_data["Currency_Account_difference"] = CurrencyAccount_data[
            "Currency_Account_difference"
        ].interpolate()
        CurrencyAccount_data = CurrencyAccount_data.reset_index()
        CurrencyAccount_data.columns = ["Date", "Currency_Account_difference"]

        self.data = self.data.merge(CurrencyAccount_data, on="Date")

    def preprocess_data(self):
        self.data.dropna(inplace=True)

    def train_test_split_time_series(self):
        # Ensure 'Date' is the DataFrame index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data["Date"] = pd.to_datetime(self.data.index)
            self.data.set_index("Date", inplace=True)

        # Sort the data by date to ensure correct time sequence
        self.data.sort_index(inplace=True)

        # Filter the data for training and testing periods
        self.train_data = self.data.loc[self.train_start : self.train_end].copy()
        self.test_data = self.data.loc[self.test_start : self.test_end].copy()

        # Calculate the age of each data point in days
        current_date = pd.to_datetime(self.train_end)
        self.train_data["DataAge"] = (current_date - self.train_data.index).days

        # Apply exponential decay to calculate weights
        decay_rate = 0.05  # This is a parameter you can tune
        self.train_data["Weight"] = np.exp(-decay_rate * self.train_data["DataAge"])

        # Now sample from training data with these weights
        sample_size = min(len(self.train_data), 1500)
        self.train_data = self.train_data.sample(
            n=sample_size, replace=False, weights=self.train_data["Weight"]
        )
        self.train_data.sort_index(inplace=True)
        self.train_data.to_csv("outputs/inspect_training_set.csv")
        self.test_data.to_csv("outputs/inspect_testing_set.csv")

        feature_set = [
            # List your features here
            # 'Short_Moving_Avg_2nd_Deriv',
            # 'Long_Moving_Avg_2nd_Deriv',
            # 'RSI',
            f"DaysSincePeak_{self.trading_instrument}",
            f"DaysSinceTrough_{self.trading_instrument}",
            f"FourierSignalSell_{self.trading_instrument}",
            f"FourierSignalBuy_{self.trading_instrument}",
            # f"%K_{instrument}",
            # f"%D_{instrument}",
            # 'Slow %K',
            # 'Slow %D',
            # 'KalmanFilterEst_1st_Deriv',
            # 'KalmanFilterEst_2nd_Deriv',
            # 'Interest_Rate_Difference_Change'
        ]

        feature_set2 = [
            # List your features here
            # 'Short_Moving_Avg_2nd_Deriv',
            # 'Long_Moving_Avg_2nd_Deriv',
            # 'RSI',
            f"Close_{self.trading_instrument}",
            # f"%K_{instrument}",
            # f"%D_{instrument}",
            # 'Slow %K',
            # 'Slow %D',
            # 'KalmanFilterEst_1st_Deriv',
            # 'KalmanFilterEst_2nd_Deriv',
            # 'Interest_Rate_Difference_Change'
        ]

        # Extract the features for training and testing sets
        self.X_train = self.train_data[feature_set]
        self.X_test = self.test_data[feature_set]
        self.y_train = self.train_data[f"Label_{self.trading_instrument}"]
        self.y_test = self.test_data[f"Label_{self.trading_instrument}"]
        self.data.ffill(inplace=True)

        if (self.naming == "Rachel"):
            self.X_train = self.train_data[feature_set2]
            self.X_test = self.test_data[feature_set2]
            self.y_train = self.train_data[f"Label_{self.trading_instrument}"]
            self.y_test = self.test_data[f"Label_{self.trading_instrument}"]

        print("len X train: ", len(self.X_train))
        print("len X test: ", len(self.X_test))
        print("len y train: ", len(self.y_train))
        print("len y test: ", len(self.y_test))

    def retrieve_test_set(self):
        return self.test_data

    def train(self):
        # Implement or leave empty to override in derived classes
        pass

    def predict(self):
        # Implement or leave empty to override in derived classes
        pass

    def evaluate(self, X, y):
        # Implement evaluation logic
        pass
