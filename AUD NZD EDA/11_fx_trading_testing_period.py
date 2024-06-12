import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def generate_signal(Z_score, threshold_long, threshold_short):
    signal = pd.Series(index=Z_score.index, dtype=int)
    signal.iloc[0] = 0
    
    for t in range(1, len(Z_score)):
        if signal.iloc[t-1] == 0:
            if Z_score.iloc[t] <= threshold_long.iloc[t]:
                signal.iloc[t] = 1
            elif Z_score.iloc[t] >= threshold_short.iloc[t]:
                signal.iloc[t] = -1
            else:
                signal.iloc[t] = 0
        elif signal.iloc[t-1] == 1:
            if Z_score.iloc[t] >= 0:
                signal.iloc[t] = 0
            else:
                signal.iloc[t] = signal.iloc[t-1]
        else:
            if Z_score.iloc[t] <= 0:
                signal.iloc[t] = 0
            else:
                signal.iloc[t] = signal.iloc[t-1]
    
    return signal

def analyze_spread_LS(data, benchmark_data, pct_training=0.7, threshold=0.7):
    T = len(data)
    T_trn = round(pct_training * T)
    
    sgdusd = data['SGDUSD']
    gbpusd = data['GBPUSD']
    
    spread = sgdusd - gbpusd
    spread_mean = spread.iloc[:T_trn].mean()
    spread_std = spread.iloc[:T_trn].std()
    
    Z_score = (spread.iloc[T_trn:] - spread_mean) / spread_std
    Z_score_full = pd.Series(index=data.index, dtype=float)
    Z_score_full.iloc[:T_trn] = np.nan
    Z_score_full.iloc[T_trn:] = Z_score
    
    threshold_long = pd.Series(threshold, index=Z_score_full.index)
    threshold_short = -threshold_long
    
    signal = generate_signal(Z_score_full, threshold_long, threshold_short)
    
    portf_return = spread.diff() * signal.shift(1)
    
    # Filter to testing period
    portf_return_testing = portf_return.iloc[T_trn:]
    benchmark_return_testing = benchmark_data.pct_change().iloc[T_trn:]
    
    # Align the data to handle missing values and ensure same lengths
    portf_return_testing, benchmark_return_testing = portf_return_testing.align(benchmark_return_testing, join='inner')
    
    # Calculate Sharpe ratio
    mean_return = portf_return_testing.mean()
    std_return = portf_return_testing.std()
    sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized Sharpe ratio assuming daily returns
    
    # Calculate Alpha and Beta
    excess_return_testing = portf_return_testing - benchmark_return_testing
    X = sm.add_constant(benchmark_return_testing)
    model = sm.OLS(portf_return_testing, X, missing='drop').fit()
    alpha = model.params[0]
    beta = model.params[1]
    
    # Calculate Value at Risk (VaR)
    var_95 = np.percentile(portf_return_testing.dropna(), 5)
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot cumulative P&L for testing period
    axes.plot((1 + portf_return_testing).cumprod(), label='Cumulative Return')
    axes.axvline(x=portf_return_testing.index[0], color='blue', linestyle='--')  # Mark the start of the testing period
    axes.set_title('Cumulative P&L (Multiples of Notional) - Testing Period')
    axes.set_ylabel('Cumulative Return')
    axes.legend()

    # Set x-axis limits to start from the first data point of portf_return_testing
    axes.set_xlim(portf_return_testing.index[0], portf_return_testing.index[-1])

    # Display performance metrics
    sharpe_text = f'Sharpe Ratio: {sharpe_ratio:.2f}'
    alpha_text = f'Alpha: {alpha:.5f}'
    beta_text = f'Beta: {beta:.2f}'
    var_text = f'VaR (95%): {var_95:.2f}'
    metrics_text = f'{sharpe_text}\n{alpha_text}\n{beta_text}\n{var_text}'

    # Calculate the position for the text
    max_return = (1 + portf_return_testing).cumprod().max()
    text_x = portf_return_testing.index[-1] - pd.Timedelta(days=30)  # Adjusting x position for padding
    text_y = (1 + portf_return_testing).cumprod().min()  # Adjusting y position for padding

    # Display the performance metrics text
    axes.text(text_x, text_y, metrics_text, fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')

    # Add a dotted line for y=0
    axes.axhline(y=1, color='grey', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig('X11_trade_performance_testing_period.png')
    plt.show()


# Fetching data from Yahoo Finance
sgdusd_data = yf.download('SGDUSD=X', start='2016-01-01', end='2024-05-29')
gbpusd_data = yf.download('GBPUSD=X', start='2016-01-01', end='2024-05-29')
sp500_data = yf.download('^GSPC', start='2016-01-01', end='2024-05-29')  # Using S&P 500 as benchmark

# Extract the adjusted close price
sgdusd_close = sgdusd_data['Adj Close']
gbpusd_close = gbpusd_data['Adj Close']
sp500_close = sp500_data['Adj Close']

# Combine into a DataFrame
data = pd.concat([sgdusd_close, gbpusd_close], axis=1)
data.columns = ['SGDUSD', 'GBPUSD']

# Run the spread analysis
analyze_spread_LS(data, sp500_close)

