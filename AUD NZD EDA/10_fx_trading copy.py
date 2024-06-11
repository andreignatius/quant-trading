import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def analyze_spread_LS(data, pct_training=0.7, threshold=0.7):
    T = len(data)
    T_trn = round(pct_training * T)
    
    sgdusd = data['SGDUSD']
    gbpusd = data['GBPUSD']
    
    spread = sgdusd - gbpusd
    spread_mean = spread.iloc[:T_trn].mean()
    spread_std = spread.iloc[:T_trn].std()
    
    Z_score = (spread - spread_mean) / spread_std
    
    threshold_long = pd.Series(threshold, index=Z_score.index)
    threshold_short = -threshold_long
    
    signal = generate_signal(Z_score, threshold_long, threshold_short)
    
    portf_return = spread.diff() * signal.shift(1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot Z-score and thresholds, and trading signal
    axes[0].plot(Z_score, label='Z-score', color='blue')
    axes[0].plot(threshold_long, linestyle='--', color='green', label='Long Threshold')
    axes[0].plot(threshold_short, linestyle='--', color='red', label='Short Threshold')
    axes[0].plot(signal.index, signal, color='orange', linewidth=1.2, label='Trading Signal')
    axes[0].legend()
    
    gap = 5
    axes[0].text(data.index[T_trn-gap], Z_score.min(), 'Training Period     ', color='blue', verticalalignment='bottom', horizontalalignment='right')
    axes[0].text(data.index[T_trn+gap], Z_score.min(), '     Testing Period', color='blue', verticalalignment='bottom', horizontalalignment='left')
    axes[0].text(data.index[len(data)//2], threshold, f'Long Threshold: {threshold}', fontsize=10, color='green', ha='center')
    axes[0].text(data.index[len(data)//2], -threshold, f'Short Threshold: {-threshold}', fontsize=10, color='red', ha='center')

    # Plot cumulative P&L
    axes[1].plot((1 + portf_return).cumprod(), label='Cumulative Return')
    axes[1].axvline(x=data.index[T_trn], color='blue', linestyle='--')
    axes[1].set_title('Cumulative P&L (Multiples of Notional)')
    axes[1].set_ylabel('Cumulative Return')
    axes[1].legend()
    
    axes[1].text(data.index[T_trn-gap], (1 + portf_return).cumprod().min(), 'Training Period     ', color='blue', verticalalignment='bottom', horizontalalignment='right')
    axes[1].text(data.index[T_trn+gap], (1 + portf_return).cumprod().min(), '     Testing Period', color='blue', verticalalignment='bottom', horizontalalignment='left')

    plt.tight_layout()
    plt.savefig('X10_trade_performance.png')
    plt.show()

# Fetching data from Yahoo Finance
# Fetching data from Yahoo Finance for the specified period
sgdusd_data = yf.download('SGDUSD=X', start='2015-11-01', end='2020-03-01')
gbpusd_data = yf.download('GBPUSD=X', start='2015-11-01', end='2020-03-01')


# Extract the adjusted close price
sgdusd_close = sgdusd_data['Adj Close']
gbpusd_close = gbpusd_data['Adj Close']

# Combine into a DataFrame
data = pd.concat([sgdusd_close, gbpusd_close], axis=1)
data.columns = ['SGDUSD', 'GBPUSD']

# Run the spread analysis
analyze_spread_LS(data)
