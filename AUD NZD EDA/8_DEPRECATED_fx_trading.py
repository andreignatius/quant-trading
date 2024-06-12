import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Fetching data from Yahoo Finance
sgdusd_data = yf.download('SGDUSD=X', start='2016-01-01', end='2024-05-29')
gbpusd_data = yf.download('GBPUSD=X', start='2016-01-01', end='2024-05-29')

# Extracting Close prices for the last 1 year starting from 2017
sgdusd_close = sgdusd_data['Close']['2017-05-29':'2024-05-29']  # 1 year from 2017-05-29 to 2018-05-29
gbpusd_close = gbpusd_data['Close']['2017-05-29':'2024-05-29']

# Set the lookback period for standard scaling to 100 days
lookback_period = 200

# Standard Scaling
scaler = StandardScaler()
sgdusd_scaled = scaler.fit_transform(sgdusd_close.values.reshape(-1, 1))
gbpusd_scaled = scaler.fit_transform(gbpusd_close.values.reshape(-1, 1))

# Convert the scaled arrays back to pandas Series
sgdusd_scaled_series = pd.Series(sgdusd_scaled.flatten(), index=sgdusd_close.index)
gbpusd_scaled_series = pd.Series(gbpusd_scaled.flatten(), index=gbpusd_close.index)

# Create a DataFrame for original currency rates and scaled values
scaled_df = pd.DataFrame({
    'SGDUSD_Close': sgdusd_close,
    'GBPUSD_Close': gbpusd_close,
    'Scaled_SGDUSD': sgdusd_scaled_series,
    'Scaled_GBPUSD': gbpusd_scaled_series
})

# Create another column for the difference between scaled SGDUSD and GBPUSD
scaled_df['scaled_residuals'] = scaled_df['Scaled_SGDUSD'] - scaled_df['Scaled_GBPUSD']

# Slice the DataFrame to include data from the lookback period onwards
scaled_df = scaled_df.iloc[lookback_period:]

# Initialize variables
initial_capital = 50000  # Starting capital in USD
long_short_per_residual = 1000 / 0.1  # Amount to long/short per 0.1 of scaled residuals
position_size_sgdusd = 0  # Initial position size for SGDUSD
position_size_gbpusd = 0  # Initial position size for GBPUSD
cumulative_pnl = []

# Iterate through each day's scaled residuals
for index, row in scaled_df.iterrows():
    # Calculate long and short positions based on scaled residuals
    long_position = row['scaled_residuals'] * long_short_per_residual
    short_position = -row['scaled_residuals'] * long_short_per_residual
    
    # Calculate daily P&L for each currency position
    daily_pnl_sgdusd = position_size_sgdusd * (row['SGDUSD_Close'] - sgdusd_close.iloc[0])
    daily_pnl_gbpusd = position_size_gbpusd * (row['GBPUSD_Close'] - gbpusd_close.iloc[0])
    
    # Update cumulative P&L
    daily_total_pnl = daily_pnl_sgdusd + daily_pnl_gbpusd
    cumulative_pnl.append(daily_total_pnl)
    
    # Update position sizes for the next day (rebalancing)
    position_size_sgdusd = long_position
    position_size_gbpusd = short_position
    
    # Add daily positions to DataFrame
    scaled_df.loc[index, 'Position_Size_SGDUSD'] = position_size_sgdusd
    scaled_df.loc[index, 'Position_Size_GBPUSD'] = position_size_gbpusd
    scaled_df.loc[index, 'PnL'] = daily_total_pnl  # Add P&L for the day to DataFrame

# Plot scaled residuals
plt.figure(figsize=(10, 6))
plt.plot(scaled_df.index, scaled_df['scaled_residuals'], color='blue', label='Scaled Residuals')
plt.title('Scaled Residuals (SGDUSD - GBPUSD) Over Time')
plt.xlabel('Date')
plt.ylabel('Scaled Residuals')
plt.legend()
plt.grid(True)
plt.savefig('X8_fx_trading_residuals.png')
plt.show()

# Plot cumulative P&L
plt.figure(figsize=(10, 6))
plt.plot(scaled_df.index, np.cumsum(cumulative_pnl), color='green', label='Cumulative P&L')
plt.title('Cumulative P&L Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative P&L (USD)')
plt.legend()
plt.grid(True)
plt.savefig('X8_fx_trading_pnl.png')
plt.show()

# Initialize variables for mark-to-market calculation
mark_to_market_sgdusd = 0
mark_to_market_gbpusd = 0
mark_to_market_total = []

# Iterate through each day's data
for index, row in scaled_df.iterrows():
    # Calculate mark-to-market for each currency position
    mark_to_market_sgdusd = position_size_sgdusd * (row['SGDUSD_Close'] - sgdusd_close.iloc[0])
    mark_to_market_gbpusd = position_size_gbpusd * (row['GBPUSD_Close'] - gbpusd_close.iloc[0])
    
    # Calculate total mark-to-market value
    mark_to_market_total.append(mark_to_market_sgdusd + mark_to_market_gbpusd)
    
# Add mark-to-market values to DataFrame
scaled_df['Mark_to_Market_SGDUSD'] = mark_to_market_sgdusd
scaled_df['Mark_to_Market_GBPUSD'] = mark_to_market_gbpusd
scaled_df['Mark_to_Market_Total'] = mark_to_market_total

# Plot mark-to-market
plt.figure(figsize=(10, 6))
plt.plot(scaled_df.index, scaled_df['Mark_to_Market_Total'], color='purple', label='Mark-to-Market Total')
plt.title('Mark-to-Market Total Over Time')
plt.xlabel('Date')
plt.ylabel('Mark-to-Market Total (USD)')
plt.legend()
plt.grid(True)
plt.savefig('X8_fx_trading_mtm.png')
plt.show()

# Save updated DataFrame to CSV file with P&L over time
scaled_df[['SGDUSD_Close', 'GBPUSD_Close', 'Scaled_SGDUSD', 'Scaled_GBPUSD', 'scaled_residuals', 
           'Position_Size_SGDUSD', 'Position_Size_GBPUSD', 'PnL', 'Mark_to_Market_SGDUSD', 
           'Mark_to_Market_GBPUSD', 'Mark_to_Market_Total']].to_csv('X8_fx_trading.csv')

# Display updated DataFrame
print(scaled_df)
