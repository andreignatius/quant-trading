import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fetching data from Yahoo Finance
sgdusd_data = yf.download('SGDUSD=X', start='2016-01-01', end='2024-05-29')
gbpusd_data = yf.download('GBPUSD=X', start='2016-01-01', end='2024-05-29')

# Extract the adjusted close price
sgdusd_close = sgdusd_data['Adj Close']
gbpusd_close = gbpusd_data['Adj Close']

# Engle-Granger cointegration test
coint_t, p_value, _ = coint(sgdusd_close, gbpusd_close)

# Set default p-value for determining cointegration
default_p_value = 0.01

# Check if p-value is less than the default p-value
is_cointegrated = p_value < default_p_value

# Step 1: Fit a linear model
X = sm.add_constant(sgdusd_close)
model = OLS(gbpusd_close, X).fit()
residuals = model.resid

# Step 2: Perform unit root tests on residuals
adf_result = adfuller(residuals)

# Step 3: Calculate summary statistics
diff_sgdusd = sgdusd_close.diff().dropna()
diff_gbpusd = gbpusd_close.diff().dropna()
sd_diff_sgdusd = np.std(diff_sgdusd)
sd_diff_gbpusd = np.std(diff_gbpusd)
sd_residuals = np.std(residuals)
innovations = residuals.diff().dropna()
sd_innovations = np.std(innovations)

# Step 4: Estimate half-life of mean-reverting process
rho = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
half_life = -np.log(2) / np.log(abs(rho))

# Plot and save price series
fig, ax1 = plt.subplots(figsize=(10, 7))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('SGDUSD Close', color=color)
ax1.plot(sgdusd_close, color=color, label='SGDUSD Close')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('GBPUSD Close', color=color)  
ax2.plot(gbpusd_close, color=color, label='GBPUSD Close')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.tight_layout(pad=3.0)  # Increase the padding around the titles
plt.title('Price Series')
plt.savefig('X9_price_series.png')
# plt.show()
plt.close()

# Plot and save residual series
plt.figure(figsize=(10, 7))
plt.plot(residuals, label='Residuals', color='green')
plt.axhline(y=np.mean(residuals), color='red', linestyle='--', label='Mean')
plt.title(f'Residual Series\nADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.savefig('X9_residual_series.png')
# plt.show()
plt.close()

# Plot and save innovations
plt.figure(figsize=(10, 7))
plt.plot(innovations, label='Innovations', color='purple')
plt.title('Innovations')
plt.xlabel('Date')
plt.ylabel('Innovations')
plt.legend()
plt.savefig('X9_innovations.png')
# plt.show()
plt.close()

# Print results
print(f'Cointegration test statistic: {coint_t}')
print(f'p-value: {p_value}')
print(f'Are the series cointegrated? {"Yes" if is_cointegrated else "No"}')
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value (ADF): {adf_result[1]}')
print(f'Standard Deviation of diff(SGDUSD): {sd_diff_sgdusd}')
print(f'Standard Deviation of diff(GBPUSD): {sd_diff_gbpusd}')
print(f'Standard Deviation of residuals: {sd_residuals}')
print(f'Standard Deviation of innovations: {sd_innovations}')
print(f'Half-life: {half_life}')
