import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fetching data from Yahoo Finance
sgdusd_data = yf.download('SGDUSD=X', start='2016-01-01', end='2024-05-29')
gbpusd_data = yf.download('GBPUSD=X', start='2016-01-01', end='2024-05-29')

# Extract the adjusted close price
sgdusd_close = sgdusd_data['Adj Close']
gbpusd_close = gbpusd_data['Adj Close']

# Step 1: Fit a linear model
X = sm.add_constant(sgdusd_close)
model = OLS(gbpusd_close, X).fit()
residuals = model.resid

# Split residuals into train and test periods
train_residuals = residuals[:int(len(residuals) * 0.7)]
test_residuals = residuals[int(len(residuals) * 0.7):]

# Define function to perform ADF test and display results
def run_adf_test(residuals):
    result = adfuller(residuals)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

# Perform ADF test on train residuals
print("ADF test results for train period:")
run_adf_test(train_residuals)

# Perform ADF test on test residuals
print("\nADF test results for test period:")
run_adf_test(test_residuals)

# Plot and save residual series
plt.figure(figsize=(10, 7))
plt.plot(residuals, label='Residuals', color='green')
plt.axhline(y=np.mean(residuals), color='red', linestyle='--', label='Mean')
plt.title('Residual Series')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axvline(x=residuals.index[int(len(residuals) * 0.7)], color='blue', linestyle='--', label='Start of Testing Period')

# Perform ADF test on train residuals
train_adf_result = adfuller(train_residuals)
plt.text(residuals.index[int(len(residuals) * 0.3)], np.mean(residuals) + 2 * np.std(residuals), f"ADF Stat (Train): {train_adf_result[0]:.2f}\np-value (Train): {train_adf_result[1]:.4f}", fontsize=10, color='black', ha='center')

# Perform ADF test on test residuals
test_adf_result = adfuller(test_residuals)
plt.text(residuals.index[int(len(residuals) * 0.85)], np.mean(residuals) + 2 * np.std(residuals), f"ADF Stat (Test): {test_adf_result[0]:.2f}\np-value (Test): {test_adf_result[1]:.4f}", fontsize=10, color='black', ha='center')

plt.legend()
plt.savefig('X12_residuals_ADF_trainvtest_with_values.png')
plt.show()
