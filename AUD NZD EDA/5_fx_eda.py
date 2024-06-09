import yfinance as yf
import matplotlib.pyplot as plt

# Fetching data for USD/ZAR and USD/NOK from Yahoo Finance
usdzar_data = yf.download('USDZAR=X', start='2016-01-01', end='2023-12-31')
usdnok_data = yf.download('USDNOK=X', start='2016-01-01', end='2023-12-31')

# Plotting
fig, ax1 = plt.subplots()

# Plot USD/ZAR on primary y-axis
ax1.plot(usdzar_data.index, usdzar_data['Close'], color='b', label='USD/ZAR')
ax1.set_ylabel('USD/ZAR', color='b')

# Create a secondary y-axis and plot USD/NOK
ax2 = ax1.twinx()
ax2.plot(usdnok_data.index, usdnok_data['Close'], color='r', label='USD/NOK')
ax2.set_ylabel('USD/NOK', color='r')

# Add legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# Set title
plt.title('USD/ZAR and USD/NOK Daily Close (2016-2023)')

# Show plot
plt.show()
