## Top 3 soybean producing countries
## https://fas.usda.gov/data/production/commodity/2222000
## Brazil, USA, Argentina

## Top 3 Corn producing countries
## https://www.investopedia.com/articles/markets-economy/090316/6-countries-produce-most-corn.asp
## USA, China, Brazil

## Top 3 Wheat producing countries
## https://www.worldatlas.com/articles/top-wheat-producing-countries.html
## China, India, Russia

## Extract forex data and save as CSV
import yfinance as yf
import pandas as pd

# Define the currencies
currencies = ['BRLUSD=X', 'ARSUSD=X', 'CNYUSD=X', 'INRUSD=X', 'RUBUSD=X']

# Fetch data from Yahoo Finance
data = yf.download(currencies, start="2019-05-22", end="2024-05-21")

# Extract closing prices
closing_prices = data['Adj Close'].reset_index()

# Save as CSV
closing_prices.to_csv('fx_rates.csv', index=False)

print("CSV file saved successfully.")
