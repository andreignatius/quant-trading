import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_correlation_matrix(forex_pairs, start_date, end_date, interval='1d'):
    # Create a DataFrame to store the adjusted close prices of all forex pairs
    data = pd.DataFrame()

    # Download the adjusted close prices for each forex pair
    for forex in forex_pairs:
        try:
            data[forex] = yf.Ticker(forex).history(start=start_date, end=end_date, interval=interval)["Close"]
        except Exception as e:
            print(f"Error downloading data for {forex}: {e}")
            data[forex] = pd.Series(dtype='float64')

    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    return correlation_matrix

# Usage
forex_pairs = [
    "USDJPY=X",  # US Dollar / Japanese Yen
    "USDEUR=X",  # US DOllar / Euro
    "USDGBP=X",  # US Dollar / British Pound
    "USDAUD=X",  # US Dollar / Australian Dollar
    "USDNZD=X",  # US Dollar / New Zealand Dollar
    "USDCAD=X",  # US Dollar / Canadian Dollar
    "USDCHF=X",  # US Dollar / Swiss Franc
    "USDSGD=X",  # US Dollar / Singapore Dollar
    "USDMXN=X",  # US Dollar / Mexican Peso
    "USDZAR=X",  # US Dollar / South African Rand
    "USDSEK=X",  # US Dollar / Swedish Krona
    "USDHKD=X",  # US Dollar / Hong Kong Dollar
    "USDNOK=X",  # US Dollar / Norwegian Krone
    "USDTRY=X",  # US Dollar / Turkish Lira
    "USDRUB=X",  # US Dollar / Russian Ruble
    "USDINR=X",  # US Dollar / Indian Rupee
    "USDKRW=X",  # US Dollar / South Korean Won
    "USDPHP=X",  # US Dollar / Philippine Peso
    "USDTHB=X",  # US Dollar / Thai Baht
    "USDMYR=X",  # US Dollar / Malaysian Ringgit
    "USDIDR=X",  # US Dollar / Indonesian Rupiah
]

start_date = '2016-01-01'
end_date = '2023-12-31'
correlation_matrix = generate_correlation_matrix(forex_pairs, start_date, end_date)

# Exclude correlations between each forex pair and itself
correlation_matrix = correlation_matrix.mask(pd.DataFrame(np.eye(len(correlation_matrix), dtype=bool), correlation_matrix.index, correlation_matrix.columns))

# Get the top 5 correlations
top_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()[:5]

# Generate a DataFrame from the top correlations
top_corr_df = pd.DataFrame(top_corr, columns=['Correlation'])

# Get the currency pair names and replace "=" with "/"
currency_pairs = [pair[0].replace("=", "/") + " " + pair[1].replace("=", "/") for pair in top_corr_df.index]

# Add currency pair names to the DataFrame
top_corr_df['Currency Pair'] = currency_pairs
top_corr_df = top_corr_df[['Currency Pair', 'Correlation']]


plt.figure(figsize=(5.5, 6))  # Reduce the width of the figure
table = plt.table(cellText=top_corr_df.round(3).values,  # Round to 5 decimal places
                  colLabels=top_corr_df.columns,
                  loc='center',
                  cellLoc='center',  # Center the cell text
                  colColours=['darkblue', 'darkblue'],  # Header color
                  cellColours=[['white']*len(top_corr_df.columns)]*len(top_corr_df),  # Cell color
                  )
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)  # Scale down the table horizontally
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_text_props(color='white')
plt.axis('off')

# Align text vertically to center
for key, cell in table._cells.items():
    cell._text.set_verticalalignment('center')

plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8)  # Adjust margins for full visibility

# Save the table as an image
plt.savefig('X2_top_correlation_table.png')
plt.show()
