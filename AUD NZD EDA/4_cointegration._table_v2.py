NEED TO REDO AS PER 3_cointegration_v4.py

# import yfinance as yf
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.stattools import coint
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# def generate_cointegration_matrix(forex_pairs, start_date, end_date, interval='1d'):
#     # Create a DataFrame to store the adjusted close prices of all forex pairs
#     data = pd.DataFrame()

#     # Download the adjusted close prices for each forex pair
#     for forex in forex_pairs:
#         try:
#             data[forex] = yf.Ticker(forex).history(start=start_date, end=end_date, interval=interval)["Close"]
#         except Exception as e:
#             print(f"Error downloading data for {forex}: {e}")
#             data[forex] = pd.Series(dtype='float64')

#     # Replace infinite or missing values with NaN
#     data.replace([np.inf, -np.inf], np.nan, inplace=True)
#     data.dropna(inplace=True)

#     # Split data into train and test sets
#     train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)

#     # Calculate cointegration and p-values
#     cointegration_results = pd.DataFrame(index=forex_pairs, columns=forex_pairs)
#     p_values = pd.DataFrame(index=forex_pairs, columns=forex_pairs)
    
#     for i in range(len(forex_pairs)):
#         for j in range(i+1, len(forex_pairs)):
#             result = coint(train_data[forex_pairs[i]], train_data[forex_pairs[j]])
#             p_values.loc[forex_pairs[i], forex_pairs[j]] = result[1]
#             cointegration_results.loc[forex_pairs[i], forex_pairs[j]] = result[0]

#     # Stack the DataFrame to get pairs and their corresponding p-values and cointegration scores
#     p_values_stacked = p_values.stack().reset_index()
#     p_values_stacked.columns = ['Pair 1', 'Pair 2', 'P-Value']
    
#     cointegration_stacked = cointegration_results.stack().reset_index()
#     cointegration_stacked.columns = ['Pair 1', 'Pair 2', 'Cointegration']

#     # Merge the DataFrames
#     merged_results = pd.merge(p_values_stacked, cointegration_stacked, on=['Pair 1', 'Pair 2'])

#     # Round p-values and cointegration values to 3 decimal places
#     merged_results['P-Value'] = merged_results['P-Value'].map(lambda x: f"{float(x):.5f}")
#     merged_results['Cointegration'] = merged_results['Cointegration'].map(lambda x: f"{float(x):.3f}")

#     # Sort the DataFrame by p-values and select top 10
#     top_10_results = merged_results.sort_values(by='P-Value').head(10)

#     return top_10_results


# # Usage
# forex_pairs = [
#     "USDJPY=X",  # US Dollar / Japanese Yen
#     "USDEUR=X",  # US DOllar / Euro
#     "USDGBP=X",  # US Dollar / British Pound
#     "USDAUD=X",  # US Dollar / Australian Dollar
#     "USDNZD=X",  # US Dollar / New Zealand Dollar
#     "USDCAD=X",  # US Dollar / Canadian Dollar
#     "USDCHF=X",  # US Dollar / Swiss Franc
#     "USDSGD=X",  # US Dollar / Singapore Dollar
#     "USDMXN=X",  # US Dollar / Mexican Peso
#     "USDZAR=X",  # US Dollar / South African Rand
#     "USDSEK=X",  # US Dollar / Swedish Krona
#     "USDHKD=X",  # US Dollar / Hong Kong Dollar
#     "USDNOK=X",  # US Dollar / Norwegian Krone
#     "USDTRY=X",  # US Dollar / Turkish Lira
#     # "USDRUB=X",  # US Dollar / Russian Ruble
#     "USDINR=X",  # US Dollar / Indian Rupee
#     "USDKRW=X",  # US Dollar / South Korean Won
#     "USDPHP=X",  # US Dollar / Philippine Peso
#     "USDTHB=X",  # US Dollar / Thai Baht
#     # "USDMYR=X",  # US Dollar / Malaysian Ringgit
#     "USDIDR=X",  # US Dollar / Indonesian Rupiah
# ]

# start_date = '2016-01-01'
# end_date = '2023-12-31'
# top_10_results = generate_cointegration_matrix(forex_pairs, start_date, end_date)

# plt.figure(figsize=(7, 6))  # Adjust figure size if needed
# table = plt.table(cellText=top_10_results.values,
#                   colLabels=top_10_results.columns,
#                   loc='center',
#                   cellLoc='center',  # Center the cell text
#                   colColours=['darkblue']*len(top_10_results.columns),  # Header color
#                   cellColours=[['white']*len(top_10_results.columns)]*len(top_10_results),  # Cell color
#                   )
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.scale(1.2, 1.5)  # Scale down the table horizontally
# for key, cell in table.get_celld().items():
#     if key[0] == 0:
#         cell.set_text_props(color='white')
# plt.axis('off')

# # Align text vertically to center
# for key, cell in table._cells.items():
#     cell._text.set_verticalalignment('center')

# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)  # Adjust margins for full visibility

# # Save the table as an image
# plt.savefig('X4_Cointegration_table.png')
# plt.show()
