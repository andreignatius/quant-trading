import pandas as pd

# Read CSV files into separate dataframes
corn_df = pd.read_csv('Corn_data.csv')
soybeans_df = pd.read_csv('Soybeans_data.csv')
wheat_df = pd.read_csv('Wheat_data.csv')

# Select the 'adj_close' column from each dataframe
date = corn_df['Date']
corn_adj_close = corn_df['Adj Close']
soybeans_adj_close = soybeans_df['Adj Close']
wheat_adj_close = wheat_df['Adj Close']

# Concatenate selected columns into a single dataframe
agri_commod_df = pd.concat([date,corn_adj_close, soybeans_adj_close, wheat_adj_close], axis=1)
agri_commod_df.columns = ['Date','Corn', 'Soybeans', 'Wheat']  # Rename columns
# print(agri_commod_df)

# Read CSV file
fx_df = pd.read_csv('fx_rates.csv')
# print(fx_df)

# Merge on Date
agri_fx_merged_df= pd.merge(agri_commod_df, fx_df, on='Date')


### PREPARE DATA FOR CORRELATION ANALYSIS
# Extract commodity columns
commodity_cols = agri_fx_merged_df.columns[1:4]

# Extract FX rate columns
fx_cols = agri_fx_merged_df.columns[4:]

# Create a new DataFrame with only commodity and FX rate columns
df_selected = agri_fx_merged_df[commodity_cols.tolist() + fx_cols.tolist()]

# Calculate correlation matrix for commodities and FX rates
correlation_matrix = df_selected.corr().loc[commodity_cols, fx_cols]

# Display the correlation matrix
# print(correlation_matrix)

### VISUALISE THE CORRELATION VIA A HEAT MAP
import seaborn as sns
import matplotlib.pyplot as plt

# Set the size of the plot
plt.figure(figsize=(10, 6))

# Plot heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Add title
plt.title('Correlation Heatmap of Commodities and FX Rates')

# Show plot
plt.show()
