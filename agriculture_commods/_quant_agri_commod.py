import subprocess

# Load the agri commod data
subprocess.run(["python", "agri_commod_data_load.py"])

# Visualize the agri commod data
subprocess.run(["python", "agri_commod_data_vis.py"]) # OPTIONAL

# Load the fx rate data
subprocess.run(["python", "fx_rate_data_load.py"])

# Visualize the fx rate data
subprocess.run(["python", "fx_rate_data_vis.py"]) # OPTIONAL

# Correlation between commod and agri
subprocess.run(["python", "agri_commod_fx_correl.py"])

# Cointegration between commod and agri
subprocess.run(["python", "agri_commod_fx_cointegrate.py"])

# Most variables (commodities and exchange rates) are non-stationary based on ADF tests
# Russian Ruble exchange rate is stationary (evidence of price manipulation?)
# No evidence of cointegration (long-run equilibrium relationship) between commodities and exchange rates
# Movements of commodities and exchange rates are not closely linked in the long run
# Deviations from current relationships are likely permanent rather than temporary