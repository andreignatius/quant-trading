import subprocess

# Define the paths to your Python files
file_paths = ["1_correlation_matrix.py",
              "2_correlation_table.py",
              "3_cointegration_v4.py",
              "4_cointegration_table_v2.py",
              "5_fx_eda.py",]

# Loop through each file and execute it
for file_path in file_paths:
    subprocess.run(["python", file_path])
