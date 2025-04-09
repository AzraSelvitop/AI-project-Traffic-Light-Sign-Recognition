import pandas as pd

# Path to the GT file
gt_file = 'C:/GTSRB/Test/GT/GT-final_test.csv'

# Load the CSV file
gt_data = pd.read_csv(gt_file, sep=';')

# Display the first few rows
print(gt_data.head())
