import pandas as pd

# Load the Titanic dataset
data = pd.read_csv("dataset/titanic.csv")

# Show the first rows of the dataset
print("\nFirst rows of the dataset:")
print(data.head())

# Show the shape of the dataset (rows and columns)
print("\nDataset shape:")
print(data.shape)

# Show column names
print("\nColumns in the dataset:")
print(data.columns)

# Show general information about the dataset
print("\nDataset information:")
print(data.info())

# Show statistical summary
print("\nStatistical summary:")
print(data.describe())