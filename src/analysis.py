import pandas as pd

# Load the Titanic dataset
data = pd.read_csv("dataset/titanic.csv")

# Display the first rows of the dataset
print(data.head())

# Show general information about the dataset
print(data.info())

# Show basic statistical summary
print(data.describe())