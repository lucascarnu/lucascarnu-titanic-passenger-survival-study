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
data.info()

# Show statistical summary
print("\nStatistical summary:")
print(data.describe())

# Check missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Check duplicated rows
print("\nDuplicated rows:")
print(data.duplicated().sum())

# Passengers younger than 10
print("\nPassengers younger than 10:")
print(data[data["Age"] <= 10].head())

# Survival rate by gender
print("\nSurvival by gender:")
print(data.groupby("Sex")["Survived"].mean())

# Survival rate by passenger class
print("\nSurvival by class:")
print(data.groupby("Pclass")["Survived"].mean())