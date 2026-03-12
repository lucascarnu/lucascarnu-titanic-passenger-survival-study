import pandas as pd
import matplotlib.pyplot as plt

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
print(data[data["age"] <= 10].head())

# Survival rate by gender
print("\nSurvival by gender:")
print(data.groupby("sex")["survived"].mean())

# Survival rate by passenger class
print("\nSurvival by class:")
print(data.groupby("pclass")["survived"].mean())

# Survival rate by gender (visualization)
print("\nPlotting survival rate by gender...")

data.groupby("sex")["survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.show()


# Survival rate by passenger class (visualization)
print("\nPlotting survival rate by passenger class...")

data.groupby("pclass")["survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()


# Age distribution of passengers
print("\nPlotting age distribution...")

data["age"].hist(bins=20)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# Survival rate by age group
print("\nSurvival rate by age group:")

data["age_group"] = pd.cut(
    data["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
)

print(data.groupby("age_group")["survived"].mean())