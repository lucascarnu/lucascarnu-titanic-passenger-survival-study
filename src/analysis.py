import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    # Load the Titanic dataset
    return pd.read_csv(file_path)


def validate_columns(data, required_columns):
    # Validate that required columns exist in the dataset
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def prepare_data(data):
    # Handle missing values
    data["age"] = data["age"].fillna(data["age"].median())
    data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])
    data["embark_town"] = data["embark_town"].fillna(data["embark_town"].mode()[0])

    return data


def explore_data(data):
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


def analyze_data(data):
    # Passengers younger than 10
    print("\nPassengers younger than 10:")
    print(data[data["age"] <= 10].head())

    # Survival rate by gender
    print("\nSurvival by gender:")
    print(data.groupby("sex")["survived"].mean())

    # Survival rate by passenger class
    print("\nSurvival by class:")
    print(data.groupby("pclass")["survived"].mean())

    # Survival rate by age group
    print("\nSurvival rate by age group:")

    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Young Adult", "Adult", "Senior"]
    )

    print(data.groupby("age_group")["survived"].mean())


def create_visualizations(data):
    # Create output folder for images
    os.makedirs("images", exist_ok=True)

    # Survival rate by gender
    print("\nPlotting survival rate by gender...")

    data.groupby("sex")["survived"].mean().plot(kind="bar")
    plt.title("Survival Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig("images/survival_rate_by_gender.png")
    plt.show()
    plt.close()

    # Survival rate by passenger class
    print("\nPlotting survival rate by passenger class...")

    data.groupby("pclass")["survived"].mean().plot(kind="bar")
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.tight_layout()
    plt.savefig("images/survival_rate_by_class.png")
    plt.show()
    plt.close()

    # Age distribution
    print("\nPlotting age distribution...")

    data["age"].hist(bins=20)
    plt.title("Age Distribution of Passengers")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig("images/age_distribution.png")
    plt.show()
    plt.close()


def main():
    file_path = "dataset/titanic.csv"
    required_columns = [
        "survived", "pclass", "sex", "age",
        "embarked", "embark_town"
    ]

    data = load_data(file_path)
    validate_columns(data, required_columns)
    data = prepare_data(data)
    explore_data(data)
    analyze_data(data)
    create_visualizations(data)


if __name__ == "__main__":
    main()
