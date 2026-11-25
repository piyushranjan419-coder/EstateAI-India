
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Download the house price dataset
housing = fetch_california_housing(as_frame=True)

# Convert dataset into DataFrame
df = housing.frame

# Save dataset as CSV
df.to_csv("data/california_housing.csv", index=False)

# Display results
print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())
