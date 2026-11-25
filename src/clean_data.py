
import pandas as pd

# Load dataset
df = pd.read_csv("data/california_housing.csv")

# Check duplicate records
duplicates = df.duplicated().sum()

print("Total Records:", len(df))
print("Duplicate Records:", duplicates)
# Check data types
print("\nColumn Data Types:")
print(df.dtypes)

# Check outliers using IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5 * IQR)) |
            (df > (Q3 + 1.5 * IQR))).sum()

print("\nOutliers in Each Column:")
print(outliers)
# Save cleaned dataset
df.to_csv("data/cleaned_housing.csv", index=False)

print("\nCleaned dataset saved successfully!")