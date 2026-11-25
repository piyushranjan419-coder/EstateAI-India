
import pandas as pd

df = pd.read_csv("data/prepared_indian_housing.csv")

# Remove unreliable coordinate columns
df = df.drop(columns=["LONGITUDE", "LATITUDE"])

# Save final dataset
df.to_csv("data/final_indian_housing.csv", index=False)

print("Final Dataset Shape:", df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDataset Saved Successfully!")
