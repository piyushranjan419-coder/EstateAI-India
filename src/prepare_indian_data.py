
import pandas as pd

df = pd.read_csv("data/cleaned_indian_housing.csv")

# Extract area and city
df["AREA"] = df["ADDRESS"].str.rsplit(",", n=1).str[0].str.strip()
df["CITY"] = df["ADDRESS"].str.rsplit(",", n=1).str[-1].str.strip()

# Remove original address
df.drop(columns=["ADDRESS"], inplace=True)

print("Prepared Dataset Shape:", df.shape)

print("\nNew Columns:")
print(df.columns.tolist())

print("\nTop 10 Cities:")
print(df["CITY"].value_counts().head(10))

print("\nSample Data:")
print(df[["AREA", "CITY"]].head())

# Save prepared dataset
df.to_csv("data/prepared_indian_housing.csv", index=False)

print("\nPrepared Dataset Saved Successfully!")
