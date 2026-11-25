
import pandas as pd

df = pd.read_csv("data/train.csv")

print("Original Dataset:", df.shape)

# Remove unrealistic area and BHK values
df = df[
    (df["SQUARE_FT"] >= 100) &
    (df["SQUARE_FT"] <= 10000) &
    (df["BHK_NO."] <= 10) &
    (df["BHK_NO."] >= 1)
]

# Remove extremely high prices
df = df[
    (df["TARGET(PRICE_IN_LACS)"] > 0) &
    (df["TARGET(PRICE_IN_LACS)"] <= 5000)
]

# Save cleaned dataset
df.to_csv("data/cleaned_indian_housing.csv", index=False)

print("\nCleaned Dataset:", df.shape)
print("Remaining Missing Values:", df.isnull().sum().sum())

print("\nCleaned Dataset Saved Successfully!")
