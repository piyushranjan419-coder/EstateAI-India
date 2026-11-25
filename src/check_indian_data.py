
import pandas as pd

df = pd.read_csv("data/train.csv")

print("Duplicate Rows:", df.duplicated().sum())

print("\nUnique Property Types:")
print(df["BHK_OR_RK"].value_counts())

print("\nUnique Posted By:")
print(df["POSTED_BY"].value_counts())

print("\nArea (Square Feet):")
print(df["SQUARE_FT"].describe())

print("\nPrice (Lakhs):")
print(df["TARGET(PRICE_IN_LACS)"].describe())

print("\nCity Count:")
print(df["ADDRESS"].str.split(",").str[-1].str.strip().value_counts())
