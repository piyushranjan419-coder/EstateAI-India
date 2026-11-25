
import pandas as pd

df = pd.read_csv("data/train.csv")

print("Total Records:", len(df))

print("\nArea Below 100 sq.ft.:")
print((df["SQUARE_FT"] < 100).sum())

print("\nArea Above 10,000 sq.ft.:")
print((df["SQUARE_FT"] > 10000).sum())

print("\nPrice Above 5,000 Lakhs:")
print((df["TARGET(PRICE_IN_LACS)"] > 5000).sum())

print("\nBHK Above 10:")
print((df["BHK_NO."] > 10).sum())

print("\nRecords with Area between 100 and 10,000:")
print(df["SQUARE_FT"].between(100, 10000).sum())
