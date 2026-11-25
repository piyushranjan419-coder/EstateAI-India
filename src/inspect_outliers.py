
import pandas as pd

df = pd.read_csv("data/train.csv")

print("Top 10 Highest Area Properties:")
print(
    df.nlargest(10, "SQUARE_FT")[
        ["ADDRESS", "BHK_NO.", "SQUARE_FT", "TARGET(PRICE_IN_LACS)"]
    ].to_string(index=False)
)

print("\nTop 10 Highest Price Properties:")
print(
    df.nlargest(10, "TARGET(PRICE_IN_LACS)")[
        ["ADDRESS", "BHK_NO.", "SQUARE_FT", "TARGET(PRICE_IN_LACS)"]
    ].to_string(index=False)
)

print("\nProperties with Area Below 100 sq. ft.:")
print(
    df[df["SQUARE_FT"] < 100][
        ["ADDRESS", "BHK_NO.", "SQUARE_FT", "TARGET(PRICE_IN_LACS)"]
    ].to_string(index=False)
)
