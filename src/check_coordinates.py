
import pandas as pd

df = pd.read_csv("data/prepared_indian_housing.csv")

print("LONGITUDE Summary:")
print(df["LONGITUDE"].describe())

print("\nLATITUDE Summary:")
print(df["LATITUDE"].describe())

print("\nCoordinate Samples:")
print(df[["CITY", "LONGITUDE", "LATITUDE"]].head(10))

print("\nInvalid Coordinate Counts:")
print("Longitude:", ((df["LONGITUDE"] < 60) | (df["LONGITUDE"] > 100)).sum())
print("Latitude:", ((df["LATITUDE"] < 5) | (df["LATITUDE"] > 40)).sum())
