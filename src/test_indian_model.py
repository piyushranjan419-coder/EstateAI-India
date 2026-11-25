
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

df = pd.read_csv("data/final_indian_housing.csv")

X = df.drop(columns=["TARGET(PRICE_IN_LACS)"])
y = df["TARGET(PRICE_IN_LACS)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load("models/extra_trees_model.pkl")

sample = X_test.iloc[[0]]
actual = y_test.iloc[0]
prediction = model.predict(sample)[0]

print("City:", sample["CITY"].iloc[0])
print("Area:", sample["AREA"].iloc[0])
print("Actual Price:", round(actual, 2), "Lakhs")
print("Predicted Price:", round(prediction, 2), "Lakhs")
print("Absolute Error:", round(abs(actual - prediction), 2), "Lakhs")
