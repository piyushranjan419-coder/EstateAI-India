
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/final_indian_housing.csv")

X = df.drop(columns=["TARGET(PRICE_IN_LACS)"])
y = df["TARGET(PRICE_IN_LACS)"]

cat_cols = ["POSTED_BY", "BHK_OR_RK", "AREA", "CITY"]

num_cols = [
    "UNDER_CONSTRUCTION", "RERA", "BHK_NO.",
    "SQUARE_FT", "READY_TO_MOVE", "RESALE"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ExtraTreesRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Extra Trees Training Started...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nExtra Trees Training Completed!")
print("MAE:", round(mae, 2), "Lakhs")
print("R2 Score:", round(r2 * 100, 2), "%")

joblib.dump(model, "models/extra_trees_model.pkl")

print("\nExtra Trees Model Saved Successfully!")
