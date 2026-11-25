
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/final_indian_housing.csv")

# Features and target
X = df.drop(columns=["TARGET(PRICE_IN_LACS)"])
y = df["TARGET(PRICE_IN_LACS)"]

# Categorical and numerical columns
cat_cols = ["POSTED_BY", "BHK_OR_RK", "AREA", "CITY"]

num_cols = [
    "UNDER_CONSTRUCTION", "RERA", "BHK_NO.",
    "SQUARE_FT", "READY_TO_MOVE", "RESALE"
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Started...")

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Training Completed!")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("MAE:", round(mae, 2), "Lakhs")
print("R2 Score:", round(r2 * 100, 2), "%")

# Save model
joblib.dump(model, "models/indian_house_price_model.pkl")

print("\nModel Saved Successfully!")
