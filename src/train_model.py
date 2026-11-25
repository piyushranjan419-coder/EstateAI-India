import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/cleaned_housing.csv")

# Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict house prices
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display results
print("\nModel Evaluation Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
import joblib

# Save trained model
joblib.dump(model, "models/house_price_model.pkl")

print("\nModel saved successfully!")