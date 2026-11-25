
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/california_housing.csv")

# Create histogram
plt.figure(figsize=(8, 5))
plt.hist(df["MedHouseVal"], bins=30)

# Add labels and title
plt.xlabel("House Price ($100,000 units)")
plt.ylabel("Number of Houses")
plt.title("House Price Distribution")

# Display graph
plt.show()

# Create scatter plot
plt.figure(figsize=(8, 5))

plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.3)

# Add labels and title
plt.xlabel("Median Income")
plt.ylabel("House Price ($100,000 units)")
plt.title("House Income vs House Price")

# Display graph
plt.show()
