#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("Advertising.csv")

# Remove index column if present
data = data.drop(columns=["Unnamed: 0"], errors="ignore")

# -----------------------------
# Display Dataset
# -----------------------------
print("\nFirst 5 Rows of Dataset")
print(data.head())

print("\nDataset Information")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())

print("\nStatistical Summary")
print(data.describe())

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(6,5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Prepare Features and Target
# -----------------------------
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()

model.fit(X_train, y_train)

# -----------------------------
# Make Predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("----------------------------")
print("MAE :", round(mae,2))
print("MSE :", round(mse,2))
print("RMSE:", round(rmse,2))
print("R2 Score:", round(r2,2))

# -----------------------------
# Actual vs Predicted
# -----------------------------
results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

print("\nActual vs Predicted")
print(results.head(10))

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

# -----------------------------
# Predict Future Sales
# -----------------------------
new_data = pd.DataFrame({
    "TV":[230],
    "Radio":[37],
    "Newspaper":[69]
})

future_prediction = model.predict(new_data)

print("\nPredicted Future Sales")
print("----------------------------")
print("Predicted Sales =", round(future_prediction[0],2))