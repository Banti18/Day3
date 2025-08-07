import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. Load and Preprocess the Dataset
# ==============================================================================

# Load the Housing.csv dataset into a pandas DataFrame
df = pd.read_csv('Housing.csv')

# Display the first few rows to understand the data
print("Original Dataset Head:")
print(df.head())
print("-" * 50)

# ==============================================================================
# 2. Simple Linear Regression (Price vs. Area)
# ==============================================================================

print("Performing Simple Linear Regression (Price vs. Area)...")

# Define the independent variable (feature) and the dependent variable (target)
X_simple = df[['area']]
y_simple = df['price']

# Split the data into training (80%) and testing (20%) sets
# random_state ensures reproducibility
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Create a Linear Regression model instance
model_simple = LinearRegression()

# Train the model using the training data
model_simple.fit(X_train_simple, y_train_simple)

# Use the trained model to make predictions on the test set
y_pred_simple = model_simple.predict(X_test_simple)

# Evaluate the model's performance
mae_simple = mean_absolute_error(y_test_simple, y_pred_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

print("\nSimple Linear Regression Model Evaluation:")
print(f"  - Coefficient (m): {model_simple.coef_[0]:.2f}")
print(f"  - Intercept (b): {model_simple.intercept_:.2f}")
print(f"  - Equation: Price = {model_simple.coef_[0]:.2f} * Area + {model_simple.intercept_:.2f}")
print(f"  - Mean Absolute Error (MAE): {mae_simple:,.2f}")
print(f"  - Mean Squared Error (MSE): {mse_simple:,.2f}")
print(f"  - R-squared ($R^2$): {r2_simple:.4f}")
print("-" * 50)

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Actual Prices')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Price vs. Area')
plt.xlabel('Area (in square feet)')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# 3. Multiple Linear Regression (All Features)
# ==============================================================================

print("Performing Multiple Linear Regression (All Features)...")

# Convert categorical variables into dummy/one-hot encoded variables
# drop_first=True avoids multicollinearity
df_dummies = pd.get_dummies(df, drop_first=True)

# Define the independent variables (features) and the dependent variable (target)
# 'price' is the target, so we drop it from the features
X_multiple = df_dummies.drop('price', axis=1)
y_multiple = df_dummies['price']

# Split the data into training and testing sets
X_train_multiple, X_test_multiple, y_train_multiple, y_test_multiple = train_test_split(
    X_multiple, y_multiple, test_size=0.2, random_state=42
)

# Create and train the Multiple Linear Regression model
model_multiple = LinearRegression()
model_multiple.fit(X_train_multiple, y_train_multiple)

# Make predictions on the test set
y_pred_multiple = model_multiple.predict(X_test_multiple)

# Evaluate the model's performance
mae_multiple = mean_absolute_error(y_test_multiple, y_pred_multiple)
mse_multiple = mean_squared_error(y_test_multiple, y_pred_multiple)
r2_multiple = r2_score(y_test_multiple, y_pred_multiple)

print("\nMultiple Linear Regression Model Evaluation:")
print(f"  - Mean Absolute Error (MAE): {mae_multiple:,.2f}")
print(f"  - Mean Squared Error (MSE): {mse_multiple:,.2f}")
print(f"  - R-squared ($R^2$): {r2_multiple:.4f}")
print("-" * 50)

# Get and display the coefficients for each feature
coefficients_df = pd.DataFrame({
    'Feature': X_multiple.columns,
    'Coefficient': model_multiple.coef_
})

print("\nMultiple Linear Regression Coefficients:")
print(coefficients_df.sort_values(by='Coefficient', ascending=False).to_string())
print("-" * 50)

