import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Generating a synthetic dataset for demonstration
np.random.seed(42)
data_size = 1000
area = np.random.normal(1500, 500, data_size)  # Area in square feet
bedrooms = np.random.randint(1, 6, data_size)  # Number of bedrooms
location = np.random.randint(1, 4, data_size)  # Location code (1, 2, or 3)
price = (area * 300) + (bedrooms * 50000) + (location * 10000) + np.random.normal(0, 10000, data_size)  # House price

# Creating a DataFrame
data = pd.DataFrame({'Area': area, 'Bedrooms': bedrooms, 'Location': location, 'Price': price})

# Splitting the dataset into features and target variable
X = data[['Area', 'Bedrooms', 'Location']]
y = data['Price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dictionary to store models and their names
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Performance:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R^2 Score: {r2_score(y_test, y_pred)}")
    print()

# Visualization of Actual vs Predicted Prices for Random Forest Regression
y_pred_rf = models['Random Forest Regression'].predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices - Random Forest Regression')
plt.show()
