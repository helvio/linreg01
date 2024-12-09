# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Generate some sample data
# Create a simple linear relationship: y = 2X + 1
np.random.seed(42)  # For reproducibility
X = np.random.rand(100, 1) * 10  # 100 random points between 0 and 10
y = 2 * X + 1 + np.random.randn(100, 1)  # Add some noise to the data

# Step 3: Create a Linear Regression model
model = LinearRegression()

# Step 4: Fit the model to the data
model.fit(X, y)

# Step 5: Visualize the results
# Predicted values
y_pred = model.predict(X)

# Plotting the data and the regression line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.title('Linear Regression Example')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Output the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_}')
