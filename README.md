# 🥂 Waiter Tipping Prediction 🍽️

Welcome to the Waiter Tipping Prediction project! This project leverages machine learning to predict the tip amount based on various aspects of a dining experience. Using a Decision Tree Regressor, this model helps estimate a reasonable tip amount given user ratings on food quality, service, ambiance, wait time, and the total bill amount.

## 🌟 Project Overview

The main objective of this project is to build a predictive model that can estimate the tip amount a customer would leave at a restaurant. This can be useful for:
- Restaurant staff to understand tipping patterns.
- Customers to get an idea of how much to tip based on their experience.

## 📋 Features

- **Food Quality**: Rating of the food quality (1-5)
- **Service Quality**: Rating of the service quality (1-5)
- **Ambiance**: Rating of the restaurant's ambiance (1-5)
- **Wait Time**: Rating of the wait time (1-5)
- **Price**: Total bill amount ($)

## 🛠️ Installation

To run this project, you'll need to have the following libraries installed:

- `scikit-learn`
- `numpy`
- `matplotlib`

You can install these libraries using pip:

```bash
pip install scikit-learn numpy matplotlib
```

## 🚀 Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/waiter-tipping-prediction.git
   cd waiter-tipping-prediction
   ```

2. **Open the Jupyter Notebook**:
   Open `Waiter Tipping Problem.ipynb` in Jupyter Notebook or JupyterLab.

3. **Run the Notebook**:
   Execute the cells in the notebook to train the model and make predictions based on user input.

## 📊 Example Code

Here's a brief overview of the code used in this project:

```python
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# Define the features and target (tip amount)
features = [
    ['food_quality', 'service_quality', 'ambiance', 'wait_time', 'price'],
    [[4, 5, 3, 2, 50], [3, 4, 2, 3, 40], [5, 3, 4, 1, 60]],  # Example feedback/ratings
]

target = [10, 15, 8]  # Example tip amount for each set of feedback/ratings

# Convert features and target into numpy arrays
X = np.array(features[1])
y = np.array(target)

# Create a decision tree regressor model
model = DecisionTreeRegressor()

# Train the model
model.fit(X, y)

# Get user input for feedback/ratings
food_quality = int(input("Rate the food quality (1-5): "))
service_quality = int(input("Rate the service quality (1-5): "))
ambiance = int(input("Rate the ambiance (1-5): "))
wait_time = int(input("Rate the wait time (1-5): "))
price = float(input("Enter the total bill amount: $"))

# Make predictions based on user input
user_input = np.array([food_quality, service_quality, ambiance, wait_time, price]).reshape(1, -1)
predicted_tip = model.predict(user_input)

# Plot feature importance as a line chart
plt.figure(figsize=(16, 6))

# Plot feature importance as a line chart
plt.subplot(1, 3, 1)
plt.plot(features[0], model.feature_importances_, marker='o')
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")

# Visualize user ratings as a line chart
plt.subplot(1, 3, 2)
plt.plot(features[0][:4], user_input[0][:4], marker='o', color='blue', label='Ratings')
plt.xlabel('Features')
plt.ylabel('Rating')
plt.title('User Ratings')
plt.legend()
plt.grid(True)

# Plot weighted contributions as a column chart
plt.subplot(1, 3, 3)
plt.bar(features[0], user_input[0] * model.feature_importances_)
plt.title("Weighted Contributions")
plt.xlabel("Feature")
plt.ylabel("Weighted Value")

plt.tight_layout()
plt.show()

# Check if the predicted tip is non-negative
if predicted_tip[0] >= 0:
    print(f"Predicted Tip Amount: ${predicted_tip[0]:.2f}")
else:
    print("Invalid input. Predicted tip is negative.")
```

## 🤝 Contributing

Contributions are welcome! If you have any improvements or suggestions, please feel free to open an issue or submit a pull request.

## 📬 Contact

If you have any questions, feel free to reach out:

- Email: anjalivaish343@gmail.com
- GitHub: [anjali-vaish](https://github.com/anjali-vaish)


