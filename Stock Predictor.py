# Stock Price Predictor using Linear Regression

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = yf.download("AAPL", start="2015-01-01", end="2023-12-31")

data = data[["Close"]]  
data["Prediction"] = data["Close"].shift(-30)  


X = np.array(data.drop(["Prediction"], axis=1))[:-30]
y = np.array(data["Prediction"])[:-30]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("--- Model Evaluation ---")
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
print("-" * 24)

plt.figure(figsize=(10,6))
plt.plot(y_test, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
plt.title("Stock Price Prediction (AAPL)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

future_data = np.array(data.drop(["Prediction"], axis=1))[-30:]
future_predictions = model.predict(future_data)
print("\n--- Next 30 Days Predicted Prices ---")
for i, price in enumerate(future_predictions):
    print(f"Day {i+1}: {price:.2f}")
print("-" * 35)