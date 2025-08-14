📈 Stock Price Predictor (Python)

This project is a stock price prediction system using historical stock data and Linear Regression.
It fetches stock prices from Yahoo Finance (yfinance) and predicts the stock’s future closing prices.

🚀 Features

Downloads historical stock data (e.g., Apple, Google).

Uses closing prices as features for prediction.

Applies Linear Regression to learn trends.

Evaluates model performance using MSE (Mean Squared Error) and R² Score.

Predicts the next 30 days of stock prices.

Plots actual vs predicted prices.

🛠️ How It Works
1. Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


yfinance → fetch stock data.

pandas, numpy → handle datasets.

matplotlib → plot graphs.

scikit-learn → machine learning (Linear Regression, train/test split, evaluation).

2. Fetch Stock Data
data = yf.download("AAPL", start="2015-01-01", end="2023-12-31")


Downloads Apple stock (AAPL) between 2015 and 2023.

Dataset includes Open, High, Low, Close, Adj Close, and Volume.

We only use the Close price.

3. Prepare Data
data = data[["Close"]]  
data["Prediction"] = data["Close"].shift(-30)


Keep only the Close price column.

Create a target column (Prediction) by shifting prices 30 days into the future.

Example: Today’s price will be used to predict the price 30 days later.

4. Define Features & Target
X = np.array(data.drop(["Prediction"], axis=1))[:-30]
y = np.array(data["Prediction"])[:-30]


X → all historical closing prices (except last 30 days).

y → future closing prices (shifted 30 days ahead).

5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


Training set = 80% of data.

Testing set = 20% of data.

shuffle=False because stock data is time-series (order matters).

6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)


Fit a Linear Regression model to learn the relationship between today’s price and the future price.

7. Predictions & Evaluation
predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))


Model predicts stock prices on test data.

Evaluate with:

MSE → smaller is better (measures error).

R² Score → closer to 1 is better (explains variance).

8. Visualization
plt.figure(figsize=(10,6))
plt.plot(y_test, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
plt.title("Stock Price Prediction (AAPL)")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()


Plots actual vs predicted stock prices to visually compare performance.

9. Predict Future Prices
future_data = np.array(data.drop(["Prediction"], axis=1))[-30:]
future_predictions = model.predict(future_data)
print("Next 30 days predicted prices:\n", future_predictions)


Takes the last 30 days of real data.

Predicts the next 30 future stock prices.

📊 Example Output
MSE: 14.57
R2 Score: 0.92
Next 30 days predicted prices:
[171.34, 172.15, 170.89, ...]


📈 Plot will show a line graph of actual vs predicted prices.

📌 Dependencies

Install required libraries:

pip install yfinance pandas numpy scikit-learn matplotlib

✅ Future Improvements

Use LSTM (deep learning) for better time-series prediction.

Include more features (volume, open, high, low).

Hyperparameter tuning for accuracy.

Deploy as a web app with Flask/Streamlit.
