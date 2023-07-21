import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


trainDataRatio = 0.9
featuresToUse = ["timestamp", "open", "high", "low", "close", "trades", "volume", "vwap"]

# Loading dataset
data = pd.read_csv("data.csv", usecols=featuresToUse)
N = data.shape[0]
train_data_size = int(round(N*trainDataRatio))

# Normalizing
scaler = MinMaxScaler(feature_range=(0, 1))
smoothing_window_size = 300
dt = data["open"].values.reshape((-1, 1))
for di in range(0, N, smoothing_window_size):
    scaler.fit(dt[di:di+smoothing_window_size, :])
    dt[di:di+smoothing_window_size, :] = scaler.transform(dt[di:di+smoothing_window_size, :])
data.loc[:, "open"] = dt.reshape(-1)

# Smoothening
EMA = 0.0
gamma = 0.1
for ti in range(train_data_size):
    EMA = gamma*data["open"][ti] + (1-gamma)*EMA
    data["open"][ti]
    data.loc[ti, "open"] = EMA

##############################################################################################

# Average prediction
def std_avg():
    window = 10
    pred = []
    mse_error = []

    for i in range(window, N):
        pred.append(np.mean(data.loc[i-window:i, "open"]))
        mse_error.append((data["open"][i] - pred[-1])**2)
        
    print("Total mse error =", np.sum(mse_error))
    print("Average mse erros =", np.mean(mse_error))
    plt.plot(range(N), data["open"], label="True")
    plt.plot(range(window, N), pred, label="Predicted")
    plt.legend()
    plt.show()

# std_avg()

##############################################################################################

# Build Dataset
input_size = 20
X, y = [], []
for i in range(input_size, N):
    X.append(data.loc[i-input_size:i, "open"])
    y.append(data.loc[i, "close"])
X_train, X_test = X[:train_data_size], X[train_data_size:]
y_train, y_test = y[:train_data_size], y[train_data_size:]

# Building model
model = Sequential()
model.add(LSTM(10))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
print(model.summary())