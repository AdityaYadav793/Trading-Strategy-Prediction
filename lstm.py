import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


trainDataRatio = 0.9
featuresToUse = ["timestamp", "open", "high", "low", "close", "trades", "volume", "vwap"]

# Loading dataset
data = pd.read_csv("data.csv", usecols=featuresToUse)
data = data.loc[::-1].reset_index(drop=True)
N = data.shape[0]
data["mid"] = (data["low"]+data["high"])/2
X_train = data.loc[:N*trainDataRatio]
X_test = data.loc[N*trainDataRatio:]

# Normalizing
scaler = MinMaxScaler()
smoothing_window_size = 150
dt = X_train["mid"].values.reshape((-1, 1))
for di in range(0, 900, smoothing_window_size):
    scaler.fit(dt[di:di+smoothing_window_size, :])
    dt[di:di+smoothing_window_size, :] = scaler.transform(dt[di:di+smoothing_window_size, :])
X_train.loc[:, "mid"] = dt.reshape(-1)
X_test.loc[:, "mid"] = scaler.transform(X_test["mid"].values.reshape((-1, 1))).reshape(-1)

# Smoothening
EMA = 0.0
gamma = 0.1
for ti in range(900):
    EMA = gamma*X_train["mid"][ti] + (1-gamma)*EMA
    X_train["mid"][ti]
    X_train.loc[ti, "mid"] = EMA

# Average prediction
def std_avg():
    window = 10
    pred = []
    mse_error = []

    for i in range(window, X_train.shape[0]):
        pred.append(np.mean(X_train.loc[i-window:i, "mid"]))
        mse_error.append((X_train["mid"][i] - pred[-1])**2)
        
    print("Total mse error =", np.sum(mse_error))
    print("Average mse erros =", np.mean(mse_error))
    plt.plot(range(X_train["mid"].shape[0]), X_train["mid"], label="True")
    plt.plot(range(window, X_train.shape[0]), pred, label="Predicted")
    plt.legend()
    plt.show()

