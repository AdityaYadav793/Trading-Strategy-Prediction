import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, scale
import talib as ta

def get_train_and_test_data(data, trainDataRatio=0.9):
    data = data.sample(frac=1)
    train_size = int(round(data.shape[0]*trainDataRatio))
    data_label = data["open"] < data["close"]
    data_label = np.where(data_label, 1, 0)
    y_train = data_label[:train_size]
    y_test = data_label[train_size:]
    
    train_data = data[:train_size]
    test_data = data[train_size:]
    X_train = pd.DataFrame({"volume": train_data["volume"], "SMA": sma(train_data), "EMA": ema(train_data), "RSI": rsi(train_data)})
    X_train["upBand"], X_train["midBand"], X_train["lowBand"] = bbands(train_data)
    X_test = pd.DataFrame({"volume": test_data["volume"], "SMA": sma(test_data), "EMA": ema(test_data), "RSI": rsi(test_data)})
    X_test["upBand"], X_test["midBand"], X_test["lowBand"] = bbands(test_data)

    return X_train, y_train, X_test, y_test

def normalize(X_train, X_test):
    scaler = MinMaxScaler()
    smoothing_window_size = 150
    dt = np.array(X_train["mid"]).reshape((900, 1))
    for di in range(0, 900, smoothing_window_size):
        scaler.fit(dt[di:di+smoothing_window_size, :])
        dt[di:di+smoothing_window_size, :] = scaler.transform(dt[di:di+smoothing_window_size, :])
    X_train.loc[:, "mid"] = dt[:, 0]
    X_test.loc[:, "mid"] = scaler.transform(X_test["mid"].values.reshape((-1, 1))).reshape(-1)
    return X_train, X_test


def sma(data, timePeriod=40):
    return ta.SMA(data["open"], timePeriod)
    
def ema(data, timePeriod=40):
    return ta.EMA(data["open"], timePeriod)
    
def bbands(data, timePeriod=20):
    return ta.BBANDS(data["open"], timePeriod)

def rsi(data, timePeriod=14):
    return ta.RSI(data["open"], timePeriod)


def train(X_train, y_train, estimators=100, max_depth=16, learning_rate=1):
    model = xgb.XGBClassifier(n_estimators=estimators, max_depth=max_depth, learning_rate=learning_rate, objective="binary:logistic")
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    pred = model.predict(X_test)
    return pred


trainDataRatio = 0.9
featuresToUse = ["open", "high", "low", "close", "volume"]

data = pd.read_csv("data.csv", usecols=featuresToUse)
data = data.astype(float)
N = data.shape[0]

X_train, y_train, X_test, y_test = get_train_and_test_data(data, trainDataRatio)
for col in X_train.columns:
    X_train.loc[:, col] = scale(X_train[col])
    X_test.loc[:, col] = scale(X_test[col])
    
    
num_estimators = [5, 10, 20, 50, 100]
depth = [2, 4, 8, 16, 32]
best_model = None
hyperparams = [0, 0, 0]
best_acc = 0.0
for est in num_estimators:
    for d in depth:
        for lr in range(1, 25):
            model = train(X_train, y_train, max_depth=d, estimators=est, learning_rate=lr/4)
            acc = np.mean(predict(model, X_test) == y_test)
            if acc > best_acc:
                best_acc = acc
                best_model = model
                hyperparams = [est, d, lr/4]
                
                
print("Training accuracy =", np.mean(best_model.predict(X_train) == y_train))
print("Test accuracy =", np.mean(best_model.predict(X_test) == y_test))
print(hyperparams)