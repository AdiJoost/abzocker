import pandas as pd
import os
import numpy as np
from companySpliter import DATA_FOLDER, getStocksymbols
from keras.utils import timeseries_dataset_from_array
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Normalization, SimpleRNN, Input
from keras.callbacks import EarlyStopping

from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def main():
    # stockSymbols = getStockSymbols()
    datasets = prepareData(stockName="AAPL")
    trainModelRNN(datasets)
    
def trainModelRNN(datasets):
    X_train, y_train, X_test, y_test, X_val, y_val = datasets
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(SimpleRNN(32))
    model.add(Dense(1))

    
    earlyStopping = EarlyStopping(monitor="val_loss", patience=2)
    
    model.compile(loss=MeanSquaredError(),
                      optimizer=Adam(),
                      metrics=[MeanAbsoluteError()])
    
    model.fit(X_train,
                y_train,
                epochs=10,
                batch_size=64,
                validation_data=(X_test,y_test))
    
    # print("History")
    # # print(history)
    # val_performance = model.evaluate(X_val, y_val)
    # print("Val_performance")
    # print(val_performance)
    print(model.summary())
    
def getDataPath(stockName="AAPL"):
    cwd = os.getcwd()
    return os.path.join(cwd,DATA_FOLDER,f"{stockName}.pkl")
    
    
def prepareData(stockName="AAPL", DATA_FOLDER=""):
    filepath = getDataPath(stockName)  
    dataRaw = pd.read_pickle(filepath)
    dataRaw = cleanupData(dataRaw)
    return makeInputsAndTargets(dataRaw)
    
    
def cleanupData(data):
    data = data.drop(["Symbol"], axis=1)
    data["Date"] = (data["Date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    return data

def makeInputsAndTargets(data):
    timestepCount = len(data)
    featureLength = 5
    targetLength = 1
    
    totalWindowSize = featureLength + targetLength
    windowCount = timestepCount // totalWindowSize
    
    X_data = []
    y_data = []
    for i in range(0, windowCount):
        X_data.append(data.iloc[i*totalWindowSize:i*totalWindowSize + featureLength])
        y_data.append(data.iloc[i*totalWindowSize + featureLength]["Close"])

    X_train, y_train, X_test, y_test, X_val, y_val = trainTestValSplit(X_data, y_data)

    return X_train, y_train, X_test, y_test, X_val, y_val
    
def trainTestValSplit(X,y):
    # train 0.8, test 0.1, val 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
    X_train = np.array(X_train,  dtype=float)
    y_train = np.array(y_train,  dtype=float)   
    X_test = np.array(X_test,  dtype=float)
    y_test = np.array(y_test,  dtype=float)    
    X_val = np.array(X_val,  dtype=float)
    y_val = np.array(y_val,  dtype=float)   
    return X_train, y_train, X_test, y_test, X_val, y_val

if __name__ == "__main__":
    main()
