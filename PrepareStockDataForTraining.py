import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os

from keras.utils import timeseries_dataset_from_array

from ta import add_all_ta_features # add all here select only needed one in training later




projectDir = os.getcwd()
dataPath = os.path.join(projectDir, "data")
os.makedirs(dataPath, exist_ok=True)
tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = tickers.Symbol.to_list()

trainDataStart = "2010-01-01"
trainDataEnd = "2015-01-01"
testDataStart = "2015-01-02" #test set ist vorgegeben
testDataEnd = "2022-12-12"
valDataStart = "2022-12-12"
valDataEnd = "2024-01-01"

sequenceLength = 20



def main():
    # list of preprocessed but not sliced stocks
    trainData, features = getStockData(tickers, trainDataStart, trainDataEnd)
    valData, features = getStockData(tickers, valDataStart, valDataEnd)
    testData, features = getStockData(tickers, testDataStart, testDataEnd)
    
    x_train, y_train = sliceStockData(trainData, features)
    x_val, y_val = sliceStockData(valData, features)
    x_test, y_test = sliceStockData(testData, features)
    
    np.save(os.path.join(dataPath,"x_train.npy"), x_train)
    np.save(os.path.join(dataPath,"y_train.npy"), y_train)
    np.save(os.path.join(dataPath,"x_val.npy"), x_val)
    np.save(os.path.join(dataPath,"y_val.npy"), y_val)
    np.save(os.path.join(dataPath,"x_test.npy"), x_test)
    np.save(os.path.join(dataPath,"y_test.npy"), y_test)
    
    
    # # To load the Data into TensorDatasets
    # batchSize = 64
    
    # x_trainLoaded = np.load(os.path.join(dataPath,"x_train.npy"))
    # y_trainLoaded = np.load(os.path.join(dataPath,"y_train.npy"))
    # x_valLoaded = np.load(os.path.join(dataPath,"x_val.npy"))
    # y_valLoaded = np.load(os.path.join(dataPath,"y_val.npy"))
    # x_testLoaded = np.load(os.path.join(dataPath,"x_test.npy"))
    # y_testLoaded = np.load(os.path.join(dataPath,"y_test.npy"))
    
    # trainDataset = tf.data.Dataset.from_tensor_slices((x_trainLoaded, y_trainLoaded)).batch(batchSize)
    # valDataset = tf.data.Dataset.from_tensor_slices((x_valLoaded, y_valLoaded)).batch(batchSize)
    # testDataset = tf.data.Dataset.from_tensor_slices((x_testLoaded, y_testLoaded)).batch(batchSize)
    
    # print(f"Shape: {x_trainLoaded.shape}")
    # print(f"Shape: {y_trainLoaded.shape}")
    # print(f"Shape: {x_valLoaded.shape}")
    # print(f"Shape: {y_valLoaded.shape}")
    # print(f"Shape: {x_testLoaded.shape}")
    # print(f"Shape: {y_testLoaded.shape}")
    
    
def shiftDateXDaysEarlier(dateStr, numDaysEarlier):
    date = dt.datetime.strptime(dateStr, "%Y-%m-%d")
    date = date - dt.timedelta(days=numDaysEarlier)
    return date.strftime("%Y-%m-%d")

def dataPreprocessing(data, startDate):
    try:
        data = add_all_ta_features(data,open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        data = data[startDate:]
    except:
        return None,None
    data = data.dropna()
    if not data.empty:
        featurelist = data.columns
        data = MinMaxScaler().fit_transform(data)
    else:
        return None, None
    return data, featurelist


def getStockData(tickers, startDate, endDate, preRunTimesteps = 200):
    shiftedStartDate = shiftDateXDaysEarlier(startDate, preRunTimesteps*1.5)

    stockDataList = []
    featureList = []
    for ticker in tickers:
        data = yf.download(ticker,shiftedStartDate, endDate, auto_adjust=True, keepna=False, progress=False, threads=8)
        if not data.empty:
            data, features = dataPreprocessing(data, startDate)
            
            if data is not None:
                stockDataList.append(data)
                
                if len(featureList) == 0:
                    featureList = features
            
                
    # print(f"Number of stocks in dataset: {len(stockDataList)}")
    return stockDataList, featureList


# target one of 'Open', 'High', 'Low', 'Close', 'Volume', ... one of the added indicators
def getTimeWindowsFromStock(data, sequenceLength, features, target = "Close"):
    generatedTimeseries = timeseries_dataset_from_array(data, targets=None, batch_size=None, sequence_length=sequenceLength + 1)
    X = np.zeros((len(generatedTimeseries), sequenceLength, data.shape[1]))
    Y = np.zeros((len(generatedTimeseries)))
    for i, timeWindow in enumerate(generatedTimeseries):
        X[i] = timeWindow[:-1,:]
        Y[i] = timeWindow[-1, features.get_loc(target)]
    
    return X, Y


def sliceStockData(data, features):
    X = []
    Y = []

    for stock in data:
        XNew, YNew = getTimeWindowsFromStock(stock, sequenceLength, features)
        X.append(XNew)
        Y.append(YNew)

    # Combine along first axis    
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    # add another dimension after last one
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)  
    
    return X,Y


if __name__ == "__main__":
    main()



