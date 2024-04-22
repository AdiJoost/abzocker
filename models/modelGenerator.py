from keras.utils import timeseries_dataset_from_array
import numpy as np
import pandas as pd
import os
from targetTypeEnum import TargetType
#from dataHandling.StockDataExtractor import getStocksymbols
np.random.seed(42)


def main():
    generateSlicedTimeseriesForNStocks(25)
    
def generateSlicedTimeseriesForNStocks(numberOfStocks: int, saveAsOneFile=True):
    cwd = os.getcwd().split("abzocker")[0]
    dataPath = os.path.join(cwd, "abzocker", "Resources", "sp500_stocks.csv")
    stockSymbols = pd.read_csv(dataPath).Symbol.unique()
    stockPicks = np.random.choice(stockSymbols, numberOfStocks)

    if saveAsOneFile:
        X = []
        Y = []
        for stock in stockPicks:
            XNew, YNew = generateDataSet(f"{stock}.pkl", 20, TargetType.CLOSE, saveAsFile=False)
            print(f"X_shape: {XNew.shape}, Y_shape: {YNew.shape}")
            X.append(XNew)
            Y.append(YNew)
        
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)    
        print(f"X_total shape {X.shape}, Y_total shape {Y.shape}")
        
        _saveFile("combined.pkl", X, Y, True)
    
    else:
        for stock in stockPicks:
            generateDataSet(f"{stock}.pkl", 20, TargetType.CLOSE, saveAsFile=False)


def generateDataSet(filename: str, sequenceLength: int, targetType:TargetType, saveAsFile=False, forceOverwrite=False):
    """
    returns X, Y: timeSeries array and target array
    filename: .pkl file to load. EG: "mmm.pkl"
    sequence: Length of the output sequences (in number of timesteps).
    targetType: Enum for desired target value in y.
    saveAsFile: If True, saves X and Y in .../timeSeries/X_{filename}.npy and .../timeSeries/Y_{filename}.npy
    forceOverwrite: If False, method wants to save and saving file already exist, ask user to confirm overwrite. If true, always overwrites.
    """
    data = _loadData(filename)
    X, Y = _getTimeWindow(data, sequenceLength, targetType)
    
    if saveAsFile:
        _saveFile(filename,X,Y,forceOverwrite)
    return X, Y


def _removeNan(data):
    return data.dropna()
    
def _saveFile(filename, X, Y, forceOverwrite):
    xPath = _getAndValidateSavePath(f"X_{filename[:-4]}.npy", forceOverwrite)
    yPath = _getAndValidateSavePath(f"Y_{filename[:-4]}.npy", forceOverwrite)
    with open(xPath, "wb") as file:
        np.save(file, X)
    with open(yPath, "wb") as file:
        np.save(file, Y)

def _getTimeWindow(data, sequenceLength, targetType):
    generatedTimeseries = timeseries_dataset_from_array(data, targets=None, batch_size=None, sequence_length=sequenceLength + 1)
    X = np.zeros((len(generatedTimeseries), sequenceLength, 7))
    Y = np.zeros((len(generatedTimeseries)))
    for i, batch in enumerate(generatedTimeseries):
        X[i,:,:] = batch[:-1,:]
        Y[i] = batch[-1,targetType.value[0]]
    return X, Y

def _loadData(filename):
    path = _getPath(filename)
    data = pd.read_pickle(path)
    data["Date"] = data["Date"].astype("int64")
    data = _removeNan(data)
    return data.to_numpy()

def _getPath(filename):
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    if len(head) < 1:
        raise NameError("Cannot get the working Path. Look at Source Code and Debug. :/")
    return os.path.join(head[0], "abzocker", "generatedData", filename)

def _getAndValidateSavePath(filename, forceOverwrite):
    path = _getSavePath(filename)
    if not forceOverwrite and os.path.exists(path):
        userInput = input(f"Flie {filename} already exists. Overwrite? [y/N]:")
        if userInput != "y":
            raise NameError("Abort saving")
    return path

def _getSavePath(filename):
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    if len(head) < 1:
        raise NameError("Cannot get the working Path. Look at Source Code and Debug. :/")
    return os.path.join(head[0], "abzocker", "generatedData", "timeSeries", filename)

if __name__ == "__main__":
    main()