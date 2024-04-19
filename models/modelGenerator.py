from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import pickle as pkl
from targetTypeEnum import TargetType

def main():
    numberOfDatapointsForPrediction = 10
    X, Y = generateDataSet("coolFile.pkl", 20, TargetType.CLOSE, True)
    print(X.shape)
    print(Y)
    
def generateDataSet(filename, sequenceLength, targetType:TargetType, saveAsFile=False, forceOverwrite=False):
    """
    returns X, Y: timeSeries array and target array
    (String)filename: .pkl file to load. EG: "mmm.pkl"
    (Integer)sequence: Length of the output sequences (in number of timesteps).
    (TargetType)targetType: Enum for desired target value in y.
    (Bool)saveAsFile: If True, saves X and Y in .../timeSeries/X_{filename}.npy and .../timeSeries/Y_{filename}.npy
    (Bool)forceOverwrite: If False, method wants to save and saving file already exist, ask user to confirm overwrite. If true, always overwrites.
    """
    data = _loadData(filename)
    X, Y = _getTimeWindow(data, sequenceLength, targetType)
    if saveAsFile:
        _saveFile(filename,X,Y,forceOverwrite)
    return X, Y

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