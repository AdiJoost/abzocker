from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
from targetTypeEnum import TargetType

def main():
    data =_loadData()
    numberOfDatapointsForPrediction = 10
    
def generateDataSet(filename, sequenzLength, targetType:TargetType, saveAsFile=False):
    data = _loadData(filename)

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


if __name__ == "__main__":
    main()