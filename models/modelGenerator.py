from keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

def main():
    data =_loadData()
    print(data)

def getModel():
    pass

def _loadData():
    path = _getPath("coolFile.pkl")
    data = pd.read_pickle(path)
    print(data.head())
    return data.to_numpy()

def _getPath(filename):
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    if len(head) < 1:
        raise NameError("Cannot get the working Path. Look at Source Code and Debug. :/")
    return os.path.join(head[0], "abzocker", "generatedData", filename)


if __name__ == "__main__":
    main()