import os
import pandas as pd

RESOURCE_FOLDER = "Resources"
DATA_FOLDER = "generatedData"
DATE_COLUMN = "Date"


def main():
    #get dataPath
    dataPath = getDatapath("sp500_stocks.csv")
    #LOAD DATASET form datapath
    df = loadDataset(dataPath)
    #Filter dataset by company
    df = filterSet("MMMfjh", df)
    #save Dataset
    saveDataset("coolFile.csv", df)
    pass

def getDatapath(filename):
    cwd = os.getcwd()
    return os.path.join(cwd, RESOURCE_FOLDER, filename)


def loadDataset(path):
    df = pd.read_csv(path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(by=DATE_COLUMN, ascending=True)
    return df


def filterSet(companyName, df, column="Symbol"):
    #maybe do some validation if you have fun and time
    returnDf = df[df[column] == companyName]
    return returnDf

def saveDataset(fileName, df):
    path = getAndValidatePath(fileName)
    df.to_csv(path)

def getAndValidatePath(filename):
    path = getSavingPath(filename)
    if (os.path.exists(path)):
        if not askForOverwritePermission(filename):
            raise ("Saving Path already exists, abort")
    return path

def getSavingPath(filename):
    cwd = os.getcwd()
    return os.path.join(cwd, DATA_FOLDER, filename)

def askForOverwritePermission(filename):
    userInput = input(f"{filename} already exists, do you want to overwrite? [y/N]")
    return userInput.lower() == "y"

if __name__ == "__main__":
    main()