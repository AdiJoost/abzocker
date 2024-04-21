import os
import pandas as pd

RESOURCE_FOLDER = "Resources"
DATA_FOLDER = "generatedData"
DATA_FILE = "sp500_stocks.csv"
DATE_COLUMN = "Date"

def main():
    extractAndSaveAllStocksInDataset()


def extractAndSaveAllStocksInDataset():
    stockSymbols = getStocksymbols()
    
    for stock in stockSymbols:
        extractAndSaveCompanyStocks(companyName=stock, saveFile=f"{stock}.pkl")
    
def getStocksymbols():
    dataPath = getDatapath(DATA_FILE)
    df = loadDataset(dataPath)
    return df.Symbol.unique()
    
def extractAndSaveCompanyStocks(companyName="MMM", saveFile="coolFile.pkl", dropCompanyName=True, autoOverwrite=False):
    """
    companyName: String with the Symbol-Name of the company name you wish to extract
    saveFile: name of the file, where the extracted dataFrame is saved. Example 'myFile.csv'
    autoOverwrite: false -> the script asks for permission to overwrite an existing datafile. True -y The script just overwrites a file, if the saveFile already exists.
    """
    dataPath = getDatapath(DATA_FILE)
    df = loadDataset(dataPath)
    df = filterSet(companyName, df)
    if dropCompanyName:
        df = df.drop("Symbol", axis=1)
    saveDataset(saveFile, df, autoOverwrite)

def getDatapath(filename, resourceFolder=RESOURCE_FOLDER):
    cwd = os.getcwd()
    return os.path.join(cwd, resourceFolder, filename)


def loadDataset(path):
    df = pd.read_csv(path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(by=DATE_COLUMN, ascending=True)
    return df


def filterSet(companyName, df, column="Symbol"):
    returnDf = df[df[column] == companyName]
    return returnDf

def saveDataset(fileName, df, autoOverwrite):
    path = getAndValidatePath(fileName, autoOverwrite)
    df.to_pickle(path)

def getAndValidatePath(filename, autoOverwrite):
    path = getSavingPath(filename)
    if (not autoOverwrite and os.path.exists(path)):
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