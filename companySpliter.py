import os

DATA_FOLDER = "Resources"

def main():
    #get dataPath
    print(getDatapath("sp500_companies.csv"))
    #LOAD DATASET form datapath
    df = loadDataset()
    #Filter dataset by company

    #save Dataset
    pass

def getDatapath(filename):
    cwd = os.getcwd()
    return os.path.join(cwd, DATA_FOLDER, filename)


def loadDataset(path):
    pass

def filterSet(companyName):
    pass

def saveDataset(savingPath):
    pass

if __name__ == "__main__":
    main()