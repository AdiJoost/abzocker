import pandas as pd
import os
from companySpliter import DATA_FOLDER, getStockSymbols

def main():
    stockSymbols = getStockSymbols()
    prepareAllData(stockSymbols)
    
    
    
def prepareAllData(stockSymbols=["AAPL"]):
    for stock in stockSymbols:     
        filepath = getDataPath(stock)  
        pd.read_pickle()
        prepareDataFrame(stockName=stock)
        
        
def getDataPath(stockName="AAPL"):
    cwd = os.getcwd()
    return os.path.join(cwd,DATA_FOLDER,stockName)
    

def prepareDataFrame(stockName="AAPL", DATA_FOLDER=""):
    pass

def cleanData():
    pass

def enrichData():
    pass

def prepareModelInputs(inputlength=14):
    pass

if __name__ == "__main__":
    main()
