import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import pickle
from keras.utils import timeseries_dataset_from_array
from scipy import stats

from ta import add_all_ta_features # add all here select only needed one in training later

from PrepareStockDataForTraining import getStockData, shiftDateXDaysEarlier
from tqdm import tqdm


# This notebook is for downloading the stock data from yfinance, adding the features.
# It is needed to test the models performance in a realisic setting as opposed to the other test set that just contains random snp500 stock data timewindows. 
# Finally it is saved in the "data" directory


def main():
    #Config
    projectDir = os.getcwd()
    dataPath = os.path.join(projectDir, "data")

    testDataStart = "2015-01-02"
    testDataEnd = "2022-12-12"

    sequenceLength = 20

    targetFeature = "Close"
    
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = tickers.Symbol.to_list()
    tickers = list(tickers)
    
    
    
    # Download    
    stockHistData = []
    stockNextDayData = []
    
    indexesToDelete = []

    startDate = shiftDateXDaysEarlier(testDataStart, 1.5 * 200)
    testDataStart = shiftDateXDaysEarlier(testDataStart, sequenceLength) # first prediction is for first day
    
    for index, ticker in enumerate(tqdm(tickers, desc="Processing Tickers")):
        stockHistData.append([])    
        stockNextDayData.append([])
        data = yf.download(ticker, startDate, testDataEnd, auto_adjust=True, keepna=True, progress=False, threads=8)
        data = data.interpolate(method="spline", order=3)
        
        if data is None or len(data) == 0 or data.empty:          
            continue
        
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        data = data[testDataStart:].astype(np.float32)
        print(f"Data shape: {data.shape}, index: {index}")
        targetFeatureLoc =  data.columns.get_loc(targetFeature)
        generatedTimeseries = timeseries_dataset_from_array(data, targets=None, batch_size=None, sequence_length=sequenceLength + 1)
        
        for timeWindow in generatedTimeseries:
            stockHistData[index].append(timeWindow[:-1])
            stockNextDayData[index].append(timeWindow[-1, targetFeatureLoc])
            
    print(f"len stockHistData   : {len(stockHistData)}")
    print(f"len stockNextDayData: {len(stockNextDayData)}")

    mostOftenTimeWindowLen = stats.mode([len(stock) for stock in stockHistData]).mode
    print(f"mostOftenTimeWindowLen: {mostOftenTimeWindowLen}")

    indexesToDelete = [index for index, value in enumerate(stockHistData) if len(value) != mostOftenTimeWindowLen]
    print(f"num of indexes to delete: {len(indexesToDelete)}")
    print(f"toDelete: {indexesToDelete}")

   
    # remove stocks that dont have the same amount of timeWindows
    # Is because yfinance is sometimes a bit dumb (just scrapes yahoo finance, some dates missing sometimes)
    for id  in sorted(indexesToDelete, reverse=True):
        stockHistData.pop(id)
        stockNextDayData.pop(id)
        tickers.pop(id)
    
        
    print(f"len stockHistData after    : {len(stockHistData)}")
    print(f"len stockNextDayData after : {len(stockNextDayData)}")
    print(f"len tickers: {len(tickers)}")
    
    # Access a stock by ticker
    #print(list(stockHistData[goodTickers.index("MMM")][0]))
    stockHistData = np.asarray(stockHistData, dtype=np.float32)
    stockNextDayData = np.asarray(stockNextDayData, dtype=np.float32)
    
    # Save
    np.save(os.path.join(dataPath, "stockHistData.npy"), stockHistData)
    np.save(os.path.join(dataPath, "stockNextDayData.npy"), stockNextDayData)
    np.save(os.path.join(dataPath, "goodTickers.npy"), tickers)
    
    print(f"Testdata X saved at: {os.path.join(dataPath, "stockHistData.npy")}")
    print(f"Testdata y saved at: {os.path.join(dataPath, "stockNextDayData.npy")}")
    print(f"Tickers saved at:     {os.path.join(dataPath, "goodTickers.npy")}")

if __name__ == "__main__":
    main()