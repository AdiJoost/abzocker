import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import pickle
from keras.utils import timeseries_dataset_from_array

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
    tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = tickers.Symbol.to_list()

    testDataStart = "2015-01-02"
    testDataEnd = "2022-12-12"

    sequenceLength = 20

    targetFeature = "Close"

    goodTickersPath = os.path.join(dataPath, "goodTickers.npy")
    
    if not os.path.exists(goodTickersPath):
        testData, features, goodTickers = getStockData(tickers, testDataStart, testDataEnd)
        np.save(goodTickersPath, goodTickers)
    else:
        goodTickers = np.load(goodTickersPath)
        print(len(goodTickers))
        
    goodTickers = list(goodTickers)
    
    
    
    # Download    
    stockHistData = []
    stockNextDayData = []

    startDate = shiftDateXDaysEarlier(testDataStart, 1.5 * 200)
    testDataStart = shiftDateXDaysEarlier(testDataStart, sequenceLength) # first prediction is for first day
    
    for index, ticker in enumerate(tqdm(goodTickers, desc="Processing Tickers")):
        stockHistData.append([])    
        stockNextDayData.append([])
        
        data = yf.download(ticker, startDate, testDataEnd, auto_adjust=True, keepna=False, progress=False, threads=8)
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
        data = data[testDataStart:].astype(np.float32)
        
        targetFeatureLoc =  data.columns.get_loc(targetFeature)
        generatedTimeseries = timeseries_dataset_from_array(data, targets=None, batch_size=None, sequence_length=sequenceLength + 1)
        
        for timeWindow in generatedTimeseries:
            stockHistData[index].append(timeWindow[:-1])
            stockNextDayData[index].append(timeWindow[-1, targetFeatureLoc])
        

    
    # Access a stock by ticker
    # print(list(stockHistData[goodTickers.index("MMM")][0]))
    
    # Save
    np.save(os.path.join(dataPath, "stockHistData.npy"), stockHistData)
    np.save(os.path.join(dataPath, "stockNextDayData.npy"), stockNextDayData)

    print(f"Testdata X saved at: {os.path.join(dataPath, "stockHistData.npy")}")
    print(f"Testdata y saved at: {os.path.join(dataPath, "stockNextDayData.npy")}")

    # Example loading Data
    # stockHistData = np.load(os.path.join(dataPath, "stockHistData.npy"))
    # stockNextDayData = np.load(os.path.join(dataPath, "stockNextDayData.npy"))

if __name__ == "__main__":
    main()