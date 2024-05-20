from tqdm import tqdm
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from Models import MoE_CNN
from tensorflow.keras.saving import load_model


projectDir = os.getcwd().split("abzocker")[0]
dataPath = os.path.join(projectDir, "abzocker", "data")

testDataStart = "2015-01-02"
testDataEnd = "2022-12-12"

# 1. load data
# 2. load model
# 3. make a prediction for the i-th timewindow of each stock
# 4. Trading strategy top k buy and hold

def main():
    # stockHistData    -> (stocks, timesteps, features)
    # stockNextDayData -> (stocks, target) 
    stockHistData, stockNextDayData, tickers = loadEvaluationData()
    #model = MoE_CNN.loadModel()
    model = _loadLSTM()

    tradingPeriodReturn = topKBuyAndHold(model, 10,stockHistData, stockNextDayData, initialCapital=10000)
    print(f"tradingPeriodReturn: {tradingPeriodReturn}")


def topKBuyAndHold(model, k, stockHistData, stockNextDayData, initialCapital = 10000):
    portfolioValue = initialCapital
    portfolio = {}
    dailyReturns = []
    monthlyReturns = []
    
    riskFreeRate = 0.02  # Example risk-free rate, adjust as needed
    # in recent years there has been a move towards commission free tradin, so we could set it to 0%
    transactionCostBuy = 0.0015  # 0.15% for buying shares 
    transactionCostSell = 0.0025  # 0.25% for selling shares
    
    
    for t in range(len(stockNextDayData[0]) - 1):
         
        predictedPrices = getPredictedStockValueAtDay(model, stockHistData, t)
        predictedStockValueChangePercent = getPredictedStockValuesChangePercent(stockHistData, predictedPrices, t)
        
        returnsWithStockIndices = [(predictedStockValueChangePercent[i], i) for i in range(len(predictedStockValueChangePercent))]
        sortedStockReturns = sorted(returnsWithStockIndices, reverse=True)
        
        
        topKStocks = sortedStockReturns[:k]
        
        # Sell stocks not in top k
        for stock in list(portfolio.keys()):
            if stock not in [stock[1] for stock in topKStocks]:
                sellPrice = getPriceOfStocksToday(stockHistData, t)
                portfolioValue += (portfolio[stock] * sellPrice[stock]) * (1 - transactionCostSell)
                del portfolio[stock]
        
        investmentPerStock = portfolioValue / k
        # buy top k stocks
        for stock in [stock[1] for stock in topKStocks]:
            if stock not in portfolio:
                buyPrice = getPriceOfStocksToday(stockHistData, t)
                portfolio[stock] = investmentPerStock / buyPrice[stock] * (1 - transactionCostBuy)
                

        # calculate portfolio value at end of day (check if predictions were good)
        portfolioValue = 0
        for stock, investmentAmmount in portfolio.items():
            portfolioValue += investmentAmmount * getActualStockValuesAtNextDay(stockNextDayData, t)[stock]
            
        # calculate daily return
        dailyReturn =  (portfolioValue / initialCapital) - 1
        dailyReturns.append(dailyReturn)
        
        
        # update initial capital for next day
        initialCapital = portfolioValue
        
    np.save("dailyReturns.npy", dailyReturns)

    #monthlyReturns = dailyReturns[::21] # average days to trade in a month
    # calculate return
    tradingPeriodReturn = portfolioValue / initialCapital
    print(f"Trading Period Return: {tradingPeriodReturn}")
    
    # for the volatility of the investment plot daily returns
    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    ax1.plot(dailyReturns)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Percent")
    ax1.set_title("Portfolio daily returns data")
    plt.show()
    plt.savefig("DailyReturns.png")
    
    plt.hist(dailyReturns, bins=100, density=True)
    plt.xlabel("Daily returns %")
    plt.ylabel("Precent")
    plt.show()
    plt.savefig("DailyReturnsHistogramm.png")
        
    
    # total growth
    dates = pd.date_range(start=testDataStart, end=testDataEnd, freq='B')  # Business days frequency

    cumReturns = np.cumprod([dr+1 for dr in dailyReturns])
    print(f"lendates {len(dates)}")
    print(f"len cumreturns {len(cumReturns)}")
    plt.plot(dates, cumReturns, marker='o', linestyle='-')
    plt.show()
    plt.savefig("cumReturns.png")
    return tradingPeriodReturn


def getPredictedStockValueAtDay(model, stockHistData, t):
    # predict for all stocks, but only timestep i, all features
    return model.predict(np.expand_dims(stockHistData[:,t], axis=-1)).squeeze()


def getActualStockValuesAtNextDay(stockNextDayData, t):
    return stockNextDayData[:,t]

def getPredictedStockValuesChangePercent(stockHistData, predictedPrices, t):
    return predictedPrices / getPriceOfStocksToday(stockHistData, t)
    
def getPriceOfStocksToday(stockHistData, t):
    return stockHistData[:,t, -1, getIndexOfFeature("Close")]





def loadEvaluationData():
    stockHistData = np.load(os.path.join(dataPath, "stockHistData.npy"))
    stockNextDayData = np.load(os.path.join(dataPath, "stockNextDayData.npy"))
    tickers = np.load(os.path.join(dataPath, "goodTickers.npy"))
    return stockHistData, stockNextDayData, tickers





def getIndexOfFeature(featureList=None, targetFeature = "Close"):
    if not featureList:
        featureList = getFeatures()
    indexes = np.where(np.atleast_1d(featureList == targetFeature))[0]
    if len(indexes) == 0:
        raise ValueError("Feature is not in feature list, target feature not available")
    return indexes[0]

def getFeatures():
    featuresPath = os.path.join(dataPath, "features.npy")
    if not os.path.exists(featuresPath):
        downloadFeatures()
    return np.load(featuresPath, allow_pickle=True)

def downloadFeatures():
    import yfinance as yf
    data = yf.download("AAPL", "2015-01-02", "2022-12-12", auto_adjust=True, keepna=True, progress=False, threads=8)
    data = data.interpolate(method="spline", order=3)
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    features =  data.columns
    np.save(os.path.join(dataPath, "features.npy"), features)


def _loadLSTM():
    #TODO: should be moved to modelLoader.py. But modelLoader.py seems broken. Maybe a .gitignore problem
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    lstmPath = os.path.join(head[0], "abzocker", "Models", "lstm.keras")
    model = load_model(lstmPath)
    return model
    
if __name__ == "__main__":
    main()
    
    
