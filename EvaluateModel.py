from tqdm import tqdm
import os
import numpy as np
from ta import add_all_ta_features
from Models import CNN_2D_ALT_LOSS, MoE_CNN_2D, MoE_LSTM, lstmOnBanking, CNN_2D, MoE_LSTM_ALT_LOSS, MoE_CNN_2D_ALT_LOSS
import math
from tqdm import tqdm

projectDir = os.getcwd().split("abzocker")[0]
dataDir = os.path.join(projectDir, "abzocker", "data")
perfromanceDir = os.path.join(projectDir, "abzocker", "performance")

for dir in [dataDir, perfromanceDir]: 
    if not os.path.exists(dir):
        os.makedirs(dir)

testDataStart = "2015-01-02"
testDataEnd = "2022-12-12"


def main():
    # stockHistData    -> (stocks, timesteps, features)
    # stockNextDayData -> (stocks, target) 
    
    stockHistData, stockNextDayData, tickers = loadEvaluationData()

    # or for all Models
    modelList = [
        # CNN_2D_ALT_LOSS,
        # CNN_2D,
        # MoE_CNN_2D,
        # lstmOnBanking,
        # MoE_LSTM_ALT_LOSS,
        # MoE_LSTM,
        MoE_CNN_2D_ALT_LOSS
    ]
    
    for m in modelList:
        topKBuyAndHold(m, 10, stockHistData, stockNextDayData, initialCapital=10000)
        


def topKBuyAndHold(model, k, stockHistData, stockNextDayData, initialCapital = 10000):
    
    model, modelname = model.loadModel()
    modelPerformanceDir = os.path.join(perfromanceDir, modelname)
    if not os.path.exists(modelPerformanceDir):
        os.makedirs(modelPerformanceDir)
    
    
    cash = initialCapital
    portfolioValue = 0
    dailyValue = initialCapital
    
    portfolio = {}
    dailyReturns = []
    portfolioValueDevelopment = []
        
    # in recent years there has been a move towards commission free tradin, so we could set it to 0%
    transactionCostBuy = 0.0015  # 0.15% for buying shares 
    transactionCostSell = 0.0025  # 0.25% for selling shares
    
    
    investmentFactorNewStocks = 0.1

    for t in tqdm(range(len(stockNextDayData[0]) - 1)):
        
        predictedPrices = getPredictedStockValueAtDay(model, stockHistData, t)
        predictedStockValueChangePercent = getPredictedStockValuesChangePercent(stockHistData, predictedPrices, t)
        
        returnsWithStockIndices = [(predictedStockValueChangePercent[i], i) for i in range(len(predictedStockValueChangePercent))]
        sortedStockReturns = sorted(returnsWithStockIndices, reverse=True)
        
        topKStocks = sortedStockReturns[:k]

        
        # Sell stocks not in top k
        for stock in list(portfolio.keys()):
            if stock not in [stock[1] for stock in topKStocks]:
                sellPrice = getPriceOfStocksToday(stockHistData, t)
                cash += (portfolio[stock] * sellPrice[stock]) * (1 - transactionCostSell)
                # portfolioValue -= (portfolio[stock] * sellPrice[stock])
                del portfolio[stock]
        
        
        # Buy new stocks
        stockCount = len(portfolio.keys())
        investmentPerNewStock = math.floor(cash * investmentFactorNewStocks)
        for stock in [stock[1] for stock in topKStocks]:
            if stock not in portfolio:
                buyPrice = getPriceOfStocksToday(stockHistData, t)
                portfolio[stock] = float(investmentPerNewStock) / (buyPrice[stock] * (1 - transactionCostBuy))
                cash -= investmentPerNewStock

        
        # invest leftover money by predicted return
        returnsSum = np.sum([stock[0] for stock in topKStocks])
        investmentPerStock = [((stock[0]/returnsSum)*cash, stock[1]) for stock in topKStocks]
        for investment, stock in investmentPerStock:
                buyPrice = getPriceOfStocksToday(stockHistData, t)
                portfolio[stock] += float(investment) / (buyPrice[stock] * (1 - transactionCostBuy))
                cash -= investment
            
            
        # calculate portfolio value at end of day
        portfolioValue = 0
        for stock, investmentAmmount in portfolio.items():
            portfolioValue += investmentAmmount * getActualStockValuesAtNextDay(stockNextDayData, t)[stock]
            
        portfolioValueDevelopment.append(portfolioValue)
        
        dailyReturn = (portfolioValue / dailyValue) - 1
        dailyReturns.append(dailyReturn)
        dailyValue = portfolioValue
        
        
        
        
        
    cumReturns  = np.cumprod([dr+1 for dr in dailyReturns]) - 1
    np.save(os.path.join(modelPerformanceDir, "dailyReturns.npy"), dailyReturns)
    np.save(os.path.join(modelPerformanceDir, "cumulativeReturns.npy"), cumReturns)
    np.save(os.path.join(modelPerformanceDir, "portfolioValueDevelopment.npy"), portfolioValueDevelopment)

    # calculate return over entire trading period
    tradingPeriodReturn = portfolioValue / initialCapital
    
    # Average yearly return
    cumulativeReturn = cumReturns[-1]
    tradingDaysPerYear = 252
    numDays = len(dailyReturns)
    annualReturn = (1 + cumulativeReturn) ** (tradingDaysPerYear / numDays) - 1

    with open(os.path.join(modelPerformanceDir, "tradingstats.txt"), 'w') as file:
        file.write(f"TradingPeriodReturn: {tradingPeriodReturn} \n")    
        file.write(f"Annual Return: {annualReturn}")    




def getPredictedStockValueAtDay(model, stockHistData, t):
    # predict for all stocks, but only timestep i, all features
    return model.predict(np.expand_dims(stockHistData[:,t], axis=-1), verbose=0).squeeze()

def getActualStockValuesAtNextDay(stockNextDayData, t):
    return stockNextDayData[:,t]

def getPredictedStockValuesChangePercent(stockHistData, predictedPrices, t):
    return predictedPrices / getPriceOfStocksToday(stockHistData, t)
    
def getPriceOfStocksToday(stockHistData, t):
    return stockHistData[:,t, -1, getIndexOfFeature("Close")]





def loadEvaluationData():
    stockHistData = np.load(os.path.join(dataDir, "stockHistData.npy"))
    stockNextDayData = np.load(os.path.join(dataDir, "stockNextDayData.npy"))
    tickers = np.load(os.path.join(dataDir, "goodTickers.npy"))
    return stockHistData, stockNextDayData, tickers



def getIndexOfFeature(featureList=None, targetFeature = "Close"):
    if not featureList:
        featureList = getFeatures()
    indexes = np.where(np.atleast_1d(featureList == targetFeature))[0]
    if len(indexes) == 0:
        raise ValueError("Feature is not in feature list, target feature not available")
    return indexes[0]

def getFeatures():
    featuresPath = os.path.join(dataDir, "features.npy")
    if not os.path.exists(featuresPath):
        downloadFeatures()
    return np.load(featuresPath, allow_pickle=True)

def downloadFeatures():
    import yfinance as yf
    data = yf.download("AAPL", "2015-01-02", "2022-12-12", auto_adjust=True, keepna=True, progress=False, threads=8)
    data = data.interpolate(method="spline", order=3)
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    features =  data.columns
    np.save(os.path.join(dataDir, "features.npy"), features)

    
    
if __name__ == "__main__":
    main()
    
    
