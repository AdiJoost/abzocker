import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import os
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import numpy as np

cwd = os.getcwd()
head = cwd.split("abzocker")
dataDir = os.path.join(head[0], "abzocker", "data")    
modelDir = os.path.join(head[0], "abzocker", "Models", "Models")
performanceDir = os.path.join(head[0], "abzocker", "performance")


def main():
    allPortfolioValueDevelopment = []
    allDailyReturns = []
    allCumulativeReturns = []
    for root, directories, files in os.walk(performanceDir):
        for dir in directories:
            modelPerformanceDir = os.path.join(root, dir)
        
            portfolioValueDevelopment = np.load(os.path.join(modelPerformanceDir,"portfolioValueDevelopment.npy"))
            dailyReturns = np.load(os.path.join(modelPerformanceDir,"dailyReturns.npy"))
            cumulativeReturns = np.load(os.path.join(modelPerformanceDir,"cumulativeReturns.npy"))
            
            plot(portfolioValueDevelopment, dailyReturns, cumulativeReturns, dir, modelPerformanceDir)
            
            allPortfolioValueDevelopment.append((dir, portfolioValueDevelopment))
            allDailyReturns.append((dir, dailyReturns))
            allCumulativeReturns.append((dir, cumulativeReturns))

            
    compareAll(allCumulativeReturns, allPortfolioValueDevelopment)



def compareAll(allCumulativeReturns, allPortfolioValueDevelopment):
    tradingDays = getEstimatedDates(len(allCumulativeReturns[0][1]))
    snp500daily, snp500cum = getSNP500Data(len(tradingDays))
    
    formatter = FuncFormatter(toPercent)

    plt.figure(figsize=(15, 10))
    for model, cumulativeReturns in allCumulativeReturns:
        plt.plot(tradingDays, cumulativeReturns, linestyle='-', label=f"{model}")
    
    plt.plot(tradingDays, snp500cum, linestyle='-', label=f"S&P 500 Index")

    plt.title(f"Portfolio Returns %")
    plt.ylabel("Returns Percent")
    plt.xlabel("Date")
    plt.xticks(tradingDays[::252])
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(performanceDir, f"CumulativeReturns_ALL.png"))
    plt.close()
    
    

def plot(portfolioValueDevelopment, dailyReturns, cumulativeReturns, model, savePath):
    tradingDays = getEstimatedDates(len(dailyReturns))
    
    plotHistogramm(dailyReturns, model, savePath)
    plotCumReturns(cumulativeReturns, tradingDays, model, savePath)
    plotPortfolioValue(portfolioValueDevelopment, tradingDays,  model, savePath)
    plotDailyReturns(dailyReturns, tradingDays, model, savePath)
    
    
def plotHistogramm(dailyReturns, model, savePath):
    plt.hist(dailyReturns, bins=100, density=True)
    plt.xlabel("Daily returns %")
    plt.ylabel("Percent")
    plt.title(f"Distribution of returns: {model}")
    plt.show()
    plt.savefig(os.path.join(savePath, f"DistributionOfReturns_{model}.png"))
    plt.close()




def plotCumReturns(cumulativeReturns, tradingDays,  model, savePath):
    
    snp500daily, snp500cum = getSNP500Data(len(cumulativeReturns))
    
    formatter = FuncFormatter(toPercent)

    plt.figure(figsize=(15, 5))
    plt.plot(tradingDays, cumulativeReturns, linestyle='-', label=f"Portfolio by: {model}")
    plt.plot(tradingDays, snp500cum, linestyle='-', label=f"S&P 500 Index")

    plt.title(f"Portfolio Returns %: {model}")
    plt.ylabel("Returns Percent")
    plt.xlabel("Date")
    plt.xticks(tradingDays[::252])
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(savePath, f"CumulativeReturns_{model}.png"))
    plt.close()


def toPercent(y, _):
    return '{:.0f}%'.format(y * 100)

def plotPortfolioValue(portfolioValueDevelopment, tradingDays,  model, savePath):
    
    snp500daily, snp500cum = getSNP500Data(len(portfolioValueDevelopment))
    
    indexValDev = [10000]
    for dr in snp500daily[1:]:
        indexValDev.append(indexValDev[-1]* (1+dr))

    plt.figure(figsize=(15, 5))
    plt.plot(tradingDays, portfolioValueDevelopment, linestyle='-', label=f"Portfolio by: {model}")
    plt.plot(tradingDays, indexValDev, linestyle='-', label=f"S&P 500 Index")

    plt.title(f"Portfolio Value: {model}")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Date")
    plt.xticks(tradingDays[::252])
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(savePath, f"PortfolioValue_{model}.png"))
    plt.close()



def plotDailyReturns(dailyReturns,tradingDays,  model, savePath):
    plt.figure(figsize=(15, 5))
    plt.plot(tradingDays, dailyReturns)
    plt.xticks(tradingDays[::252])
    plt.title(f"Portfolio Returns: {model}")
    plt.xlabel("Date")
    plt.ylabel("Returns %")
    plt.show()
    plt.savefig(os.path.join(savePath, f"PortfolioReturns_{model}.png"))
    plt.close()

    
    
    
    
def getSNP500Data(numberOfDays):
    ticker = "^GSPC"
    
    tradingDays = getEstimatedDates(numberOfDays)
    
    startDate = tradingDays[0]
    endDate = tradingDays[-1]
    
    data = yf.download(ticker, start=startDate, end=endDate)
    dailyReturns = data['Close'].pct_change() 
    
    cumReturns = np.cumprod(1+dailyReturns) - 1
    return dailyReturns[:numberOfDays], cumReturns[:numberOfDays]
    
    
def getEstimatedDates(numberOfDays):
    import holidays
    
    startDate = datetime(2015, 1, 2)

    holidays = holidays.US()

    def isTradingDay(date):
        return date.weekday() < 5 and date not in holidays

    tradingDaysList = []
    currentDate = startDate
    while len(tradingDaysList) < numberOfDays:
        if isTradingDay(currentDate):
            tradingDaysList.append(currentDate)
        currentDate += timedelta(days=1)

    tradingDays = [date.strftime('%Y-%m-%d') for date in tradingDaysList]
    


    return tradingDays

    

if __name__ == "__main__":
    main()