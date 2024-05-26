# Description
We test several deepLearning approaches for predicting the stock market, specifically the stocks in the snp500
1. LSTM
2. CNN
3. Mixture of Experts LSTM
4. Mixture of Experts CNN
5. The above with an alternative loss function, that combines Mean Absolute Error with the Mean Directional Accuracy (weighted 50/50)



# Setup
0. Install the environment.yaml in a new environment #todo add env specification to project
1. Run PrepareStockDataForTraining.py: This will get the data needed for training, and evaluating the model from yfinance and adds features using the ta package. The Data will then be saved as .npy files in the data folder
2. From the Models directory, run the scripts of the different models to train them. They will be saved in Models/Models
3. Run PrepareStockDataForEvaluation.py to get the data into the right format for realistic training. Will be saved in the data folder
4. Run EvaluateModel.py to check the preformance of the models in combination with our trading strategy, in a "realistic" setting. This will output the daily returns, cumulative returns... in the preformance dir, each model gets its own subdirectory here. Comment out the models in the list at the beginning, that you dont want to evaluate/havent trained a model of. The model needs to be in the Models/Models dir.
5. run the plots.py script. This will Plot histogramm of daily returns/ daily returns over time/ cumulative returns, and a comparison of all the models, against the snp500 index. The plots will be saved in the preformance dir for each model. Some outputs, like SHARPE, Annual return are outputed on the command line


# To be done
- Add environment file
- Check the trading strategy, maybe we do something wrong there
- Check for Data Leakage, move validationset before test set (time wise)
- Change the exponential decay to a step count that is usefull, it is way to high right now. Retrain models
- Try different weights for the Alternative Loss function
- Clean up the code 
- Don't download the train data twice, reuse the test data that is downloaded by PrepareStockDataForEvaluation.py


# Files

PrepareStockDataForEvaluation: Download test data for evaluation with trading strategy
PrepareStockDataForTraining: Download train, test, val data
EvaluateModel: Tests the models, with the trading strategy
plots: Makes plots of the returns of the model, calculates some stats

/Models: Scripts to train the different models
/Models/Models: Contains the trained Models
/Performance: Performance of models during training and evaluation on trading strategy + Plots

/Unused: Some stuff we tried at the beginning #todo remove