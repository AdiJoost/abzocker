import os
from Models.MoE_CNN import MixtureOfExperts, ExpertModel, CNNBlock 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.saving import load_model

import numpy as np
import os
import keras


cwd = os.getcwd()
head = cwd.split("abzocker")
modelPath = os.path.join(head[0], "abzocker", "Models", "checkpoints")



def main():   
    model = loadMoE_CNN()

    dataPath = os.path.join(head[0], "abzocker", "data")    
    x_test = np.load(os.path.join(dataPath,"x_test.npy")).astype(np.float32)[0:10000]
    y_test = np.load(os.path.join(dataPath,"y_test.npy")).astype(np.float32)[0:10000]
    
    print(model.summary())
    
    
    
    
    predictions = model.predict(x_test)
    print("Predictions on Test set")
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test,  predictions)
    print("MSE:  ", mse)
    print("MAE:  ", mae)
    print("RMSE: ", np.sqrt(mse))
    
    print("startpred500")
    p = model.predict(x_test[0:500])
    mean_absolute_error(p, y_test[0:500])
    
    print("endpred500")
    pass
    
    
    
    
def loadMoE_CNN(): 
    custom_objects = {
        "MixtureOfExperts": MixtureOfExperts,
        "ExpertModel": ExpertModel,
        "CNNBlock": CNNBlock
    }

    model_filepath = os.path.join(modelPath, "MoE_CNN_2D.keras")

    model = keras.models.load_model(model_filepath, custom_objects=custom_objects)
    #model = keras.models.load_model("/home/schafhdaniel@edu.local/banking/abzocker/Models/checkpoints/MoE_CNN_2D_tmp.keras", custom_objects=custom_objects)

    return model


def loadCNN():
    print("NotImplemented")
    pass

def loadLSTM():
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    lstmPath = os.path.join(head[0], "abzocker", "Models", "lstm.keras")
    model = load_model(lstmPath)
    return model


def loadMoE_CNN_SMALL():
    print("NotImplemented")
    pass
    
def loadMoE_CNN_CUSTOM_LOSS():
    print("NotImplemented")
    pass

if __name__ == "__main__":
    main()