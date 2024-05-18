#import tensorflow as tf
import os
# important
from MoE_CNN import MixtureOfExperts, ExpertModel, CNNBlock  # replace `your_module_path` with the actual module path
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import os
import keras






def main():   
    model = loadMoE_CNN()
    cwd = os.getcwd()
    head = cwd.split("abzocker")
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
    
    
    
    
#def loadModel(path = )
def loadMoE_CNN(): 
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    checkpoint_filepath = os.path.join(head[0], "abzocker", "Models", "checkpoints")

    # Define the custom objects dictionary
    custom_objects = {
        "MixtureOfExperts": MixtureOfExperts,
        "ExpertModel": ExpertModel,
        "CNNBlock": CNNBlock
    }

    model_filepath = os.path.join(checkpoint_filepath, "MoE_CNN_2D.keras")

    # Load the model
    # model = keras.models.load_model(model_filepath, custom_objects=custom_objects)
    model = keras.models.load_model("/home/schafhdaniel@edu.local/banking/abzocker/Models/checkpoints/MoE_CNN_2D_tmp.keras", custom_objects=custom_objects)


    print("Loaded Model")

    return model

    # cwd = os.getcwd()
    # head = cwd.split("abzocker")
    # dataPath = os.path.join(head[0], "abzocker", "data")    
    # x_test = np.load(os.path.join(dataPath,"x_test.npy")).astype(np.float32)[0:10000]
    # y_test = np.load(os.path.join(dataPath,"y_test.npy")).astype(np.float32)[0:10000]

    # print(model.summary())




    # predictions = model.predict(x_test)
    # print("Predictions on Test set")
    # mse = mean_squared_error(y_test, predictions)
    # mae = mean_absolute_error(y_test,  predictions)
    # print("MSE:  ", mse)
    # print("MAE:  ", mae)
    # print("RMSE: ", np.sqrt(mse))




if __name__ == "__main__":
    main()