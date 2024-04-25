import tensorflow as tf
from keras import layers, losses, optimizers, metrics
from keras.utils import to_categorical, plot_model
import numpy as np
from keras.datasets import cifar10
import os
from sklearn.model_selection import train_test_split
np.random.seed(42)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
def main():
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    dataPath = os.path.join(head[0], "abzocker", "generatedData", "timeSeries")    
    
    X = np.load(os.path.join(dataPath, "X_combined.npy"))
    y = np.load(os.path.join(dataPath, "Y_combined.npy"))

    X = np.expand_dims(X, axis=-1).astype("float32")
    y = np.expand_dims(y, axis=-1).astype("float32")

    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)
    
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
    testDataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)
    valDataset = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(128)
    
    inputLength = x_train[0].shape[0]
    numberOfFeatures = x_train[0].shape[1]
    
    input_shape = (inputLength, numberOfFeatures, 1) # len, features, how many of those per timestep (depth)
    
    model = CNN2D(inputLength, numberOfFeatures, name="CNN", input_shape=input_shape)
    
    model.compile(optimizer=optimizers.Adam(),
                    loss=losses.MeanAbsoluteError(),
                    metrics=[metrics.MeanAbsoluteError()]) # review loss/metric, maybe custom
    
    history = model.fit(trainDataset, epochs=20, validation_data=valDataset, verbose=1)
    print(model.summary())
    
    plt.plot(history.history["loss"], label= "Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validataion Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    predictions = model.predict(x_test)
    errors = np.abs(predictions-y_test)
    plt.plot(errors, label="Absolute Errors")
    plt.title("ABs errors between Predictions and Ground Truth")
    plt.xlabel("Sample")
    plt.ylabel("Abs Error")
    plt.legend()
    plt.show()
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test,  predictions)
    print("MSE:  ", mse)
    print("MAE:  ", mae)
    print("RMSE: ", np.sqrt(mse))
    


# Maybe try 1d conv too makes sense too
class CNNBlock(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), pool_size=(3,1), activation="relu", name=None, **kwargs):
        super(CNNBlock, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, activation=activation, padding="same", name=f"{name}_conv")
        self.maxPool = layers.MaxPooling2D(pool_size=pool_size, name=f'{name}_pool') # pool size
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.maxPool(x)
        return x 


class CNN2D(tf.keras.Model):
    def __init__(self, inputLength, numberOfFeatures, name=None, **kwargs):
        super(CNN2D, self).__init__(name=name, **kwargs)
        self.norm = layers.BatchNormalization()
        self.cnn1 = CNNBlock(filters=4, kernel_size=(3, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn1")
        self.cnn2 = CNNBlock(filters=8, kernel_size=(3, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn2")
        self.cnn3 = CNNBlock(filters=16,kernel_size=(3, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn3")
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.dense_units = 128 # default 
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.fc2 = layers.Dense(1, name=f"{name}_output")
    
    
    def call(self, inputs):
        x = self.norm(inputs)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.flatten(x)
        self.dense_units = x.shape[-1] # number of units in the flattened x 
        print(self.dense_units)
        self.fc1.units = self.dense_units
        x = self.fc1(x)
        return self.fc2(x)
    

if __name__ == "__main__":
    main()