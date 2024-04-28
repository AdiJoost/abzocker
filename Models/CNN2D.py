import tensorflow as tf
from keras import layers, losses, optimizers, metrics
from keras.utils import to_categorical, plot_model
import numpy as np
from keras.datasets import cifar10
import os
from sklearn.model_selection import train_test_split
np.random.seed(42)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers.schedules import ExponentialDecay, CosineDecay

import matplotlib.pyplot as plt
import datetime

def main():
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    dataPath = os.path.join(head[0], "abzocker", "generatedData", "timeSeries")    
    
    X = np.load(os.path.join(dataPath, "X_combined.npy"))
    y = np.load(os.path.join(dataPath, "Y_combined.npy"))
        
    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)

    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)
    
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    testDataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(64)
    valDataset = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(64)
    
    inputLength = x_train[0].shape[0]
    numberOfFeatures = x_train[0].shape[1]
    
    input_shape = (inputLength, numberOfFeatures, 1) # len, features, how many of those per timestep (depth)
    
    model = CNN2D(inputLength, numberOfFeatures, name="CNN", input_shape=input_shape)
    
    initial_learning_rate = 0.1
    lr_schedule = ExponentialDecay(initial_learning_rate,
                                   decay_steps=80000,
                                   decay_rate=0.96,)
    
    earlyStop = EarlyStopping(monitor="loss", patience=3)
    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                    loss=losses.MeanAbsoluteError(),
                    metrics=[metrics.MeanAbsoluteError()]) # review loss/metric, maybe custom
    
    logDir = os.path.join(head[0], "abzocker", "Models", "logs", datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=logDir, histogram_freq=1)
    history = model.fit(trainDataset, epochs=10, validation_data=valDataset, verbose=1, callbacks=[tensorboard, earlyStop])
    
    print(model.summary())

    predictions = model.predict(x_test)
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test,  predictions)
    
    print("MSE:  ", mse)
    print("MAE:  ", mae)
    print("RMSE: ", np.sqrt(mse))
    print(f"Mean of y: {np.mean(y_test)}")


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
        self.norm1 = layers.BatchNormalization()
        self.cnn1 = CNNBlock(filters=8,  kernel_size=(2, 1), pool_size=(2,1), name=f"{name}_cnn1")
        self.cnn2 = CNNBlock(filters=16, kernel_size=(1, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn2")
        self.norm2 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.2)
        
        self.cnn3 = CNNBlock(filters=32, kernel_size=(3, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn3")
        self.cnn4 = CNNBlock(filters=64, kernel_size=(5, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn4")
        self.norm3 = layers.BatchNormalization()
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.dense_units = 128 # default 
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.fc2 = layers.Dense(1, name=f"{name}_output")
    
    
    def call(self, inputs):
        x = self.norm1(inputs)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.norm2(x)
        x = self.dropout1(x)
        
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.norm3(x)
        x = self.flatten(x)
        self.dense_units = x.shape[-1] # number of units in the flattened x 
        self.fc1.units = self.dense_units
        x = self.fc1(x)
        return self.fc2(x)

if __name__ == "__main__":
    main()