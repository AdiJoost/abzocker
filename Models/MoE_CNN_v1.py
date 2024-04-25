import tensorflow as tf
from keras import layers, losses, optimizers, metrics
from keras.utils import to_categorical
import numpy as np
from keras.datasets import cifar10
import os
from sklearn.model_selection import train_test_split
np.random.seed(42)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def main():
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    dataPath = os.path.join(head[0], "abzocker", "generatedData", "timeSeries")    
    
    X = np.load(os.path.join(dataPath, "X_combined.npy"))
    y = np.load(os.path.join(dataPath, "Y_combined.npy"))

    print(np.isin(X, [0]).sum())

    X = np.expand_dims(X, axis=-1).astype("float32")
    y = np.expand_dims(y, axis=-1).astype("float32")
        

    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)
    
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
    testDataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)
    valDataset = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(128)
    
    numberOfExperts = 3
    inputLength = x_train[0].shape[0]
    numberOfFeatures = x_train[0].shape[1]
    
    input_shape = (inputLength, numberOfFeatures, 1) # len, features, how many of those per timestep (depth)
    
    moe_model = MixtureOfExperts(numberOfExperts, inputLength, numberOfFeatures, name="mixture_of_experts", input_shape=input_shape)
    
    moe_model.compile(optimizer=optimizers.Adam(),
                    loss=losses.MeanSquaredError(),
                    metrics=[metrics.MeanSquaredError()]) # review loss/metric, maybe custom
    
    history = moe_model.fit(trainDataset, epochs=2, validation_data=valDataset, verbose=1)
    
    print(moe_model.summary())
    print(f"training loss", history.history["loss"])
    
    predictions = moe_model.predict(x_test)
    
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


class ExpertModel(tf.keras.Layer):
    def __init__(self, inputLength, numberOfFeatures, name=None, **kwargs):
        super(ExpertModel, self).__init__(name=name, **kwargs)
        self.cnn1 = CNNBlock(filters=4, kernel_size=(3, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn1")
        self.cnn2 = CNNBlock(filters=8, kernel_size=(3, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn2")
        self.cnn3 = CNNBlock(filters=16,kernel_size=(3, numberOfFeatures), pool_size=(2,1), name=f"{name}_cnn3")
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.dense_units = 128 # default 
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.fc2 = layers.Dense(1, name=f"{name}_output")
    
    
    def call(self, inputs):
        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.flatten(x)
        self.dense_units = x.shape[-1] # number of units in the flattened x 
        self.fc1.units = self.dense_units
        x = self.fc1(x)
        return self.fc2(x)
    



class MixtureOfExperts(tf.keras.Model):
    def __init__(self, numberOfExperts, inputLength, numberOfFeatures, name=None , **kwargs):
        super(MixtureOfExperts, self).__init__(name=name, **kwargs)
        self.norm = layers.BatchNormalization()
        self.experts = [ExpertModel(inputLength, numberOfFeatures, name=f"expert_{i}") for i in range(numberOfExperts)]
        self.dense_units = 128
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.gating_network = layers.Dense(numberOfExperts, activation="softmax", name="gating_network")
        
    def call(self, inputs):
        x = self.norm(inputs)
        expert_outputs = [expert(x) for expert in self.experts] 
        x = self.flatten(inputs) # for the gating network to look at
        self.dense_units = x.shape[-1] # just compute directly one time? features * inputlen = shape[-1]
        self.fc1.units = self.dense_units # fully connected
        x = self.fc1(x)
        expert_weights = self.gating_network(x) # based on input get expert weights that lead to best output
        
        expert_outputs = tf.stack(expert_outputs, axis=1)  # Stack outputs to (batch_size, num_experts, num_classes)
        expert_weights = tf.expand_dims(expert_weights, axis=2)  # Expand dims to (batch_size, num_experts, 1)
        weighted_expert_outputs = tf.reduce_sum(expert_outputs * expert_weights, axis=1)  # Weighted sum
        print(f"weighted Expert output\033[91m{weighted_expert_outputs}\033[0m expertOuts {expert_outputs} Expert weights {expert_weights}")
        
        return weighted_expert_outputs





if __name__ == "__main__":
    main()