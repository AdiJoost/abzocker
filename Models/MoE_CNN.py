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
from keras.optimizers.schedules import ExponentialDecay, CosineDecay
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime

def main():
    
    # Load Data
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    dataPath = os.path.join(head[0], "abzocker", "generatedData", "timeSeries")    
    
    X = np.load(os.path.join(dataPath, "X_combined.npy"))
    y = np.load(os.path.join(dataPath, "Y_combined.npy"))
        
        
    # Prepare Data
    X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)
        
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5)
    
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    #testDataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(64)
    valDataset = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(32)
    
    
    # Model
    numberOfExperts = 30
    inputLength = x_train[0].shape[0]
    numberOfFeatures = x_train[0].shape[1]
    input_shape = (inputLength, numberOfFeatures, 1) 
    moe_model = MixtureOfExperts(numberOfExperts, inputLength, numberOfFeatures, name="mixture_of_experts", input_shape=input_shape)
    
    
    # Callbacks
    initial_learning_rate = 0.01
    lr_schedule = ExponentialDecay(initial_learning_rate,
                                    decay_steps=80000,
                                    decay_rate=0.96,)
    
    earlyStop = EarlyStopping(monitor="loss", patience=4)
    
    moe_model.compile(optimizer=optimizers.Adam(),
                    loss=losses.MeanAbsoluteError(),
                    metrics=[metrics.MeanAbsoluteError()]) 
    
    logDir = os.path.join(head[0], "abzocker", "Models", "logs", datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=logDir, histogram_freq=1)
    
    checkpoint_filepath = os.path.join(head[0], "abzocker", "Models", "checkpoints", "checkpoint.model.keras")
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True
    )
    # train model
    history = moe_model.fit(trainDataset, epochs=400, validation_data=valDataset, verbose=1, callbacks=[tensorboard, earlyStop, checkpoint])
    
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


class MixtureOfExperts(tf.keras.Model):
    def __init__(self, numberOfExperts, inputLength, numberOfFeatures, name=None , **kwargs):
        super(MixtureOfExperts, self).__init__(name=name, **kwargs)
        self.norm = layers.BatchNormalization()
        self.experts = [ExpertModel(inputLength, numberOfFeatures, name=f"expert_{i}") for i in range(numberOfExperts)]
        self.dense_units = 128
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.fc2 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc2")
        self.gating_network = layers.Dense(numberOfExperts, activation="softmax", name="gating_network")
        
    def call(self, inputs):
        x = self.norm(inputs)
        expert_outputs = [expert(x) for expert in self.experts] 
        x = self.flatten(inputs) # for the gating network to look at
        self.dense_units = x.shape[-1] # just compute directly one time? features * inputlen = shape[-1]
        self.fc1.units = self.dense_units # fully connected
        self.fc2.units = int(self.dense_units*0.7)
        x = self.fc1(x)
        
        expert_weights = self.gating_network(x) # based on input get expert weights that lead to best output
        
        expert_outputs = tf.stack(expert_outputs, axis=1)  # Stack outputs to (batch_size, num_experts, num_classes)
        expert_weights = tf.expand_dims(expert_weights, axis=2)  # Expand dims to (batch_size, num_experts, 1)
        weighted_expert_outputs = tf.reduce_sum(expert_outputs * expert_weights, axis=1)  # Weighted sum
        
        return weighted_expert_outputs





if __name__ == "__main__":
    main()