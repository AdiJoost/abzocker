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
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime
import keras


def main():
    
    # Load Data
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    dataPath = os.path.join(head[0], "abzocker", "data")    
    

    # To load the Data into TensorDatasets
    batchSize = 128
    
    x_train = np.load(os.path.join(dataPath,"x_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(dataPath,"y_train.npy")).astype(np.float32)
    x_val = np.load(os.path.join(dataPath,"x_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(dataPath,"y_val.npy")).astype(np.float32)
    #x_test = np.load(os.path.join(dataPath,"x_test.npy")).astype(np.float32)
    #y_test = np.load(os.path.join(dataPath,"y_test.npy")).astype(np.float32)    
    
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batchSize)
    valDataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batchSize)
    #testDataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchSize)
    
    print(f"Shape: {x_train.shape}")
    print(f"Shape: {y_train.shape}")
    print(f"Shape: {x_val.shape}")
    print(f"Shape: {y_val.shape}")
    #print(f"Shape: {x_test.shape}")
    #print(f"Shape: {y_test.shape}")       

    
    
    # Model
    numberOfExperts = 10
    inputLength = x_train[0].shape[0]
    numberOfFeatures = x_train[0].shape[1]
    input_shape = (inputLength, numberOfFeatures, 1) 
    model = MixtureOfExperts(numberOfExperts, inputLength, numberOfFeatures, name="mixture_of_experts", input_shape=input_shape)
    
    
    # Callbacks
    initial_learning_rate = 0.01
    lr_schedule = ExponentialDecay(initial_learning_rate,
                                    decay_steps=80000,
                                    decay_rate=0.92,)
    
    earlyStop = EarlyStopping(monitor="loss", patience=4, min_delta=1)
    
    model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule),
                    loss=losses.MeanAbsoluteError(),
                    metrics=[metrics.MeanAbsoluteError(), metrics.R2Score()]) 
    
    logDir = os.path.join(head[0], "abzocker", "Models", "logs", datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=logDir, histogram_freq=1)
    
    
    
    checkpoint_filepath = os.path.join(head[0], "abzocker", "Models", "checkpoints", "MoE_CNN_2D.keras")
    checkpoint = ModelCheckpoint( 
        filepath=checkpoint_filepath,
        save_best_only=True, 
        save_weights_only=False
    )

        
    # train model
    history = model.fit(trainDataset, epochs=100, validation_data=valDataset, verbose=1, callbacks=[tensorboard, earlyStop]) # , checkpoint
    
    # print(model.summary())
    # print(f"training loss", history.history["loss"])
    
    # predictions = model.predict(x_test)
    # print("Predictions on Test set")
    # mse = mean_squared_error(y_test, predictions)
    # mae = mean_absolute_error(y_test,  predictions)
    # print("MSE:  ", mse)
    # print("MAE:  ", mae)
    # print("RMSE: ", np.sqrt(mse))

    keras.models.save_model(model, checkpoint_filepath, overwrite=True)


@keras.saving.register_keras_serializable(package="my_custom_package", name="CNNBlock")
class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), pool_size=(3,1), activation="relu", name=None, **kwargs):
        super(CNNBlock, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.activation = activation
        self.name = name
        
        self.conv = layers.Conv2D(filters, kernel_size, activation=activation, padding="same", name=f"{name}_conv")
        self.maxPool = layers.MaxPooling2D(pool_size=pool_size, name=f'{name}_pool') # pool size
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.maxPool(x)
        return x 
    
    def get_config(self):
        config = super().get_config()
        # save constructor args
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["pool_size"] = self.pool_size
        config["activation"] = self.activation
        config["name"] = self.name
        
        return config
    

    
@keras.saving.register_keras_serializable(package="my_custom_package", name="ExpertModel")
class ExpertModel(tf.keras.layers.Layer):
    def __init__(self, inputLength, numberOfFeatures, name=None, **kwargs):
        super(ExpertModel, self).__init__(name=name, **kwargs)
        # save inputs for get_config
        self.inputLength = inputLength
        self.numberOfFeatures = numberOfFeatures
        self.name = name
        
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
    
    def get_config(self):
        config = super().get_config()
        # save constructor args
        config["inputLength"] = self.inputLength
        config["numberOfFeatures"] = self.numberOfFeatures
        config["name"] = self.name
    
        return config


@keras.saving.register_keras_serializable(package="my_custom_package", name="MixtureOfExperts")
class MixtureOfExperts(tf.keras.Model):
    def __init__(self, numberOfExperts, inputLength, numberOfFeatures, name=None, input_shape=None, **kwargs):
        super(MixtureOfExperts, self).__init__(name=name, **kwargs)
        self.numberOfExperts = numberOfExperts
        self.inputLength = inputLength
        self.numberOfFeatures = numberOfFeatures
        self.name = name
        self.input_shape = input_shape
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
        x = self.flatten(inputs)  # for the gating network to look at
        self.dense_units = x.shape[-1]  # just compute directly one time? features * inputlen = shape[-1]
        self.fc1.units = self.dense_units  # fully connected
        self.fc2.units = int(self.dense_units * 0.7)
        x = self.fc1(x)
        
        expert_weights = self.gating_network(x)  # based on input get expert weights that lead to best output
        
        expert_outputs = tf.stack(expert_outputs, axis=1)  # Stack outputs to (batch_size, num_experts, num_classes)
        expert_weights = tf.expand_dims(expert_weights, axis=2)  # Expand dims to (batch_size, num_experts, 1)
        weighted_expert_outputs = tf.reduce_sum(expert_outputs * expert_weights, axis=1)  # Weighted sum
        
        return weighted_expert_outputs
    

        
    def get_config(self):
        config = super().get_config()
        # save constructor args
        config["numberOfExperts"] = self.numberOfExperts
        config["numberOfFeatures"] = self.numberOfFeatures
        config["inputLength"] = self.inputLength
        config["name"] = self.name
        config["input_shape"] = self.input_shape

        return config


    @classmethod
    def from_config(cls, config):
        return cls(
            numberOfExperts=config['numberOfExperts'],
            inputLength=config['inputLength'],
            numberOfFeatures=config['numberOfFeatures'],
            name=config['name'],
            input_shape=config['input_shape']
        )



if __name__ == "__main__":
    main()