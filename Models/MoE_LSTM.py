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
from keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import datetime
import keras



cwd = os.getcwd()
head = cwd.split("abzocker")
dataDir = os.path.join(head[0], "abzocker", "data")    
modelDir = os.path.join(head[0], "abzocker", "Models", "Models")
perfromanceDir = os.path.join(head[0], "abzocker", "performance")

modelName = "MoE_LSTM.keras"

tensorboardLogDir = os.path.join(head[0], "abzocker", "Models", "logs", datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S"))

for dir in [dataDir, modelDir, tensorboardLogDir, perfromanceDir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
            
            
def main():
    
    # To load the Data into TensorDatasets
    batchSize = 128
    
    x_train = np.load(os.path.join(dataDir,"x_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(dataDir,"y_train.npy")).astype(np.float32)
    x_val = np.load(os.path.join(dataDir,"x_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(dataDir,"y_val.npy")).astype(np.float32)
    x_test = np.load(os.path.join(dataDir,"x_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(dataDir,"y_test.npy")).astype(np.float32)
    
    print(f"Shape: {x_train.shape}")
    print(f"Shape: {y_train.shape}")
    print(f"Shape: {x_val.shape}")
    print(f"Shape: {y_val.shape}")
    print(f"Shape: {x_test.shape}")
    print(f"Shape: {y_test.shape}")
    

    
    # Model
    numberOfExperts = 10
    
    inputLength = x_train[0].shape[0]
    numberOfFeatures = x_train[0].shape[1]
    input_shape = (inputLength, numberOfFeatures, 1) 
    
    model = MixtureOfExpertsLSTM(numberOfExperts, inputLength, numberOfFeatures, name="mixture_of_experts_LSTM", input_shape=input_shape)
    
    model.compile(optimizer=getOptimizer(),
                    loss=losses.MeanAbsoluteError(),
                    metrics=[metrics.MeanAbsoluteError(), metrics.R2Score()]) 

    # train model
    history = model.fit(x_train, y_train, epochs=100, batch_size=batchSize, validation_data=(x_val, y_val), verbose=1, callbacks=getCallbacks()) # 
    
    print(model.summary())

    keras.models.save_model(model, model, overwrite=True)
    
        
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print(results)    
    
    with open(os.path.join(perfromanceDir, modelName.split(".")[0], "modelstats.txt"), 'w') as file:
        file.write(results)    


def getOptimizer():
    initial_learning_rate = 0.01
    lrSchedule =  ExponentialDecay(initial_learning_rate,
                                    decay_steps=80000,
                                    decay_rate=0.92,)
    return optimizers.Adam(learning_rate=lrSchedule)


def getCallbacks():
    earlyStop = EarlyStopping(monitor="loss", patience=4, min_delta=1)
    tensorboard = TensorBoard(log_dir=tensorboardLogDir, )
    checkpoint = ModelCheckpoint( 
        filepath=os.path.join(modelDir, modelName),
        save_best_only=True, 
        save_weights_only=False,
        mode='min',
        monitor='val_loss',
    
    )
    return [earlyStop, checkpoint] #, tensorboard
        

def loadModel():
    trainedModelPath = os.path.join(modelDir, modelName)
    custom_objects = {
        "MixtureOfExperts": MixtureOfExpertsLSTM,
        "ExpertModel": ExpertModelLSTM,
    }
    model = keras.models.load_model(trainedModelPath, custom_objects=custom_objects)
    return model, modelName.split(".")[0]

@keras.saving.register_keras_serializable(package="my_custom_package", name="MixtureOfExpertsLSTM")
class MixtureOfExpertsLSTM(tf.keras.Model):
    def __init__(self, numberOfExperts, inputLength, numberOfFeatures, name=None, input_shape=None, **kwargs):
        super(MixtureOfExpertsLSTM, self).__init__(name=name, **kwargs)
        self.numberOfExperts = numberOfExperts
        self.inputLength = inputLength
        self.numberOfFeatures = numberOfFeatures
        self.name = name
        self.input_shape = input_shape
        
        self.norm = layers.BatchNormalization()
        self.experts = [ExpertModelLSTM(inputLength, numberOfFeatures, name=f"expert_{i}") for i in range(numberOfExperts)]
        self.dense_units = 128
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.fc2 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc2")
        self.gating_network = layers.Dense(numberOfExperts, activation="softmax", name="gating_network")
        
    def call(self, inputs):
        x = self.norm(inputs)
        expert_outputs = [expert(x) for expert in self.experts] 
        x = self.flatten(inputs)  # for the gating network to look at
        self.dense_units = x.shape[-1]  
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



    
@keras.saving.register_keras_serializable(package="my_custom_package", name="ExpertModelLSTM")
class ExpertModelLSTM(tf.keras.layers.Layer):
    def __init__(self, inputLength, numberOfFeatures, name=None, **kwargs):
        super(ExpertModelLSTM, self).__init__(name=name, **kwargs)
        # save inputs for get_config
        self.inputLength = inputLength
        self.numberOfFeatures = numberOfFeatures
        self.name = name
        
        # Define layers
        self.reshape = layers.Reshape((inputLength, numberOfFeatures))
        self.lstm1 = layers.LSTM(20, activation='relu', return_sequences=True, dropout=0.4)
        self.lstm2 = layers.LSTM(20, activation='relu', dropout=0.4)
        self.dense = layers.Dense(1)
    
    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        # save constructor args
        config["inputLength"] = self.inputLength
        config["numberOfFeatures"] = self.numberOfFeatures
        config["name"] = self.name
        return config




if __name__ == "__main__":
    main()