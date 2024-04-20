import tensorflow as tf
from keras import layers, losses, optimizers, metrics
from keras.utils import to_categorical
import numpy as np
from keras.datasets import cifar10

"""
General implementation of a mixture of experts model using the keras subclassing api
example is for the cifar10 dataset of images (32,32,3)

"""



def main():
    trainMoE(3,10,6)
        
def trainMoE(numberOfExperts=3, inputLength=10, numberOfFeatures=6):
    
    input_shape = (inputLength, numberOfFeatures, 1) # len, features, how many of those per timestep
    
    moe_model = MixtureOfExperts(numberOfExperts, inputLength, numberOfFeatures, name="mixture_of_experts", input_shape=input_shape)
    
    moe_model.compile(optimizer=optimizers.Adam(),
                    loss=losses.MeanAbsoluteError(),
                    metrics=[metrics.MeanAbsoluteError()]) # review loss/metric, maybe custom
    
    moe_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)
    
    print(moe_model.summary())




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
        self.experts = [ExpertModel(inputLength, numberOfFeatures, name=f"expert_{i}") for i in range(numberOfExperts)]
        self.dense_units = 128
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.gating_network = layers.Dense(numberOfExperts, activation="softmax", name="gating_network")
        
    def call(self, inputs):
        expert_outputs = [expert(inputs) for expert in self.experts] 
        x = self.flatten(inputs) # for the gating network to look at
        self.dense_units = x.shape[-1] # just compute directly one time? features * inputlen = shape[-1]
        self.fc1.units = self.dense_units # fully connected
        x = self.fc1(x)
        expert_weights = self.gating_network(x) # based on input get expert weights that lead to best output
        
        expert_outputs = tf.stack(expert_outputs, axis=1)  # Stack outputs to (batch_size, num_experts, num_classes)
        expert_weights = tf.expand_dims(expert_weights, axis=2)  # Expand dims to (batch_size, num_experts, 1)
        weighted_expert_outputs = tf.reduce_sum(expert_outputs * expert_weights, axis=1)  # Weighted sum
        
        return weighted_expert_outputs





if __name__ == "__main__":
    main()