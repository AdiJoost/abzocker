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
        
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    num_classes = 10
    num_experts = 3
    input_shape = input_shape=(32, 32, 3)
    
    # input shape is passed vie **kwargs
    moe_model = MixtureOfExperts(num_experts, num_classes, name="mixture_of_experts", input_shape=input_shape)
    
    moe_model.compile(optimizer=optimizers.Adam(),
                    loss=losses.SparseCategoricalCrossentropy(),
                    metrics=[metrics.SparseCategoricalAccuracy()])
    
    moe_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=2)
    
    print(moe_model.summary())




class CNNBlock(layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), activation="relu", name=None, **kwargs):
        super(CNNBlock, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, activation=activation, padding="same", name=f"{name}_conv")
        self.maxPool = layers.MaxPooling2D(name=f'{name}_pool')
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.maxPool(x)
        return x 


class ExpertModel(tf.keras.Layer):
    def __init__(self, num_classes, name=None, **kwargs):
        super(ExpertModel, self).__init__(name=name, **kwargs)
        # define model structure/layers/components and store them in class
        self.cnn1 = CNNBlock(32, name=f"{name}_cnn1")
        self.cnn2 = CNNBlock(64, name=f"{name}_cnn2")
        self.cnn3 = CNNBlock(128, name=f"{name}_cnn3")
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.dense_units = 128 # default 
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.fc2 = layers.Dense(num_classes, activation="softmax", name=f"{name}_fc2")
    
    
    def call(self, inputs):
        # define how the forward pass looks like
        # assemble the components from the init into a network
        # return prediction of model
        x = self.cnn1(inputs) #apply cnn1 to inputs 
        x = self.cnn2(x)
        x = self.flatten(x)
        self.dense_units = x.shape[-1] # number of units in the flattened x 
        self.fc1.units = self.dense_units
        x = self.fc1(x)
        return self.fc2(x)
    



class MixtureOfExperts(tf.keras.Model):
    def __init__(self, num_experts, num_classes, name=None , **kwargs):
        super(MixtureOfExperts, self).__init__(name=name, **kwargs)
        self.experts = [ExpertModel(num_classes, name=f"expert_{i}") for i in range(num_experts)]
        self.dense_units = 128
        self.flatten = layers.Flatten(name=f"{name}_flatten")
        self.fc1 = layers.Dense(self.dense_units, activation="relu", name=f"{name}_fc1")
        self.gating_network = layers.Dense(num_experts, activation="softmax", name="gating_network")
        
    def call(self, inputs):
        expert_outputs = [expert(inputs) for expert in self.experts] 
        x = self.flatten(inputs)
        self.dense_units = x.shape[-1]
        self.fc1.units = self.dense_units
        x = self.fc1(x)
        expert_weights = self.gating_network(x)
        
        # Combine outputs of experts using weights
        expert_outputs = tf.stack(expert_outputs, axis=1)  # Stack outputs to (batch_size, num_experts, num_classes)
        expert_weights = tf.expand_dims(expert_weights, axis=2)  # Expand dims to (batch_size, num_experts, 1)
        weighted_expert_outputs = tf.reduce_sum(expert_outputs * expert_weights, axis=1)  # Weighted sum
        
        return weighted_expert_outputs



    def model(self):
            inputs = tf.keras.Input(shape=(32, 32, 3))
            outputs = self.call(inputs)
            return tf.keras.Model(inputs=inputs, outputs=outputs)



if __name__ == "__main__":
    main()