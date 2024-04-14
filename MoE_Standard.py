from keras.utils import to_categorical
from keras import layers, models
import tensorflow as tf
from keras import layers, losses, optimizers, metrics
import numpy as np
from keras.datasets import cifar10


# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Convert labels to one-hot encoding
num_classes = 10  # classes for images
y_train = to_categorical(y_train, num_classes)
y_test= to_categorical(y_test, num_classes)


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
    


# Expert Model
def create_expert_model(input_shape, num_outputs):
    model = models.Sequential([
        CNNBlock(32, input_shape=input_shape),
        CNNBlock(64),
        CNNBlock(128),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_outputs, activation='softmax')
    ])
    return model

# Gating Model
def create_gating_model(num_experts, num_outputs):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(num_experts,)),
        layers.Dense(num_outputs, activation='softmax')
    ])
    return model

# Mixture of Experts (MOE) Model
def create_moe_model(input_shape, num_experts, num_outputs):
    expert_input = layers.Input(shape=input_shape, name='expert_input')
    expert_model = create_expert_model(input_shape, num_outputs)
    expert_output = expert_model(expert_input)
    gating_input = layers.Input(shape=(num_experts,), name='gating_input')
    gating_model = create_gating_model(num_experts, num_outputs)
    gating_output = gating_model(gating_input)
    # Reshape gating output to match expert output shape
    gating_output = layers.Reshape((num_outputs, 1))(gating_output)
    gating_output = layers.Lambda(lambda x: layers.Flatten()(x))(gating_output)
    # Multiply expert and gating outputs
    mixture_output = layers.Multiply()([expert_output, gating_output])
    moe_model = models.Model(inputs=[expert_input, gating_input], outputs=mixture_output)
    return moe_model

# Example usage:
input_shape = (32, 32, 3)  # 
num_experts = 5  # Adjust based on the desired number of experts
num_outputs = 10  # Number of Cifar10 imageclasses
epochs = 30
moe_model = create_moe_model(input_shape, num_experts, num_outputs)
moe_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Train and Test
import matplotlib.pyplot as plt
history = moe_model.fit(
    [x_train, np.random.rand(len(x_train), num_experts)],
    y_train,
    epochs=epochs,
    batch_size=64)

plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Evaluate the model on the test set
test_loss, test_accuracy = moe_model.evaluate([x_test, np.random.rand(len(x_test), num_experts)], y_test_one_hot)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')