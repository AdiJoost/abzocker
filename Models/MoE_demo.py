import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Convert labels to one-hot encoding
num_classes = 10  # classes for images
y_train = to_categorical(y_train, num_classes)
y_test= to_categorical(y_test, num_classes)

from keras import layers, models

def create_expert_model(input_shape, num_outputs):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_outputs, activation='softmax')
    ])
    return model

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
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Evaluate the model on the test set
test_loss, test_accuracy = moe_model.evaluate([x_test, np.random.rand(len(x_test), num_experts)], y_test_one_hot)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')