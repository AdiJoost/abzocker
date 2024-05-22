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
performanceDir = os.path.join(head[0], "abzocker", "performance")

modelName = "CNN_2D_ALTERNATIVE_LOSS.keras"

tensorboardLogDir = os.path.join(head[0], "abzocker", "Models", "logs", datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S"))

for dir in [dataDir, modelDir, tensorboardLogDir, performanceDir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
            
            
def main():

    # To load the Data into TensorDatasets
    batchSize = 256
    
    x_train = np.load(os.path.join(dataDir,"x_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(dataDir,"y_train.npy")).astype(np.float32)
    x_val = np.load(os.path.join(dataDir,"x_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(dataDir,"y_val.npy")).astype(np.float32)
    x_test = np.load(os.path.join(dataDir,"x_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(dataDir,"y_test.npy")).astype(np.float32)
    trainDataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batchSize)
    valDataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batchSize)
    
    print(f"Shape: {x_train.shape}")
    print(f"Shape: {y_train.shape}")
    print(f"Shape: {x_val.shape}")
    print(f"Shape: {y_val.shape}")
    print(f"Shape: {x_test.shape}")
    print(f"Shape: {y_test.shape}")

    
    
    # Model
    inputLength = x_train[0].shape[0]
    numberOfFeatures = x_train[0].shape[1]
    input_shape = (inputLength, numberOfFeatures, 1) 
    
    model = keras.models.Sequential() 
    model.add(keras.Input(shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(CNNBlock(filters=8,  kernel_size=(2, 1), pool_size=(2,1), name=f"nn1"))
    model.add(CNNBlock(filters=16, kernel_size=(1, numberOfFeatures), pool_size=(2,1), name=f"cnn2"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(CNNBlock(filters=32, kernel_size=(3, numberOfFeatures), pool_size=(2,1), name=f"cnn3"))
    model.add(CNNBlock(filters=64, kernel_size=(5, numberOfFeatures), pool_size=(2,1), name=f"cnn4"))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten(name=f"flatten"))
    
    model.add(layers.Dense(inputLength * numberOfFeatures * 64, activation="relu", name=f"_fc1"))
    model.add(layers.Dense(inputLength * numberOfFeatures, activation="relu", name=f"_fc2"))
    model.add(layers.Dense(1, name=f"output"))
     
     
     
    model.compile(optimizer=getOptimizer(),
                    loss=customLoss,
                    metrics=[metrics.MeanAbsoluteError(), metrics.R2Score()]) 
    history = model.fit(trainDataset, epochs=100, validation_data=valDataset, verbose=1, callbacks=getCallbacks()) 
    
    print(model.summary())

    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print(results)    
        
    modelPerformanceDir = os.path.join(performanceDir, modelName.split(".")[0])
    if not os.path.exists(modelPerformanceDir):
        os.makedirs(modelPerformanceDir)
    
    with open(os.path.join(modelPerformanceDir, "modelstats.txt"), 'w') as file:
        file.write(results)    



def getOptimizer():
    initial_learning_rate = 0.01
    lrSchedule =  ExponentialDecay(initial_learning_rate,
                                    decay_steps=80000,
                                    decay_rate=0.92,)
    return optimizers.Adam(learning_rate=lrSchedule)


def getCallbacks():
    earlyStop = EarlyStopping(monitor="loss", patience=8, min_delta=1)
    tensorboard = TensorBoard(log_dir=tensorboardLogDir)
    checkpoint = ModelCheckpoint( 
        filepath=os.path.join(modelDir,modelName),
        save_best_only=True, 
        save_weights_only=False
    )
    return [earlyStop, tensorboard, checkpoint]

@keras.saving.register_keras_serializable(package="my_custom_package", name="customLoss")
@tf.function()
def customLoss(y_true, y_pred, mda_weight=0.5):
    
    # Mean Absolute Error
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Mean Directional Accuracy
    true_diff = y_true[1:] - y_true[:-1]
    pred_diff = y_pred[1:] - y_pred[:-1]
    
    true_sign = tf.sign(true_diff)
    pred_sign = tf.sign(pred_diff)
    
    directional_accuracy = tf.reduce_mean(tf.cast(tf.equal(true_sign, pred_sign), tf.float32))
    
    # Combine MAE and MDA with the weighting factor
    combined_loss = mae - mda_weight * directional_accuracy
    
    return combined_loss


        
def loadModel():
    trainedModelPath = os.path.join(modelDir, modelName)
    custom_objects = {
        "CNNBlock": CNNBlock
    }
    
    # , "customLoss": customLoss
    model = keras.models.load_model(trainedModelPath, custom_objects=custom_objects, compile=False)
    return model, modelName.split(".")[0]




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
    
    
    
if __name__ == "__main__":
    main()