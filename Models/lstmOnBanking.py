from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np

import os
import logging
logger = logging.getLogger(__name__)

numberOfSteps = 20
numberOfFeatures = 91

epochs = 200
batches = 200


cwd = os.getcwd()
head = cwd.split("abzocker")
modelDir = os.path.join(head[0], "abzocker", "Models", "Models")
performanceDir = os.path.join(head[0], "abzocker", "performance")

for dir in [modelDir, performanceDir]:
    if not os.path.exists(dir):
        os.makedirs(dir)
    
modelName = "lstm.keras"

def main():
    try:
        logger.info("Loaded Dataset")
        x_train, y_train, x_test, y_test, x_val, y_val = loadSets()
        logger.info("creating model")
        model = getModel()
        logger.info("Training Model")
        model.fit(x_train, y_train, epochs=epochs, batch_size=batches, validation_data=(x_val, y_val), shuffle="True", callbacks=getCallback())
        logger.info("Training completed")
        score = model.evaluate(x_test, y_test)
        modelPerformanceDir = os.path.join(performanceDir, modelName.split(".")[0])
        if not os.path.exists(modelPerformanceDir):
            os.makedirs(modelPerformanceDir)
        with open(os.path.join(modelPerformanceDir, "modelstats.txt"), 'w') as file:
            file.write(score)    
    
        logger.info(f"Last Model got a score of: {score}")
    except Exception as e:
        logger.error(f"Error in execution {str(e)}")



def loadSets():
    x_train = np.load(getDataPath("x_train.npy"))
    y_train = np.load(getDataPath("y_train.npy"))
    x_test = np.load(getDataPath("x_test.npy"))
    y_test = np.load(getDataPath("y_test.npy"))
    x_val = np.load(getDataPath("x_val.npy"))
    y_val = np.load(getDataPath("y_val.npy"))
    return x_train, y_train, x_test, y_test, x_val, y_val

def getDataPath(filename):
    cwd = os.getcwd()
    head = cwd.split("abzocker")
    return os.path.join(head[0], "abzocker", "data", filename)
 
def getModel():
    #Optimizer
    INITIAL_LEARNING_RATE = 0.01
    DECAY_STEPS = 100
    DECAY_RATE = 0.99
    DECAY_STAIRCASE = True

    exponetialDecay = ExponentialDecay(
            INITIAL_LEARNING_RATE,
            DECAY_STEPS,
            DECAY_RATE,
            staircase=DECAY_STAIRCASE
        )

    optimizer = Adam(learning_rate=exponetialDecay)

    model = Sequential()

    model.add(Input((numberOfSteps, numberOfFeatures)))
    model.add(
        LSTM(20, activation='relu', return_sequences=True, dropout=0.4)
    )
    model.add(
        LSTM(20, activation='relu', dropout=0.4)
    )
    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss='mae', metrics=[mean_absolute_error])
    return model

def getCallback():
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(modelDir, modelName),
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        save_weights_only=False
    )
    earlyStopping = EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=0)
    return [checkpoint, earlyStopping]

def loadModel():
    trainedModelPath = os.path.join(modelDir, modelName)
    model = load_model(trainedModelPath)
    return model, modelName.split(".")[0]


def _setLogger():
    cwd = os.getcwd()
    logPath = os.path.join(cwd, "log")
    os.makedirs(logPath, exist_ok=True)
    logname = os.path.join(logPath, "lstmOnBanking.log")
    logging.basicConfig(filename=logname, level=logging.INFO)

if __name__ == "__main__":
    _setLogger()
    main()