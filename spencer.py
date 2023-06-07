import pandas as pd
# Import the libraries we need 
# TensorFlow and tf.keras
import tensorflow as tf
# Commonly used modules
import numpy as np
from tensorflow import keras
# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model

class SpencerModel:

    def __init__(self):
        self.history = None
        self.model = None

    def _preprocess(self, dataset: pd.DataFrame) -> pd.DataFrame:
        #self.model = dataset.drop(columns=["dteday"])
        pass

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        self._preprocess(X_train)
        norm = MinMaxScaler().fit(X_train)

# transform training data
        X_train = norm.transform(X_train)
        inputs = Input(shape=(len(X_train[0]),))
        x = Dense(80, activation='tanh')(inputs)
        x = Dense(432, activation='tanh')(x)
        x = Dense(272, activation='relu')(x)
        output1 = Dense(1,activation='relu')(x)
        output2 = Dense(1, activation='relu')(x)

        model = Model(inputs=inputs, outputs=[output1, output2])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=30) #if it doesn't go down after 30 more epochs, just stop
        opt = keras.optimizers.Adam(learning_rate=.001)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mse'])

        history = model.fit(X_train, y_train, epochs=500, batch_size=8,callbacks=[early_stop])
        self.model = model
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = self._preprocess(X)
        predictions = self.model.predict(X_test)
        return (predictions[0] + predictions[1]) #this isn't quite the right dimensions
        pass 


# This can show our 95% Confidence Interval
#(y_test-predicted).quantile(.95)