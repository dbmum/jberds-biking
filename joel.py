# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Make NumPy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

# Load the dataset
bike_rentals = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv")
bike_rentals_holdout = pd.read_csv(
    "https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes_december.csv")

# Define X/Y
bike_rentals["total_rentals"] = bike_rentals.casual + bike_rentals.registered
y = bike_rentals.total_rentals
X = bike_rentals.drop(["total_rentals", "dteday"], axis='columns')

# Set aside the test set immediately
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the data
# normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(np.array(X_train))
# print(normalizer.mean.numpy())


scaler = MinMaxScaler()
print(X_train.head())
X_train_normal = scaler.fit_transform(X_train)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])