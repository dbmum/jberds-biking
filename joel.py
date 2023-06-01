# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

print(tf.__version__)

# Make NumPy printouts easier to read
np.set_printoptions(precision=3, suppress=True)


# Load the dataset
bike_rentals = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv")
bike_rentals_holdout = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes_december.csv")

# Define X/Y
bike_rentals["total_rentals"] = bike_rentals.casual + bike_rentals.registered

# Set aside the test set immediately
bike_rentals = train_test_split()