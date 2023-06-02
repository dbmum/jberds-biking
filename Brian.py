import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from matplotlib import pyplot

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')
df['target'] = df['casual'] + df['registered']
df['dteday'] = pd.to_datetime(df['dteday'])
initial_date = pd.to_datetime('2011-01-01')
df['days_since_initial'] = (df['dteday'] - initial_date).dt.days
df.drop(columns='dteday', inplace=True)
print("DataFrame shape:", df.shape)
print("Column names:", df.columns)
columns_to_drop = ['target', 'casual', 'registered']
X = df.drop(columns=columns_to_drop, axis = 1)
# columns = X.columns.tolist()
# X = pd.get_dummies(X, columns=columns)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 17)

# Scale the data
minmax_scaler = preprocessing.MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train) # fit the scale to the training data
X_test = minmax_scaler.transform(X_test) # use the same scale on the testing data

# Initialize the Neural Network
model = Sequential() # Sequential just means the network doesn't have loops--the outputs of one layer of neurons go to the next layer of neurons

model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu')) # This layer has 16 neurons. They are each connected (dense) to the input neurons.
# Note: We want the input dimension to match the number of features at our input layer
model.add(Dropout(0.2))
# Add another "hidden layer"
model.add(Dense(8, activation = 'relu')) # This layer has 8 neurons
model.add(Dropout(0.2))
# Add the "output layer"

model.add(Dense(1, activation='linear')) # Our last layer doesn't need a non-linear activation function, unless it is useful for the type of answer we want
# The ouput layer should have the same number of neurons as outputs you are generating. In this case, it is just producing one number.

# Compile model
model.compile(loss='MSE', optimizer= 'Adam', metrics=['mean_squared_error'])

history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 500, verbose = 0)

# Evaluate the model on the training data
_, train_mse = model.evaluate(X_train, y_train, verbose = 1)

# Evaluate the model on the testing data
_, test_mse = model.evaluate(X_test, y_test, verbose = 1)

# Get predictions for the testing data
predictions = model.predict(X_test)

# Get the r^2
r2 = r2_score(y_test, predictions)
print(f'R^2: {r2}')

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()