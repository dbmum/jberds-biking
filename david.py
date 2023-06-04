import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import Dense
from keras_tuner import RandomSearch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.metrics import r2_score
from matplotlib import pyplot


df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')
df['total_bikes_rented'] = df['casual'] + df['registered']
df.drop(columns=['casual', 'registered'], inplace=True)
df['dteday'] = pd.to_datetime(df['dteday'], format='%m/%d/%y')
df['dteday'] = df['dteday'].astype('int64') // 10 ** 9
df = pd.get_dummies(df, columns=['season', 'hr', 'weathersit'])

X = df.drop(["total_bikes_rented"], axis=1)
y = df['total_bikes_rented']

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, random_state=42)

# Scale the data
minmax_scaler = preprocessing.MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train)
X_test = minmax_scaler.transform(X_test)
X_val = minmax_scaler.transform(X_val)


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('layer_1', min_value=16, max_value=512, step=32),
                           activation=hp.Choice("activation_1", ["relu", "tanh", "leaky_relu"]),
                           input_shape=(39,)))
    # Tune whether to use dropout.
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(units=hp.Int('layer_2', min_value=16, max_value=512, step=32),
                           activation=hp.Choice("activation_2", ["relu", "tanh", "leaky_relu"])))
    model.add(layers.Dense(units=hp.Int('layer_3', min_value=16, max_value=512, step=32),
                           activation=hp.Choice("activation_3", ["relu", "tanh", "leaky_relu"])))
    model.add(layers.Dense(units=hp.Int('layer_4', min_value=16, max_value=512, step=32),
                           activation=hp.Choice("activation_4", ["relu", "tanh", "leaky_relu"])))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])),
                  loss=hp.Choice('loss', values=['MSE', ]),
                  metrics=['mean_squared_error'])

    return model

tuner = RandomSearch(
    build_model,
    objective="mean_squared_error",
    max_trials=50,
    executions_per_trial=3,
    directory='my_dir',
    project_name='my_project'
)

tuner.search(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=100)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_hps_config = best_hps.get_config()

# Unpack the best parameters
layer_1_units = best_hps_config['values']['layer_1']
activation_1 = best_hps_config['values']['activation_1']
dropout = best_hps_config['values']['dropout']
layer_2_units = best_hps_config['values']['layer_2']
activation_2 = best_hps_config['values']['activation_2']
layer_3_units = best_hps_config['values']['layer_3']
activation_3 = best_hps_config['values']['activation_3']
layer_4_units = best_hps_config['values']['layer_4']
activation_4 = best_hps_config['values']['activation_4']
learning_rate = best_hps_config['values']['learning_rate']
loss = best_hps_config['values']['loss']

# Print the best parameters
print("Best Hyperparameters:")
print("layer_1_units =", layer_1_units)
print("activation_1 =", activation_1)
print("dropout =", dropout)
print("layer_2_units =", layer_2_units)
print("activation_2 =", activation_2)
print("layer_3_units =", layer_3_units)
print("activation_3 =", activation_3)
print("layer_4_units =", layer_4_units)
print("activation_4 =", activation_4)
print("learning_rate =", learning_rate)
print("loss =", loss)


# my results after 3 hs
# Best mean_squared_error So Far: 781.0992431640625
# Total elapsed time: 03h 24m 42s
# Best Hyperparameters:
# layer_1_units = 144
# activation_1 = tanh
# dropout = False
# layer_2_units = 80
# activation_2 = tanh
# layer_3_units = 432
# activation_3 = tanh
# layer_4_units = 272
# activation_4 = relu
# learning_rate = 0.001
# loss = MSE




model = Sequential() # Sequential just means the network doesn't have loops--the outputs of one layer of neurons go to the next layer of neurons

model.add(Dense(144, input_dim=39, activation='tanh'))
model.add(Dense(80, activation='tanh'))
model.add(Dense(432, activation='tanh'))
model.add(Dense(272, activation='relu'))
model.add(Dense(1, activation='linear')) # Our last layer doesn't need a non-linear activation function, unless it is useful for the type of answer we want

# Define the learning rate
learning_rate = 0.001

# Compile the model with the Adam optimizer and specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# The ouput layer should have the same number of neurons as outputs you are generating. In this case, it is just producing one number.

# Compile model
model.compile(loss='MSE', optimizer=optimizer, metrics=['mean_squared_error'])
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