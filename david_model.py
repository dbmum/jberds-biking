import pandas as pd
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import r2_score
from keras.layers import Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')
df['total_bikes_rented'] = df['casual'] + df['registered']
df.drop(columns=['casual', 'registered'], inplace=True)
df['dteday'] = pd.to_datetime(df['dteday'], format='%m/%d/%y')
df['dteday'] = df['dteday'].astype('int64') // 10 ** 9
df = pd.get_dummies(df, columns=['season', 'hr', 'weathersit'])
X = df.drop(["total_bikes_rented"], axis=1)
y = df['total_bikes_rented']

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.5, shuffle=False)

# Scale the data
minmax_scaler = preprocessing.MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train)
X_test = minmax_scaler.transform(X_test)
X_val = minmax_scaler.transform(X_val)

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

# Create your model
model = Sequential()
model.add(Dense(144, input_dim=39, activation='tanh'))
model.add(Dropout(rate=0.25))
model.add(Dense(80, activation='tanh'))
model.add(Dropout(rate=0.25))
model.add(Dense(432, activation='tanh'))
model.add(Dropout(rate=0.25))
model.add(Dense(272, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(1, activation='linear'))

# Define the learning rate
learning_rate = 0.001

# Compile the model with the Adam optimizer and specified learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='MSE', optimizer=optimizer, metrics=['mean_squared_error'])

# Fit the model with early stopping
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0,
          callbacks=[early_stopping],
                    shuffle=False)

# Get predictions for the testing data
predictions = model.predict(X_val)

# Get the r^2
r2 = r2_score(y_val, predictions)
print(f'R^2 - val: {r2}')

# Get predictions for the testing data
predictions_test = model.predict(X_test)

# Get the r^2
r2_test = r2_score(y_test, predictions_test)
print(f'R^2 - test: {r2_test}')
