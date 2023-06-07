import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score
from matplotlib import pyplot
from keras import regularizers
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
import numpy as np
from keras.utils import timeseries_dataset_from_array

# Important values for our model, these values define that our LSTM layers will receive 24 items of data at a time which represent 1 day.
timesteps=24
window_size=24

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')

def time_data_sliding_window_generator(X_train):
    """
    :param X_train: Takes a dataset to transform
    :return: Returns a dataset which contains the dimensions [batch_size, timesteps, window_size, data]
    """
    stride = timesteps  # Number of timesteps to skip between windows

    # Get the number of samples and number of features in the input data
    num_samples, num_features = X_train.shape

    # Calculate the number of windows based on the window size and stride
    num_windows = (num_samples - window_size) // stride + 1

    # Initialize an empty array to store the windowed data
    windowed_data = np.zeros((num_samples, window_size, num_features))

    # Create the sliding window dataset
    for i in range(num_windows):
        start_index = i * stride
        end_index = start_index + window_size
        window = X_train[start_index:end_index, :]
        windowed_data[end_index-1] = window

    # Print the shape of the sliding window dataset
    print("Sliding window dataset shape:", windowed_data.shape)
    return windowed_data


def dataset_expander(*args):
    """
    :param args: A tuple of numpy arrays to transform
    :return: A tuple of numpy arrays with the last (window_size - 1) * 2 rows added onto array_2.
             This enables us to later drop these rows without having NaN values at the beginning of our test and holdout set
    """
    modified_data = []
    for item in args:
        # Scale the data
        array_1, array_2 = item
        combined_array = np.concatenate((array_1[-((window_size - 1) * 2):], array_2), axis=0)
        modified_data.extend([array_1, combined_array])
    return tuple(modified_data)

def dataset_trimmer(*args):
    """
    :param args: A tuple of numpy arrays to trim
    :return: The trimmed tuple of numpy arrays
    """
    modified_data = []
    for item in args:
        modified_item = item[(window_size - 1) * 2:]
        modified_data.append(modified_item)
    return tuple(modified_data)

def data_scaler(*args):
    """
    :param args: Tuple of Dataframes to apply a MinMaxScale operation to
    :return: The list of dataframes that have been scaled.
    """
    modified_dataframes = []
    for item in args:
        # Scale the data
        minmax_scaler = MinMaxScaler()
        df1_1, df1_2 = item
        df1_1 = minmax_scaler.fit_transform(df1_1)
        df1_2 = minmax_scaler.transform(df1_2) # use the same scale on the testing data
        modified_dataframes.extend([df1_1, df1_2])
    return tuple(modified_dataframes)

def preprocess_data(df, test_size=.3):
    """
    :param df: The dataframe for preprocessing
    :param test_size: The amount of the dataframe to use for testing
    :return: The various dataframes needed by our model's branches
    """
    # Preprocessing to create 3 branches
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['month'] = df['dteday'].dt.month
    initial_date = pd.to_datetime('2011-01-01')
    df['days_since_initial'] = (df['dteday'] - initial_date).dt.days
    # Create lag columns for 'casual' and 'registered' variables
    df['casual_lag1'] = df['casual'].shift(1)
    df['casual_lag2'] = df['casual'].shift(2)
    df['casual_lag3'] = df['casual'].shift(3)
    df['registered_lag1'] = df['registered'].shift(1)
    df['registered_lag2'] = df['registered'].shift(2)
    df['registered_lag3'] = df['registered'].shift(3)

    window_size = 24  # Define the window size for rolling calculations

    # Calculate rolling mean
    df['casual_rolling_mean'] = df['casual'].shift().rolling(window=window_size).mean()
    df['registered_rolling_mean'] = df['registered'].shift().rolling(window=window_size).mean()

    # Calculate rolling standard deviation
    df['casual_rolling_std'] = df['casual'].shift().rolling(window=window_size).std()
    df['registered_rolling_std'] = df['registered'].shift().rolling(window=window_size).std()

    # Calculate rolling minimum and maximum
    df['casual_rolling_min'] = df['casual'].shift().rolling(window=window_size).min()
    df['casual_rolling_max'] = df['casual'].shift().rolling(window=window_size).max()
    df['registered_rolling_min'] = df['registered'].shift().rolling(window=window_size).min()
    df['registered_rolling_max'] = df['registered'].shift().rolling(window=window_size).max()

    # Calculate rolling median
    df['casual_rolling_median'] = df['casual'].shift().rolling(window=window_size).median()
    df['registered_rolling_median'] = df['registered'].shift().rolling(window=window_size).median()

    # Calculate rolling sum
    df['casual_rolling_sum'] = df['casual'].shift().rolling(window=window_size).sum()
    df['registered_rolling_sum'] = df['registered'].shift().rolling(window=window_size).sum()

    # Calculate shifted rolling statistics for Q1 and Q3 for 'casual' variable
    df['casual_q1'] = df['casual'].shift().rolling(window=window_size).quantile(0.25)
    df['casual_q3'] = df['casual'].shift().rolling(window=window_size).quantile(0.75)

    # Calculate shifted rolling statistics for Q1 and Q3 for 'registered' variable
    df['registered_q1'] = df['registered'].shift().rolling(window=window_size).quantile(0.25)
    df['registered_q3'] = df['registered'].shift().rolling(window=window_size).quantile(0.75)

    # Drop rows with NaN values resulting from shifting
    #df = df.dropna()

    # Encode hour as cyclical feature
    df['hour_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hr'] / 24)

    # Encode month as cyclical feature
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['dteday_ordinal'] = df['dteday'].map(lambda x: x.toordinal())
    # Encode date as cyclical feature
    days_in_month = 30  # Assuming a month with 30 days
    df['date_sin'] = np.sin(2 * np.pi * df['dteday_ordinal'] / days_in_month)
    df['date_cos'] = np.cos(2 * np.pi * df['dteday_ordinal'] / days_in_month)

    df.drop(columns='dteday', inplace=True)
    columns_to_drop = ['casual', 'registered']
    y = df[columns_to_drop]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    time_columns = ['days_since_initial', 'month', 'hr', 'holiday', 'workingday', 'season', 'date_sin', 'date_cos',
                    'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'casual_rolling_mean', 'registered_rolling_mean',
                    'casual_lag1', 'casual_lag2', 'casual_lag3', 'registered_lag1', 'registered_lag2', 'registered_lag3',
                    'casual_rolling_std', 'registered_rolling_std', 'casual_rolling_min', 'casual_rolling_max',
                    'registered_rolling_min', 'registered_rolling_max', 'casual_rolling_std', 'registered_rolling_std',
                    'casual_rolling_sum', 'registered_rolling_sum', 'casual_q1', 'casual_q3', 'registered_q1',
                    'registered_q3']
    time_df = df[time_columns]
    time_encode = ['month', 'hr', 'season']
    time_encoded = pd.get_dummies(time_df, columns=time_encode)
    temp_columns = ['weathersit', 'temp_c', 'feels_like_c', 'hum', 'windspeed', 'season', 'days_since_initial', 'month',
                    'hr', 'holiday', 'workingday', 'season', 'date_sin', 'date_cos', 'month_sin', 'month_cos', 'hour_sin',
                    'hour_cos']
    temp_df = df[temp_columns]
    temp_encode = ['weathersit']
    temp_encoded = pd.get_dummies(temp_df, columns=temp_encode)
    all_encode = ['weathersit', 'season', 'month', 'hr']
    all_encoded = pd.get_dummies(df, columns=all_encode)

    # Split training and testing data
    X_train_all, X_test_all, y_train, y_test = train_test_split(all_encoded, y, test_size=test_size, shuffle=False)
    X_train_time, X_test_time, _, _, = train_test_split(time_encoded, y, test_size=test_size, shuffle=False)
    X_train_temp, X_test_temp, _, _, = train_test_split(temp_encoded, y, test_size=test_size, shuffle=False)

    # Normalize data
    X_train_all, X_test_all, X_train_time, X_test_time, X_train_temp, X_test_temp = data_scaler((X_train_all, X_test_all),(X_train_time, X_test_time),(X_train_temp, X_test_temp))

    # Add the first rows of each tuple item onto the last
    X_train_all, X_test_all, X_train_time, X_test_time, X_train_temp, X_test_temp = dataset_expander((X_train_all, X_test_all),(X_train_time, X_test_time),(X_train_temp, X_test_temp))

    X_train_time = time_data_sliding_window_generator(X_train_time)
    X_train_temp = time_data_sliding_window_generator(X_train_temp)
    X_test_time = time_data_sliding_window_generator(X_test_time)
    X_test_temp = time_data_sliding_window_generator(X_test_temp)

    # Remove first rows of all datasets
    X_train_all, X_test_all, X_train_time, X_test_time, X_train_temp, X_test_temp, y_train = dataset_trimmer(X_train_all, X_test_all, X_train_time, X_test_time, X_train_temp, X_test_temp, y_train)

    return X_train_all, X_test_all, X_train_time, X_test_time, X_train_temp, X_test_temp, y_train, y_test

X_train_all, X_test_all, X_train_time, X_test_time, X_train_temp, X_test_temp, y_train, y_test = preprocess_data(df)

# Begin structure for branch that fits the whole dataset:
input_layer_all = Input(shape=(80,), name='input_all')
first_dense_all = Dense(units='512', activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer_all)
dropout_1_all = Dropout(0.1)(first_dense_all)
second_dense_all = Dense(units='256', activation='relu', kernel_regularizer=regularizers.l2(0.01))(dropout_1_all)
third_dense_all = Dense(units='128', activation='relu', kernel_regularizer=regularizers.l2(0.01))(second_dense_all)
fourth_dense_all = Dense(units='64', activation='relu', kernel_regularizer=regularizers.l2(0.01))(third_dense_all)
fifth_dense_all = Dense(units='32', activation='relu', kernel_regularizer=regularizers.l2(0.01))(fourth_dense_all)
sixth_dense_all = Dense(units='16', activation='relu', kernel_regularizer=regularizers.l2(0.01))(fifth_dense_all)
seventh_dense_all = Dense(units='8', activation='relu', kernel_regularizer=regularizers.l2(0.01))(sixth_dense_all)
y_all_out = Dense(units='2', name='output_all')(seventh_dense_all)
y_all_out_reshaped = RepeatVector(int(timesteps))(y_all_out)

# Begin structure for RNN
input_layer_time = Input(shape=(timesteps, 71), name='input_time')
first_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(input_layer_time)
second_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(first_recurrent_time)
third_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(second_recurrent_time)
fourth_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(third_recurrent_time)
fifth_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(fourth_recurrent_time)
y_time_out = TimeDistributed(Dense(units=2, name='output_time'))(fifth_recurrent_time)

# Begin structure for RNN
input_layer_temp = Input(shape=(timesteps, 21), name='input_temp')
first_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(input_layer_temp)
second_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(first_recurrent_temp)
third_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(second_recurrent_temp)
fourth_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(third_recurrent_temp)
fifth_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(fourth_recurrent_temp)
y_temp_out = TimeDistributed(Dense(units='2', name='output_time'))(fifth_recurrent_temp)

# Build Predictions
concatenated = Concatenate()([y_all_out_reshaped, y_time_out, y_temp_out])
final_dense_first = TimeDistributed(Dense(units='256', activation='relu', kernel_regularizer=regularizers.l2(0.01)))(concatenated)
final_dense_second = TimeDistributed(Dense(units='128', activation='relu', kernel_regularizer=regularizers.l2(0.01)))(final_dense_first)
final_dense_third = TimeDistributed(Dense(units='64', activation='relu', kernel_regularizer=regularizers.l2(0.01)))(final_dense_second)
final_dense_fourth = TimeDistributed(Dense(units='32', activation='relu', kernel_regularizer=regularizers.l2(0.01)))(final_dense_third)
final_dense_fifth = TimeDistributed(Dense(units='16', activation='relu', kernel_regularizer=regularizers.l2(0.01)))(final_dense_fourth)
final_dense_sixth = TimeDistributed(Dense(units='8', activation='relu', kernel_regularizer=regularizers.l2(0.01)))(final_dense_fifth)
# Remove Time Data
flattened = Flatten()(final_dense_fifth)
y1_y2_combined = Dense(units='2', name='output_time')(flattened)

# Initialize the Neural Network
model = Model(inputs=[input_layer_all, input_layer_time, input_layer_temp], outputs=y1_y2_combined)

optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Compile model
model.compile(loss={'output_time': 'mse'},
              metrics={'output_time':tf.keras.metrics.RootMeanSquaredError()})

print(model.summary())

# Fit the model and store the training history
history = model.fit([X_train_all, X_train_time, X_train_temp], y_train,
                    validation_data=([X_test_all, X_test_time, X_test_temp], y_test), epochs=1, verbose=0)

# Evaluate the model on the training data
loss, rmse = model.evaluate([X_train_all, X_train_time, X_train_temp], y_train, verbose=1)
print('-----training data-----')
print(f'loss: {loss}')
print(f'rmse: {rmse}')

# Evaluate the model on the testing data, kernel_regularizer=regularizers.l2(0.01)
loss, rmse = model.evaluate([X_test_all, X_test_time, X_test_temp], y_test, verbose=1)
print('-----testing data-----')
print(f'loss: {loss}')
print(f'rmse: {rmse}')

# Get predictions for the testing data
predictions = model.predict([X_test_all, X_test_time, X_test_temp])
y_test_casual = y_test['casual']
y_test_registered = y_test['registered']
predictions_casual = [y1 for y1, _ in predictions]
predictions_registered = [y2 for _, y2 in predictions]

# Get the r^2
r2_casual = r2_score(y_test_casual, predictions_casual)
r2_registered = r2_score(y_test_registered, predictions_registered)
print(f"R^2 Casual: {r2_casual}, R^2 Registered: {r2_registered}")
print(f'Epochs run: {len(history.epoch)}')
print("Done!")

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()