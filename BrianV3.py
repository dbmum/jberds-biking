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
from datetime import datetime

timesteps=30
batch_size=720

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')

def time_data_sliding_window_generator(X_train_array, y_train_array):
# Create the dataset using timeseries_dataset_from_array
    dataset = timeseries_dataset_from_array(
        X_train_array,
        y_train_array,
        sequence_length=timesteps,
        batch_size=batch_size
    )
    return dataset

def ensure_columns(df, column_names):
    missing_columns = [column for column in column_names if column not in df.columns]
    if missing_columns:
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_columns)], axis=1)
    return df
12343
def create_chronological_split(df_to_copy, columns_to_use, columns_to_encode, class_variables, all_column_names):
    df = df_to_copy.copy()
    df['date'] = pd.to_datetime(df['dteday'], format='%d/%m/%y')
    split_date = datetime(2012, 11, 1)
    test_set = df[df['date'] >= split_date]
    train_set = df[df['date'] < split_date]
    test_set_subset = test_set[columns_to_use]
    train_set_subset = train_set[columns_to_use]
    y_train = test_set_subset[class_variables]
    y_test = train_set_subset[class_variables]
    columns_to_drop = class_variables + ['dteday']
    test_set_subset.drop(columns=columns_to_drop, axis=1, inplace=True)
    train_set_subset.drop(columns=columns_to_drop, axis=1, inplace=True)
    test_set_subset = pd.get_dummies(test_set_subset, columns=columns_to_encode)
    train_set_subset = pd.get_dummies(train_set_subset, columns=columns_to_encode)
    test_set_subset = ensure_columns(test_set_subset, all_column_names)
    train_set_subset = ensure_columns(train_set_subset, all_column_names)
    minmax_scaler = MinMaxScaler()
    train_set_subset_final = minmax_scaler.fit_transform(train_set_subset) # fit the scale to the training data
    test_set_subset_final = minmax_scaler.transform(test_set_subset) # use the same scale on the testing data
    return train_set_subset_final, test_set_subset_final, y_train, y_test

def preprocess_data(df):
    """
    :param df: Take our dataframe object
    :return: Return 6 (3 branches train/test) different datasets that have been one-hot-encoded and normalized containing the below arrays of features and y_train, y_test series
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
    df = df.dropna()

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

    columns_to_drop = ['casual', 'registered']
    all_column_names = list(pd.get_dummies(df, columns=df.columns.values).columns.values)

    time_columns = ['days_since_initial', 'month', 'hr', 'holiday', 'workingday', 'season', 'date_sin', 'date_cos',
                    'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'casual_rolling_mean', 'registered_rolling_mean',
                    'casual_lag1', 'casual_lag2', 'casual_lag3', 'registered_lag1', 'registered_lag2', 'registered_lag3',
                    'casual_rolling_std', 'registered_rolling_std', 'casual_rolling_min', 'casual_rolling_max',
                    'registered_rolling_min', 'registered_rolling_max', 'casual_rolling_std', 'registered_rolling_std',
                    'casual_rolling_sum', 'registered_rolling_sum', 'casual_q1', 'casual_q3', 'registered_q1',
                    'registered_q3', 'casual', 'registered', 'dteday']
    time_encode = ['month', 'hr', 'season']
    X_train_time, X_test_time, y_train, y_test = create_chronological_split(df, time_columns, time_encode, columns_to_drop, all_column_names)
    temp_columns = ['weathersit', 'temp_c', 'feels_like_c', 'hum', 'windspeed', 'season', 'days_since_initial', 'month',
                    'hr', 'holiday', 'workingday', 'season', 'date_sin', 'date_cos', 'month_sin', 'month_cos', 'hour_sin',
                    'hour_cos', 'casual', 'registered', 'dteday']
    temp_encode = ['weathersit']
    X_train_temp, X_test_temp, _, _ = create_chronological_split(df, temp_columns, temp_encode, columns_to_drop, all_column_names)
    all_columns = list(df.columns.values)
    all_encode = ['weathersit', 'season', 'month', 'hr']
    X_train_all, X_test_all, _, _ = create_chronological_split(df, all_columns, all_encode, columns_to_drop, all_column_names)

    # Create time oriented data loaders
    X_train_temp =  time_data_sliding_window_generator(X_train_temp, y_train)
    X_train_time =  time_data_sliding_window_generator(X_train_time, y_train)
    X_test_temp =  time_data_sliding_window_generator(X_test_temp, y_train)
    X_test_time =  time_data_sliding_window_generator(X_test_time, y_train)

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
input_layer_temp = Input(shape=(timesteps, 15), name='input_temp')
first_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(input_layer_temp)
second_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(first_recurrent_temp)
third_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(second_recurrent_temp)
fourth_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(third_recurrent_temp)
fifth_recurrent_temp = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(fourth_recurrent_temp)
y_temp_out = TimeDistributed(Dense(units='2', name='output_time'))(fifth_recurrent_temp)

# Build Predictions
concatenated = Concatenate()([y_all_out_reshaped, y_time_out, y_temp_out])
final_dense_first = Dense(units='32', activation='relu', kernel_regularizer=regularizers.l2(0.01))(concatenated)
final_dense_second = Dense(units='16', activation='relu', kernel_regularizer=regularizers.l2(0.01))(final_dense_first)
final_dense_third = Dense(units='8', activation='relu', kernel_regularizer=regularizers.l2(0.01))(final_dense_second)
y1_y2_combined = Dense(units='2', name='output_time')(final_dense_third)

# Initialize the Neural Network
model = Model(inputs=[input_layer_all, input_layer_time, input_layer_temp], outputs=y1_y2_combined)

optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Compile model
model.compile(loss={'casual_output': 'mse'},
              metrics={'casual_output':tf.keras.metrics.RootMeanSquaredError()})

# Fit the model and store the training history
history = model.fit([X_train_all, X_train_time, X_train_temp], y_train,
                    validation_data=([X_test_all, X_test_time, X_test_temp], y_test), epochs=200, verbose=0)

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
