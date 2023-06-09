import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import *
from sklearn.metrics import r2_score
from matplotlib import pyplot
from keras import regularizers
import numpy as np

# Important values for our model, these values define that our LSTM layers will receive 24 items of data at a time which represent 1 day.
timesteps=24
window_size=24
temp_dataset_holder = []

df = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv')
holdout = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes_december.csv')
train_size = .9
train = df.iloc[:int(df.shape[0] * train_size)]  # First 90% of the DataFrame
test = df.iloc[-(int(df.shape[0]) - int(df.shape[0] * train_size)):]  # Last 10% of the DataFrame
full_column_list = ['date_cos', 'date_sin', 'days_since_initial', 'feels_like_c', 'holiday', 'hour_cos', 'hour_sin', 'hr', 'hr_0', 'hr_1', 'hr_10', 'hr_11', 'hr_12', 'hr_13', 'hr_14', 'hr_15', 'hr_16', 'hr_17', 'hr_18', 'hr_19', 'hr_2', 'hr_20', 'hr_21', 'hr_22', 'hr_23', 'hr_3', 'hr_4', 'hr_5', 'hr_6', 'hr_7', 'hr_8', 'hr_9', 'hum', 'month', 'month_1', 'month_10', 'month_11', 'month_12', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_cos', 'month_sin', 'season', 'season_1', 'season_2', 'season_3', 'season_4', 'temp_c', 'weathersit_1', 'weathersit_2', 'weathersit_3', 'weathersit_4', 'windspeed', 'workingday']

def time_data_sliding_window_generator(X_train):
    """
    :param X_train: Takes a dataset to transform
    :return: Returns a nparray which contains the dimensions [batch_size, timesteps, window_size, data]
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
        window = X_train.iloc[start_index:end_index, :]
        windowed_data[end_index-1] = window

    return windowed_data


def dataset_expander(*args):
    """
    :param args: A tuple of numpy arrays to transform
    :return: A tuple of numpy arrays with the last (window_size - 1) * 2 rows added onto array_2.
             This enables us to later drop these rows without having NaN values at the beginning of our test and holdout set
    """
    modified_dataframes = []
    for item in args:
        # Scale the data
        df1, df2 = item
        combined_df = pd.concat([df1.iloc[-((window_size - 1) * 2):], df2], axis=0)
        modified_dataframes.extend([df1, combined_df])
    return tuple(modified_dataframes)

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

scaler_all = MinMaxScaler()
scaler_time = MinMaxScaler()
scaler_temp = MinMaxScaler()

def data_scaler(df1, df2, df3, mode='train'):
    """
    :param args: Tuple of Dataframes to apply a MinMaxScale operation to
    :return: The list of dataframes that have been scaled.
    """
    args = [(df1, scaler_all), (df2, scaler_time), (df3, scaler_temp)]
    modified_dataframes = []
    for item in args:
        # Scale the data
        df, scaler = item
        if (mode == 'train'):
            df = scaler.fit_transform(df)
        else:
            df = scaler.transform(df) # use the same scale on the testing data
        modified_dataframes.append(df)
    return tuple(modified_dataframes)

def test_model(model, df_train, df_test):
    df_test = df_test.copy()
    lower_bound = 0
    upper_bound = (window_size - 1) * 2 + 1
    _, df_test = dataset_expander((df_train, df_test))
    df_test = pd.DataFrame(df_test)
    df_test.reset_index(drop=True, inplace=True)

    # Zero out the answers contained in df_test so that we know we're not peeking information:
    df_test.loc[upper_bound:, 'casual'] = 0
    df_test.loc[upper_bound:, 'registered'] = 0

    for i in range(df_test.shape[0] - (window_size) * 2):
        df_subset = df_test.iloc[lower_bound:upper_bound].copy()
        X_test_all, X_test_time, X_test_temp, _ = preprocess(df_subset)
        X_test_time = time_data_sliding_window_generator(X_test_time)
        X_test_temp = time_data_sliding_window_generator(X_test_temp)
        X_test_all, X_test_time, X_test_temp = dataset_trimmer(X_test_all, X_test_time, X_test_temp)
        predictions = model.predict([X_test_all, X_test_time, X_test_temp])
        df_test.loc[upper_bound - 1, 'casual'] = predictions[0][0]
        df_test.loc[upper_bound - 1, 'registered'] = predictions[0][1]
        lower_bound, upper_bound = lower_bound + 1, upper_bound + 1

    df_test = dataset_trimmer(df_test)
    df_test = df_test[0]
    y_out = df_test[['casual', 'registered']]
    return y_out

def preprocess(df):
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['month'] = df['dteday'].dt.month
    initial_date = pd.to_datetime('2011-01-01')
    df['days_since_initial'] = (df['dteday'] - initial_date).dt.days

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

    df.drop(columns=['dteday', 'dteday_ordinal'], inplace=True)
    columns_to_drop = ['casual', 'registered']

    y = pd.DataFrame()
    try:
        y = df[columns_to_drop]
        df.drop(columns=columns_to_drop, axis=1, inplace=True)
    except KeyError:
        pass

    all_encode = ['weathersit', 'season', 'month', 'hr']
    all_encoded = pd.get_dummies(df, columns=all_encode)

    time_columns = ['days_since_initial', 'month', 'hr', 'holiday', 'workingday', 'season', 'date_sin', 'date_cos',
                    'month_sin', 'month_cos', 'hour_sin', 'hour_cos']
    time_df = df[time_columns]
    time_encode = ['month', 'hr', 'season']
    time_encoded = pd.get_dummies(time_df, columns=time_encode)

    temp_columns = ['weathersit', 'temp_c', 'feels_like_c', 'hum', 'windspeed', 'season', 'days_since_initial', 'month',
                    'hr', 'holiday', 'workingday', 'date_sin', 'date_cos', 'month_sin', 'month_cos', 'hour_sin',
                    'hour_cos']
    temp_df = df[temp_columns]
    temp_encode = ['weathersit']
    temp_encoded = pd.get_dummies(temp_df, columns=temp_encode)

    # Get the missing columns in all_encoded
    missing_columns_all = list(set(full_column_list) - set(all_encoded.columns.values))
    zero_df_all = pd.DataFrame(0, index=time_encoded.index, columns=missing_columns_all, dtype=int)
    all_encoded = pd.concat([all_encoded, zero_df_all], axis=1)

    # Get the missing columns in time_encoded
    missing_columns_time = list(set(full_column_list) - set(time_encoded.columns.values))
    zero_df_time = pd.DataFrame(0, index=time_encoded.index, columns=missing_columns_time, dtype=int)
    time_encoded = pd.concat([time_encoded, zero_df_time], axis=1)

    # Get the missing columns in temp_encoded
    missing_columns_temp = list(set(full_column_list) - set(temp_encoded.columns.values))
    zero_df_temp = pd.DataFrame(0, index=temp_encoded.index, columns=missing_columns_temp, dtype=int)
    temp_encoded = pd.concat([temp_encoded, zero_df_temp], axis=1)

    return all_encoded, time_encoded, temp_encoded, y

def preprocess_dataframe(df):
    """
    :param df: The dataframe for preprocessing
    :param test_size: The amount of the dataframe to use for testing
    :return: The various dataframes needed by our model's branches
    """
    # Preprocessing to create 3 branches
    all_encoded, time_encoded, temp_encoded, y = preprocess(df)

    # Normalize data
    X_all, X_time, X_temp = data_scaler(all_encoded, time_encoded, temp_encoded)

    X_time = time_data_sliding_window_generator(pd.DataFrame(X_time))
    X_temp = time_data_sliding_window_generator(pd.DataFrame(X_temp))

    # Remove first rows of all datasets
    X_all, X_time, X_temp, y = dataset_trimmer(X_all, X_time, X_temp, y)

    return X_all, X_time, X_temp, y

X_train_all, X_train_time, X_train_temp, y_train, = preprocess_dataframe(df)
X_test_all, X_test_time, X_test_temp, y_test = preprocess_dataframe(test)

# Begin structure for branch that fits the whole dataset:
input_layer_all = Input(shape=(60,), name='input_all')
first_dense_all = Dense(units='144', activation='tanh', kernel_regularizer=regularizers.l2(0.01))(input_layer_all)
dropout_1_all = Dropout(0.25)(first_dense_all)
second_dense_all = Dense(units='80', activation='tanh', kernel_regularizer=regularizers.l2(0.01))(dropout_1_all)
dropout_2_all = Dropout(0.25)(second_dense_all)
third_dense_all = Dense(units='432', activation='tanh', kernel_regularizer=regularizers.l2(0.01))(dropout_2_all)
dropout_3_all = Dropout(0.25)(third_dense_all)
fourth_dense_all = Dense(units='272', activation='relu', kernel_regularizer=regularizers.l2(0.01))(dropout_3_all)
y_all_out = Dense(units='2', name='output_all')(fourth_dense_all)
y_all_out_reshaped = RepeatVector(int(timesteps))(y_all_out)

# Begin structure for RNN
input_layer_time = Input(shape=(timesteps, 60), name='input_time')
first_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(input_layer_time)
second_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(first_recurrent_time)
third_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(second_recurrent_time)
fourth_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(third_recurrent_time)
fifth_recurrent_time = LSTM(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(fourth_recurrent_time)
y_time_out = TimeDistributed(Dense(units=2, name='output_time'))(fifth_recurrent_time)

# Begin structure for RNN
input_layer_temp = Input(shape=(timesteps, 60), name='input_temp')
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
                    validation_data=([X_test_all, X_test_time, X_test_temp], y_test), shuffle=False, epochs=50, verbose=True)

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
y_test_total = y_test['casual'] + y_test['registered']
predictions_casual = [y1 for y1, _ in predictions]
predictions_registered = [y2 for _, y2 in predictions]
predictions_total = [y1 + y2 for y1, y2 in zip(predictions_casual, predictions_registered)]

df_predictions = pd.DataFrame({"total": predictions_total})
df_predictions.to_csv("predictions_total.csv", index=False)

# Get the r^2
r2_casual = r2_score(y_test_casual, predictions_casual)
r2_registered = r2_score(y_test_registered, predictions_registered)
r2_total = r2_score(y_test_total, predictions_total)
print(f"R^2 Casual: {r2_casual}, R^2 Registered: {r2_registered}")
print(f"R^2 Total: {r2_total}")
print(f'Epochs run: {len(history.epoch)}')
print("Done!")

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()