import pandas
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy
from matplotlib import pyplot as plt


# Configure memory usage
# config = tensorflow.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tensorflow.Session(config=config)



def download_csv():
    source_url = "https://api.hitbtc.com/api/2/public/candles/BTCUSD?period=M1&limit=1000&sort=desc"
    return pandas.read_json(source_url, orient='columns')


# Download data
data = download_csv()


# ** Trying to predict open price

prediction_frame = 100
df_train = data.open[:len(data.close) - prediction_frame]
df_test = data.open[len(data.close) - prediction_frame:]

min_max_scaler = MinMaxScaler()
training_set = min_max_scaler.fit_transform(df_train.values.reshape(-1, 1)).reshape(-1)
X_train = training_set[0:len(training_set) - 1]
X_train = numpy.reshape(X_train, (len(X_train), 1, 1))
y_train = training_set[1:len(training_set)]


regressor = Sequential()
regressor.add(LSTM(4, activation='sigmoid', input_shape=(None, 1), return_sequences=True))
regressor.add(LSTM(2, activation='sigmoid'))
regressor.add(Dense(1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, batch_size=5, epochs=100, verbose=2)


test_set = df_test.values
inputs = numpy.reshape(test_set, (len(test_set), 1))
inputs = min_max_scaler.transform(inputs)
inputs = numpy.reshape(inputs, (len(test_set), 1, 1))

predicted_price = regressor.predict(inputs)
predicted_price = min_max_scaler.inverse_transform(predicted_price)


# Render chart
plt.figure(figsize=(19, 10), facecolor='w', edgecolor='k')
plt.plot(test_set, color='red', label='Real BTC price')
plt.plot(predicted_price[:, 0], color='blue', label='Predicted BTC price')
plt.title('BTC price prediction')
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize=40)
plt.legend(loc='best')
plt.show()


# Render error
plt.figure(figsize=(19, 10), facecolor='w', edgecolor='k')
plt.plot(predicted_price[:, 0] - df_test.values, color='red', label='Error')
plt.title('Error')
plt.xlabel('Time', fontsize=40)
plt.ylabel('Error size in USD', fontsize=40)
plt.legend(loc='best')
plt.show()


# Print results
print('Error: {:-9}'.format(mean_squared_error(predicted_price[:, 0], df_test.values)))




