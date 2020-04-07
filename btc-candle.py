import pandas
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy
from matplotlib import pyplot as plt


# def layer_maker(n_layers, n_nodes, activation, drop=None, d_rate=.5):
#     """
#     Create a specified number of hidden layers for an RNN
#     Optional: Adds regularization option, dropout layer to prevent potential overfitting if necessary
#     """
#
#     # Creating the specified number of hidden layers with the specified number of nodes
#     for x in range(1, n_layers + 1):
#         model.add(LSTM(n_nodes, activation=activation, return_sequences=True))
#
#         # Adds a Dropout layer after every Nth hidden layer (the 'drop' variable)
#         try:
#             if x % drop == 0:
#                 model.add(Dropout(d_rate))
#         except:
#             pass


def download_csv():
    source_url = "https://api.hitbtc.com/api/2/public/candles/BTCUSD?period=M1&limit=1000&sort=desc"
    return pandas.read_json(source_url, orient='columns')


# Download data
data = download_csv()


# ** Trying to predict open price

prediction_frame = 30
df_train = data.open[:len(data.close) - prediction_frame]
df_test = data.open[len(data.close) - prediction_frame:]

min_max_scaler = MinMaxScaler()
training_set = min_max_scaler.fit_transform(df_train.values.reshape(-1, 1)).reshape(-1)
X_train = training_set[0:len(training_set) - 1]
X_train = numpy.reshape(X_train, (len(X_train), 1, 1))
y_train = training_set[1:len(training_set)]


num_units = 4
activation_function = 'sigmoid'
opmitizer = 'adam'
loss_function = 'mean_squared_error'
batch_size = 5
num_epochs = 100

regressor = Sequential()
regressor.add(LSTM(units=num_units, activation=activation_function, input_shape=(None, 1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer=opmitizer, loss=loss_function)
regressor.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs)


test_set = df_test.values
inputs = numpy.reshape(test_set, (len(test_set), 1))
inputs = min_max_scaler.transform(inputs)
inputs = numpy.reshape(inputs, (len(test_set), 1, 1))

predicted_price = regressor.predict(inputs)
predicted_price = min_max_scaler.inverse_transform(predicted_price)

plt.figure(figsize=(25, 25), dpi=80, facecolor='w', edgecolor='k')
plt.plot(test_set, color='red', label='Real BTC price')
plt.plot(predicted_price[:, 0], color='blue', label='Predicted BTC price')
plt.title('BTC price prediction')
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize=40)
plt.legend(loc='best')
plt.show()




