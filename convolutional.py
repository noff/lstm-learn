from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(256,256,1)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.summary()