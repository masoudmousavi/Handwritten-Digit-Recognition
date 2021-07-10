import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import scipy.io as sio


data_dict = sio.loadmat('ex4data1.mat')
labeled_Y = data_dict['y']
y = tf.one_hot(labeled_Y % 10, depth=10)

x = data_dict['X']
mean = np.mean(x)
sigma = np.std(x)
x = (x - mean) / sigma
shuffler = np.random.permutation(x.shape[0])
x = x[shuffler, :]
y = y.numpy()[shuffler, :]
y = y.reshape(y.shape[0], 10)
labeled_Y = labeled_Y.T[:, shuffler]
x_train = x[:4000, :]
x_test = x[4000:, :]
y_train = y[:4000, :]
y_test = y[4000:, :]
train_classes = labeled_Y[:, :4000]
test_classes = labeled_Y[:, 4000:]

model = Sequential()
model.add(Dense(128, input_shape=(400,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=15)

evaluation = model.evaluate(x_test, y_test)
print(evaluation)