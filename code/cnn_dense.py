import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras import regularizers
import scipy.io


# fixing random seed for reproducibility

seed = 6
np.random.seed(seed)

# Normalizing the data

# loading and Normalizing the data
X_train= scipy.io.loadmat('xtrain.mat')
X_train = X_train['xtrain']
X_test = scipy.io.loadmat('xtest.mat')
X_test = X_test['xtest']

X_train = (X_train - min(X_train.min(),X_test.min()))/(max(X_train.max(),X_test.max())-min(X_train.min(),X_test.min()))

X_test = (X_test - min(X_train.min(),X_test.min()))/(max(X_train.max(),X_test.max())-min(X_train.min(),X_test.min()))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = np.reshape(X_train,(4620,35,45,1))
X_test = np.reshape(X_test,(2100,35,45,1))

Y_train = scipy.io.loadmat('ytrain.mat')
Y_test = scipy.io.loadmat('ytest.mat')

Y_train = Y_train['ytrain']
Y_test = Y_test['ytest']

Y_train = Y_train-1
Y_test = Y_test-1

Y_train = np_utils.to_categorical(Y_train, 6)
Y_test = np_utils.to_categorical(Y_test, 6)

#creating the model

model = Sequential()

model.add(Conv2D(16, (2, 2), input_shape=(35, 45,1),activation= 'sigmoid', padding='same', kernel_constraint=maxnorm(3)))
model.add(Conv2D(16, (2, 2), activation= 'sigmoid', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.2))

model.add(Conv2D(16, (2, 2),activation= 'sigmoid', padding='same', kernel_constraint=maxnorm(3)))
model.add(Conv2D(16, (2, 2), activation= 'sigmoid', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3),activation= 'sigmoid', padding='same', kernel_constraint=maxnorm(3)))
model.add(Conv2D(32, (3, 3),activation= 'sigmoid', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(6, activation='softmax'))

# Compile model

lrate = 0.01
decay = 0.00025
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1)

preds = model.predict_classes(X_test,batch_size=1,verbose=0)
np.savetxt('results156.csv',preds,delimiter=',')

# Final evaluation of the model

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))