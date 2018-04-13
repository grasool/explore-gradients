# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:52:10 2018

@author: Ghulam Rasool
"""

import keras
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.callbacks import TensorBoard
import pandas
from keras.models import Model

# Please specify number of layers
N_LAYERS = 25

# Width of hidden layer, number of neurons in the hidden layer. all have same size. 
n_hwidth = 128
batch_size = 128
n_classes = 10
epochs = 2

n_layers = N_LAYERS -1

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

inputs = Input(shape=(784,))


def fCreate_Layers(n_layers, inputs):
    x = inputs
    for k in range(n_layers):
        x = Dense(n_hwidth)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        print('Layer %d added' % (k+1))
        
    return x

# Create all layers    
x_all = fCreate_Layers(n_layers, inputs)
# Output layer
predictions = Dense(n_classes, activation='softmax')(x_all)
print('Output layer added')
# Create Model
model = Model(inputs=inputs, outputs=predictions)
# Print model summary
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

ttb_dir = './MLP_No_SKIP_%s' % N_LAYERS
callbacks = [TensorBoard(log_dir=ttb_dir, histogram_freq=1, batch_size=32, write_graph=False, write_grads=True, write_images=False)]

#history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=callbacks)


#model_name = 'MLP_No_SKIP_%s.h5' % N_LAYERS
#csv_name = 'MLP_No_SKIP_%s.csv' % N_LAYERS

#model.save(model_name)
#pandas.DataFrame(history.history).to_csv(csv_name)


