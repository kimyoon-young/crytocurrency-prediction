import pandas as pd
import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU, AveragePooling1D
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# Make the program use only one GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


with h5py.File(''.join(['bitcoin2015to2019_5m_256_16.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value


output_file_name='bitcoin2015to2019_5m_close_CNN_3_relu_256_16'

step_size = datas.shape[1]
batch_size= 8
nb_features = datas.shape[2]

epochs = 1000

#split training validation
training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:]
#build model

# 2 layers
model = Sequential()


model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=1, filters=16, kernel_size=64))
#model.add(LeakyReLU())
#model.add(Dropout(0.5))
#
#model.add(AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))

model.add(Conv1D(activation='relu', strides=2, filters=16, kernel_size=64))
#model.add(AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))

#model.add(LeakyReLU())
#model.add(Dropout(0.5))
model.add(Conv1D( strides=2, filters=nb_features, kernel_size=34))


# model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=1, filters=16, kernel_size=30))
# model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
#
# #model.add(PReLU())
# #model.add(Dropout(0.5))
#
# model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=2, filters=8, kernel_size=20))
# #model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
#
#
#
# #model.add(Dropout(0.5))
#
# model.add(Conv1D( strides=2, filters=nb_features, kernel_size=16))
#

# model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=16, kernel_size=20))
# model.add(Dropout(0.5))
# model.add(Conv1D( strides=4, filters=nb_features, kernel_size=16))


# 3 layers

# model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=8))
# #model.add(LeakyReLU())
# model.add(Dropout(0.5))
# model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=8))
# #model.add(LeakyReLU())
# model.add(Dropout(0.5))
# model.add(Conv1D( strides=2, filters=nb_features, kernel_size=8))

#
# model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=32))
# #model.add(LeakyReLU())
# model.add(Dropout(0.5))
# model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=32))
# #model.add(LeakyReLU())
# model.add(Dropout(0.5))
# model.add(Conv1D( strides=2, filters=nb_features, kernel_size=16))


# model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=1, filters=8, kernel_size=64))
# #model.add(LeakyReLU())
# model.add(Dropout(0.5))
# model.add(Conv1D(activation='relu', strides=1, filters=8, kernel_size=64))
# #model.add(AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last'))
# #model.add(LeakyReLU())
#
# model.add(Dropout(0.5))
# model.add(Conv1D( strides=1, filters=nb_features, kernel_size=33))
'''
# 3 Layers
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=8))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=8))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D( strides=2, filters=nb_features, kernel_size=8))
# 4 layers
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=2, filters=8, kernel_size=2))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=2))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=2))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D( strides=2, filters=nb_features, kernel_size=2))
'''

from keras.optimizers import Adam

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(optimizer=opt, loss="mean_absolute_percentage_error")
#model.compile(loss='mse', optimizer='adam')
model.fit(training_datas, training_labels,verbose=1, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=[CSVLogger(output_file_name+'.csv', append=True),ModelCheckpoint('weights/'+output_file_name+'-{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1,mode='min', save_best_only=True)])
