











from keras import applications
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation, AveragePooling1D
from keras.callbacks import CSVLogger
import tensorflow as tf
from scipy.ndimage import imread
import numpy as np
import random
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU
from keras import backend as K
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import optimizers
import h5py
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with h5py.File(''.join(['bitcoin2015to2019_5m_256_16.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value
    input_times = hf['input_times'].value
    output_times = hf['output_times'].value
    original_inputs = hf['original_inputs'].value
    original_outputs = hf['original_outputs'].value
    original_datas = hf['original_datas'].value

scaler = MinMaxScaler()
# split training validation
training_size = int(0.8 * datas.shape[0])
training_datas = datas[:training_size, :, :]
training_labels = labels[:training_size, :, :]
validation_datas = datas[training_size:, :, :]
validation_labels = labels[training_size:, :, :]
validation_original_outputs = original_outputs[training_size:, :, :]
validation_original_inputs = original_inputs[training_size:, :, :]
validation_input_times = input_times[training_size:, :, :]
validation_output_times = output_times[training_size:, :, :]

ground_true = np.append(validation_original_inputs, validation_original_outputs, axis=1)
ground_true_times = np.append(validation_input_times, validation_output_times, axis=1)
step_size = datas.shape[1]
#batch_size = 8
nb_features = datas.shape[2]


units= 50
second_units = 30
batch_size = 8
output_size=16




#build model
model = Sequential()
model.add(LSTM(units=units,activation='tanh', input_shape=(step_size,nb_features),return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_size))
model.add(LeakyReLU())

model.load_weights('weights/bitcoin2015to2019_5m_close_256_16_close_LSTM_relu-989-0.33790.hdf5')
model.compile(loss='mse', optimizer='adam')





# 스케일링된 가격을 원래대로 돌림
predicted = model.predict(validation_datas)
predicted_inverted = []


scaler.fit(original_datas[:,0].reshape(-1,1))
predicted_inverted.append(scaler.inverse_transform(predicted))


#get only the close data
ground_true = ground_true[:,:,0].reshape(-1)
ground_true_times = ground_true_times.reshape(-1)
ground_true_times = pd.to_datetime(ground_true_times, unit='s')
# since we are appending in the first dimension
predicted_inverted = np.array(predicted_inverted)[0,:,:].reshape(-1)
print (np.array(predicted_inverted).shape)
validation_output_times = pd.to_datetime(validation_output_times.reshape(-1), unit='s')



ground_true_df = pd.DataFrame()
ground_true_df['times'] = ground_true_times
ground_true_df['value'] = ground_true

prediction_df = pd.DataFrame()
prediction_df['times'] = validation_output_times
prediction_df['value'] = predicted_inverted

print('--정답--')
print(ground_true_df.tail())
print('--예측값--')
print(prediction_df.tail())



#print(ground_true_df.loc[:300])
#print(prediction_df.loc[:300])

#ground_true_df = ground_true_df.drop_duplicates(['times'])

#print(ground_true_df.loc[:300])
#prediction_df = prediction_df.loc[(prediction_df["times"].dt.year == 2017 )&(prediction_df["times"].dt.month > 7 ),: ]
#ground_true_df = ground_true_df.loc[(ground_true_df["times"].dt.year == 2017 )&(ground_true_df["times"].dt.month > 7 ),:]

from sklearn.metrics import mean_squared_error
from math import sqrt
print('--')
print(validation_original_outputs[:,:,0].reshape(-1).shape)
print (np.array(predicted_inverted).shape)
mse = mean_squared_error(validation_original_outputs[:,:,0].reshape(-1),predicted_inverted)
rmse = sqrt(mse)
print(rmse)

plt.figure(figsize=(20,10))
plt.plot(ground_true_df.times,ground_true_df.value, label = 'Actual')
plt.plot(prediction_df.times,prediction_df.value,'ro', label='Predicted')
plt.legend(loc='upper left')

plt_name = 'bitcoin2015to2019_5m_close_256_16_close_LSTM_relu-989-0.33790 : ' + str(rmse)
plt.title(plt_name)
plt.savefig(plt_name + '.png')
#plt.show()






#real-time drawing
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Create figure for plotting
#plt.figure(3)
fig = plt.figure(4, figsize=(20,10))
# xs = []
# ys = []


# This function is called periodically from FuncAnimation
#def animate(i, pt, pv, gt, gv):
    # Draw x and y lists
ax1 = fig.add_subplot(1,1,1)

# Format plot
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(bottom=0.30)
plt.title('predicted vs true (LSTM)')
plt.ylabel('USDT')

#plt.show()
iter = 1
for gt, gv in zip(ground_true_df.times, ground_true_df.value):

    idx = prediction_df.index[prediction_df['times'] == gt].tolist()

    #ax1.clear()
    ax1.plot(gt, gv, 'bo', label='predicted')
    if idx:
        print(idx)
        print(idx[0])
        print(gt)
        print(prediction_df['times'][idx[0]])
        #ani = animation.FuncAnimation(fig, animate, fargs=(pt, pv, prediction_df['times'][idx[0]], prediction_df['values'][idx[0]]), interval=10)
        ax1.plot(prediction_df['times'][idx[0]], prediction_df['value'][idx[0]], 'r+', label='Actual')
        plt.pause(0.001)

    if iter >=2000:
        ax1.clear()
        iter = 1
    iter += 1
        #time.sleep(0.1)
#plt.show()