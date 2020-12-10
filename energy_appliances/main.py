import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

df = pd.read_csv('energydata_complete.csv')
df['Appliances'] = df['Appliances'].astype(float)
dates = df['date'].values
appliances = df['Appliances'].values

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mae') < 50.0 and logs.get('val_mae') < 50.0):
          print('\nmae dan val_mae telah kurang dari 10% skala data.')
          self.model.stop_training = True
callbacks = myCallback()

dates_train, dates_test, appliances_train, appliances_test = train_test_split(dates, appliances, test_size=0.2, shuffle=False)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

train_set = windowed_dataset(appliances_train, window_size=60, batch_size=100, shuffle_buffer=1000)
test_set = windowed_dataset(appliances_test, window_size=60, batch_size=100, shuffle_buffer=1000)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=SGD(learning_rate=1.0000e-04, momentum=0.9),
              metrics=['mae'])

history = model.fit(train_set, validation_data=(test_set), epochs=100, callbacks=[callbacks])