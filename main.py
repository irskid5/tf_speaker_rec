import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt

import data
from preprocessing import get_dataset

from config import *

# GPU stuff
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

tf.random.set_seed(HP_SEED.domain.values[0])
np.random.seed(HP_SEED.domain.values[0])

# Load dataset !! CHOOSE DATASET HERE !!
ds_train, ds_val, ds_test = get_dataset("voxceleb", VOXCELEB_DIR, data.voxceleb)

def get_model():
    model = models.Sequential(
        [
            # keras.layers.LSTM(10, input_shape=(None, 320)),
            # keras.layers.BatchNormalization(),
            layers.LSTM(1024, input_shape=(None, 320)), #, return_sequences=True),
            # layers.LSTM(1024, input_shape=(None, 1024), return_sequences=True),
            # layers.LSTM(1024, input_shape=(None, 1024), return_sequences=True),
            # layers.LSTM(1024, input_shape=(None, 1024), return_sequences=True),
            # layers.LSTM(1024, input_shape=(None, 1024)),
            layers.Dense(2048, activation="relu"),
            layers.Dense(2048, activation="relu"),
            layers.Dense(2048, activation="relu"),
            layers.Dense(2048, activation="relu"),
            layers.Dense(2048, activation="relu"),
            layers.Dense(1252, activation="softmax"),
        ]
    )
    model.summary()
    return model

model = get_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir="tb_callback_dir", histogram_freq=1,
)

model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_val,
    callbacks=[tensorboard_callback],
    verbose=1,
)

print("End")

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#
# mnist = keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# sample, sample_label = x_train[0], y_train[0]
#
#
# batch_size = 64
# # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# # Each input sequence will be of size (28, 28) (height is treated like time).
# input_dim = 28
#
# units = 2048
# output_size = 10  # labels are from 0 to 9
#
# # Build the RNN model
#
#
# def build_model(allow_cudnn_kernel=True):
#     # CuDNN is only available at the layer level, and not at the cell level.
#     # This means `LSTM(units)` will use the CuDNN kernel,
#     # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
#     if allow_cudnn_kernel:
#         # The LSTM layer with default options uses CuDNN.
#         lstm_layer = keras.layers.LSTM(units, input_shape=(
#             None, input_dim), return_sequences=True)
#     else:
#         # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
#         lstm_layer = keras.layers.RNN(
#             keras.layers.LSTMCell(units), input_shape=(None, input_dim)
#         )
#     model = keras.models.Sequential(
#         [
#             lstm_layer,
#             keras.layers.LSTM(units, input_shape=(
#                 None, units), return_sequences=True),
#             keras.layers.LSTM(units, input_shape=(
#                 None, units), return_sequences=True),
#             keras.layers.LSTM(units, input_shape=(
#                 None, units)),
#             keras.layers.BatchNormalization(),
#             keras.layers.Dense(output_size),
#         ]
#     )
#     model.summary()
#     return model
#
#
# model = build_model(allow_cudnn_kernel=True)
#
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer="sgd",
#     metrics=["accuracy"],
# )
#
#
# model.fit(
#     x_train, y_train, validation_data=(
#         x_test, y_test), batch_size=batch_size, epochs=10
# )
