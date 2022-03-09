import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, models

from data.voxceleb import *
from config import *


def get_model(hparams, num_classes, stateful=False, dtype=tf.float32):
    # CHOOSE MODEL -----------------------------------------------
    model = speaker_rec_model(hparams, num_classes, stateful, dtype)
    # ------------------------------------------------------------

    model.summary()
    return model

class TimeReduction(layers.Layer):

    def __init__(self,
                 reduction_factor,
                 batch_size=None,
                 **kwargs):

        super(TimeReduction, self).__init__(**kwargs)

        self.reduction_factor = reduction_factor
        self.batch_size = batch_size

    def call(self, inputs):

        input_shape = tf.shape(inputs)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = input_shape[0]

        max_time = input_shape[-2]
        num_units = input_shape[-1]
        extra_timestep = tf.math.floormod(max_time, self.reduction_factor)
        reduced_size = tf.math.floordiv(max_time, self.reduction_factor) + extra_timestep

        outputs = inputs

        paddings = [[0, 0], [0, extra_timestep], [0, 0]]
        outputs = tf.pad(outputs, paddings)

        return tf.reshape(outputs, (batch_size, reduced_size, num_units * self.reduction_factor))

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_factor": self.reduction_factor,
            "batch_size": self.batch_size,
        })
        return config



def speaker_rec_model(hparams, num_classes, stateful=False, dtype=tf.float32):
    # Preprocessing parameters
    SAMPLE_RATE = hparams[HP_SAMPLE_RATE.name]
    FRAME_LENGTH = int(hparams[HP_FRAME_LENGTH.name] * SAMPLE_RATE)
    FRAME_STEP = int(hparams[HP_FRAME_STEP.name] * SAMPLE_RATE)
    MAX_AUDIO_LENGTH = hparams[HP_MAX_NUM_FRAMES.name] * FRAME_STEP + FRAME_LENGTH - FRAME_STEP
    # NUM_MEL_BINS = hparams[HP_NUM_MEL_BINS.name]

    # Model hyperparameters
    num_lstm_units = hparams[HP_NUM_LSTM_UNITS.name]
    lstm_output_shape = [None, num_lstm_units]
    num_dense_units = hparams[HP_NUM_DENSE_UNITS.name]

    batch_size = None
    if stateful:
        batch_size = 1

    # input_shape = [None, hparams[HP_NUM_MEL_BINS.name] * hparams[HP_STACK_SIZE.name]]
    input_shape = [MAX_AUDIO_LENGTH]

    # Define model
    # Preprocessing layers (for on device) -------------------------------------------------
    input = tf.keras.Input(shape=input_shape, batch_size=batch_size,
                           dtype=dtype, name='INPUT')
    output = layers.Lambda(convert_to_log_mel_spec_layer, name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(group_and_downsample_spec_v2_layer, name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Conv1D(128, 3, strides=2 ,activation="relu", use_bias=True)(output)
    # output = layers.AveragePooling1D(pool_size=2, strides=2)(output)
    output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0")(output)
    output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.LSTM(num_lstm_units, return_sequences=False, name="LSTM_1")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="speaker_rec")
    return model

def speaker_rec_model_w_pooling(hparams, num_classes, stateful=False, dtype=tf.float32):
    # Preprocessing parameters
    SAMPLE_RATE = hparams[HP_SAMPLE_RATE.name]
    FRAME_LENGTH = int(hparams[HP_FRAME_LENGTH.name] * SAMPLE_RATE)
    FRAME_STEP = int(hparams[HP_FRAME_STEP.name] * SAMPLE_RATE)
    MAX_AUDIO_LENGTH = hparams[HP_MAX_NUM_FRAMES.name] * FRAME_STEP + FRAME_LENGTH - FRAME_STEP
    # NUM_MEL_BINS = hparams[HP_NUM_MEL_BINS.name]

    # Model hyperparameters
    num_lstm_units = hparams[HP_NUM_LSTM_UNITS.name]
    lstm_output_shape = [None, num_lstm_units]
    num_dense_units = hparams[HP_NUM_DENSE_UNITS.name]

    batch_size = None
    if stateful:
        batch_size = 1

    # input_shape = [None, hparams[HP_NUM_MEL_BINS.name] * hparams[HP_STACK_SIZE.name]]
    input_shape = [MAX_AUDIO_LENGTH]

    # Define model
    # Preprocessing layers (for on device) -------------------------------------------------
    input = tf.keras.Input(shape=input_shape, batch_size=batch_size,
                           dtype=dtype, name='INPUT')
    output = layers.Lambda(convert_to_log_mel_spec_layer, name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(group_and_downsample_spec_v2_layer, name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0")(output)
    # output = layers.Dropout(0.2)(output)
    # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION")(output)
    output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_1")(output)
    output = layers.TimeDistributed(layers.Dense(32, activation="relu"), name="TIME_DIST_DENSE_0")(output)
    # output = layers.GlobalAveragePooling1D(name="GLOBAL_AVG_POOL")(output)
    output = layers.Flatten()(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="speaker_rec_w_pooling")
    return model


def speaker_rec_model_convlstm(hparams, num_classes, stateful=False, dtype=tf.float32):
    # DOESNT WORK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Preprocessing parameters
    SAMPLE_RATE = hparams[HP_SAMPLE_RATE.name]
    FRAME_LENGTH = int(hparams[HP_FRAME_LENGTH.name] * SAMPLE_RATE)
    FRAME_STEP = int(hparams[HP_FRAME_STEP.name] * SAMPLE_RATE)
    MAX_AUDIO_LENGTH = hparams[HP_MAX_NUM_FRAMES.name] * FRAME_STEP + FRAME_LENGTH - FRAME_STEP
    # NUM_MEL_BINS = hparams[HP_NUM_MEL_BINS.name]

    # Model hyperparameters
    num_lstm_units = hparams[HP_NUM_LSTM_UNITS.name]
    lstm_output_shape = [None, num_lstm_units]
    num_dense_units = hparams[HP_NUM_DENSE_UNITS.name]

    batch_size = None
    if stateful:
        batch_size = 1

    # input_shape = [None, hparams[HP_NUM_MEL_BINS.name] * hparams[HP_STACK_SIZE.name]]
    input_shape = [MAX_AUDIO_LENGTH]

    # Define model
    # Preprocessing layers (for on device) -------------------------------------------------
    input = tf.keras.Input(shape=input_shape, batch_size=batch_size,
                           dtype=dtype, name='INPUT')
    output = layers.Lambda(convert_to_log_mel_spec_layer, name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(group_and_downsample_spec_v2_layer, name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    # output = layers.Reshape((tf.shape(output)[0], tf.shape(output)[1], tf.shape(output)[2], 1))(output)
    output = layers.ConvLSTM1D(64, 3, padding="same", return_sequences=True, name="LSTM_0", data_format="channels_last")(output)
    # output = layers.Dropout(0.2)(output)
    # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION")(output)
    output = layers.LSTM(num_lstm_units, return_sequences=False, name="LSTM_1")(output)
    # output = layers.TimeDistributed(layers.Dense(num_lstm_units/2, activation="relu"), name="TIME_DIST_DENSE_0")(output)
    # output = layers.GlobalAveragePooling1D(name="GLOBAL_AVG_POOL")(output)
    # output = layers.Flatten()(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="speaker_rec_convlstm")
    return model