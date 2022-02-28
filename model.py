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
    output = layers.Lambda(convert_to_log_mel_spec_layer, name="LOG_MEL_SPEC")(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(group_and_downsample_spec_v2_layer, name="DOWNSAMPLE")(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    output = layers.LSTM(num_lstm_units, input_shape=input_shape, return_sequences=True, name="LSTM_0")(output)
    output = layers.LSTM(num_lstm_units, input_shape=lstm_output_shape, return_sequences=False, name="LSTM_1")(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    output = layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="speaker_rec")
    return model
