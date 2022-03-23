import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, models
import tensorflow.keras.backend as K

from data.voxceleb import *
from config import *
from utils.model_utils import *


def get_model(hparams, num_classes, stateful=False, dtype=tf.float32):
    # CHOOSE MODEL -----------------------------------------------
    model = speaker_rec_model_rnnt_like(hparams, num_classes, stateful, dtype)
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
    output = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=1, name="NORMALIZE")(output)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Conv1D(128, 3, strides=2 ,activation="relu", use_bias=True)(output)
    # output = layers.AveragePooling1D(pool_size=2, strides=2)(output)
    output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0", stateful=stateful)(output)
    output = TimeReduction(reduction_factor=2, batch_size=batch_size, name="TIME_REDUCTION")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.LSTM(num_lstm_units, return_sequences=False, name="LSTM_1", stateful=stateful)(output)

    # output = layers.Dropout(0.2)(output)
    # output = layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="speaker_rec")
    return model


def speaker_rec_model_rnnt_like(hparams, num_classes, stateful=False, dtype=tf.float32):
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
    output = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP, pad_end=True), name="WINDOW")(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Conv1D(128, 3, strides=2 ,activation="relu", use_bias=True)(output)
    # output = layers.AveragePooling1D(pool_size=2, strides=2)(output)
    output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0", stateful=stateful)(output)
    output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION_E")(output)
    # output = layers.Dropout(0.2)(output)
    encode = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_1", stateful=stateful)(output)

    output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_2", stateful=stateful)(encode)
    # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION_P")(output)
    output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_3", stateful=stateful)(output)

    output = layers.Concatenate(axis=-1, name="CONCAT_E_AND_P")([output, encode])

    output = layers.Dense(num_classes, activation="relu", name="DENSE_0")(output)

    output = GlobalWeightedMaxPooling1D(name="VOTE")(output)

    # output = layers.Lambda(lambda x: tf.nn.sigmoid(x), name="SIGMOID_OUT")(output)
    output = layers.Softmax(name="SOFTMAX_OUT")(output)

    # output = layers.Dropout(0.2)(output)
    # output = layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = layers.Dropout(0.2)(output)
    
    # output = layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="speaker_rec_model_rnnt_like")
    return model

def fc_model(hparams, num_classes, stateful=False, dtype=tf.float32):
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
    output = layers.Lambda(
        lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams, ret_mfcc=True), 
        name="LOG_MEL_SPEC", 
        trainable=False)(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(
        lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), 
        name="DOWNSAMPLE", 
        trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    output = layers.Flatten()(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_2")(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_3")(output)
    output = layers.Dense(num_dense_units, activation="relu", name="DENSE_4")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="fc_model")
    return model

def vgg_m(hparams, num_classes, stateful=False, dtype=tf.float32):
    # Preprocessing parameters
    SAMPLE_RATE = hparams[HP_SAMPLE_RATE.name]
    FRAME_LENGTH = int(hparams[HP_FRAME_LENGTH.name] * SAMPLE_RATE)
    FRAME_STEP = int(hparams[HP_FRAME_STEP.name] * SAMPLE_RATE)
    MAX_AUDIO_LENGTH = hparams[HP_MAX_NUM_FRAMES.name] * FRAME_STEP + FRAME_LENGTH - FRAME_STEP

    # Model hyperparameters
    num_lstm_units = hparams[HP_NUM_LSTM_UNITS.name]
    lstm_output_shape = [None, num_lstm_units]
    num_dense_units = hparams[HP_NUM_DENSE_UNITS.name]

    batch_size = None
    if stateful:
        batch_size = 1

    input_shape = [MAX_AUDIO_LENGTH]

    # Define model
    # Preprocessing layers (for on device) -------------------------------------------------
    input = tf.keras.Input(shape=input_shape, batch_size=batch_size,
                           dtype=dtype, name='INPUT')
    output = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP), name="WINDOW")(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    # output = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)

    # Expand for Conv layer compatibility
    output = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name="EXPAND")(output)

    output = layers.Conv2D(96, 7, strides=(2,2), activation="relu", use_bias=True, padding="same", name="conv1")(output)
    output = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name="mpool1")(output)

    output = layers.Conv2D(256, 5, strides=(2,2), activation="relu", use_bias=True, padding="same", name="conv2")(output)
    output = layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name="mpool2")(output)
    
    output = layers.Conv2D(384, 3, strides=(2,2), activation="relu", use_bias=True, padding="same", name="conv3")(output)
    output = layers.Conv2D(256, 3, strides=(1,1), activation="relu", use_bias=True, padding="same", name="conv4")(output)
    output = layers.Conv2D(256, 3, strides=(1,1), activation="relu", use_bias=True, padding="same", name="conv5")(output)
    output = layers.MaxPool2D(pool_size=(2,2), strides=(1,1), name="mpool5")(output)

    shp = K.int_shape(output)
    shp = [shp[1], shp[-1]*shp[-2]]
    output = layers.Reshape(shp, name="SQUEEZE")(output)

    output = layers.Dense(4096, activation="relu", name="fc6")(output)

    output = layers.GlobalAveragePooling1D(name="apool6")(output)

    output = layers.Dense(1024, activation="relu", name="fc7")(output)
    output = layers.Dense(num_classes, activation="softmax", name="fc8")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="vgg_m")
    return model

def speaker_rec_model_w_pooling(hparams, num_classes, stateful=False, dtype=tf.float32):
    # Preprocessing parameters
    SAMPLE_RATE = hparams[HP_SAMPLE_RATE.name]
    FRAME_LENGTH = int(hparams[HP_FRAME_LENGTH.name] * SAMPLE_RATE)
    FRAME_STEP = int(hparams[HP_FRAME_STEP.name] * SAMPLE_RATE)
    MAX_AUDIO_LENGTH = hparams[HP_MAX_NUM_FRAMES.name] * FRAME_STEP + FRAME_LENGTH - FRAME_STEP

    # Model hyperparameters
    num_lstm_units = hparams[HP_NUM_LSTM_UNITS.name]
    lstm_output_shape = [None, num_lstm_units]
    num_dense_units = hparams[HP_NUM_DENSE_UNITS.name]

    batch_size = None
    if stateful:
        batch_size = 1

    input_shape = [MAX_AUDIO_LENGTH]

    # Define model
    # Preprocessing layers (for on device) -------------------------------------------------
    input = tf.keras.Input(shape=input_shape, batch_size=batch_size,
                           dtype=dtype, name='INPUT')
    output = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0")(output)
    # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION")(output)
    # output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_1")(output)
    # output = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_2")(output)
    # output = layers.Dropout(0.2)(output)
    output = layers.LSTM(num_classes, return_sequences=True, name="LSTM_3")(output)
    # output = layers.TimeDistributed(layers.Softmax(), name="TIME_DIST_SOFTMAX_0")(output)
    # output = layers.Lambda(lambda x: tf.math.reduce_prod(x, axis=1), name="MULT_ACROSS")(output)
    output = GlobalWeightedMaxPooling1D()(output)
    output = layers.Softmax()(output)
    # output = layers.Flatten()(output)
    # output = layers.Dense(num_classes, activation="softmax")(output)
    # output = layers.GlobalAveragePooling1D(name="GLOBAL_AVG_POOL")(output)
    # output = layers.Flatten()(output)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
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
    output = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
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


# VGGVox verification model
def vggvox_model(hparams, num_classes, stateful=False, dtype=tf.float32):
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
    input = keras.Input(shape=input_shape, batch_size=batch_size,
                           dtype=dtype, name='INPUT')
    output = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    # output = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    output = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name="EXPAND")(output)

    # Trainable layers ----------------------------------------------------------------------
    output = conv_bn_pool(output,layer_idx=1,conv_filters=96,conv_kernel_size=(7,7),conv_strides=(2,2),conv_pad=(1,1), pool='max',pool_size=(3,3),pool_strides=(2,2))
    output = conv_bn_pool(output,layer_idx=2,conv_filters=256,conv_kernel_size=(5,5),conv_strides=(2,2),conv_pad=(1,1), pool='max',pool_size=(3,3),pool_strides=(2,2))
    output = conv_bn_pool(output,layer_idx=3,conv_filters=384,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
    output = conv_bn_pool(output,layer_idx=4,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
    output = conv_bn_pool(output,layer_idx=5,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1), pool='max',pool_size=(5,3),pool_strides=(3,2))		
    output = conv_bn_dynamic_apool(output,layer_idx=6,conv_filters=4096,conv_kernel_size=(5,1),conv_strides=(1,1),conv_pad=(0,0), conv_layer_prefix='fc')
    output = conv_bn_pool(output,layer_idx=7,conv_filters=1024,conv_kernel_size=(1,1),conv_strides=(1,1),conv_pad=(0,0), conv_layer_prefix='fc')
    output = layers.Lambda(lambda y: K.l2_normalize(y, axis=3), name='norm')(output)
    output = layers.Dense(num_classes, activation="softmax", name='fc8')(output)
    m = keras.Model(input, output, name='VGGVox')
    return m