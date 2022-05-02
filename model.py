from numpy import concatenate
from operator import not_
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, models, activations, initializers, regularizers
import tensorflow.keras.backend as K

from data.voxceleb import *
from config import *
from utils.model_utils import *


def get_model(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
    # CHOOSE MODEL -----------------------------------------------
    model = speaker_rec_model_att_like(hparams, num_classes, stateful, dtype, inference)
    # ------------------------------------------------------------

    model.summary()
    return model


def speaker_rec_model(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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


def speaker_rec_model_multidimLSTM(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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
    output = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False, dtype=dtype)(input)
    # output = layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP, pad_end=True), name="WINDOW")(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False, dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)

    # Time-based lstm
    t_lstm = layers.LSTM(num_lstm_units, return_sequences=True, name="T_LSTM_0", stateful=stateful)(output)
    t_lstm = TimeReduction(reduction_factor=2, batch_size=batch_size, name="TIME_REDUCTION")(t_lstm)
    t_lstm = layers.LSTM(num_lstm_units, return_sequences=True, name="T_LSTM_1", stateful=stateful)(t_lstm)

    # Freq-based lstm
    f_lstm = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]), name="TRANSPOSE")(output)
    f_lstm = layers.LSTM(num_lstm_units, return_sequences=True, name="F_LSTM_0", stateful=stateful)(f_lstm)
    f_lstm = TimeReduction(reduction_factor=2, batch_size=batch_size, name="FREQ_REDUCTION")(f_lstm)
    f_lstm = layers.LSTM(num_lstm_units, return_sequences=True, name="F_LSTM_1", stateful=stateful)(f_lstm)
    

    output = layers.Concatenate(axis=-1, name="CONCAT_T_AND_F")([t_lstm[:,-1, :], f_lstm[:, -1, :]])

    # output = GlobalWeightedAveragePooling1D(name="VOTE_AVG")(output)

    output = layers.Dense(num_classes, activation="relu", name="DENSE_0")(output)

    # output = layers.Lambda(lambda x: tf.nn.sigmoid(x), name="SIGMOID_OUT")(output)
    output = layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)

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


def speaker_rec_model_att_like(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
    # Preprocessing parameters
    SAMPLE_RATE = hparams[HP_SAMPLE_RATE.name]
    FRAME_LENGTH = int(hparams[HP_FRAME_LENGTH.name] * SAMPLE_RATE)
    FRAME_STEP = int(hparams[HP_FRAME_STEP.name] * SAMPLE_RATE)
    MAX_AUDIO_LENGTH = hparams[HP_MAX_NUM_FRAMES.name] * FRAME_STEP + FRAME_LENGTH - FRAME_STEP
    # NUM_MEL_BINS = hparams[HP_NUM_MEL_BINS.name]

    # Model parameters
    num_lstm_units = hparams[HP_NUM_LSTM_UNITS.name]
    num_self_att_units = hparams[HP_NUM_SELF_ATT_UNITS.name]
    num_self_att_hops = hparams[HP_NUM_SELF_ATT_HOPS.name]
    num_dense_units = hparams[HP_NUM_DENSE_UNITS.name]

    batch_size = None
    if stateful:
        batch_size = 1

    input_shape = [None] # if inference else [MAX_AUDIO_LENGTH]

    # Define model
    # Preprocessing layers (for on device) -------------------------------------------------
    input = keras.Input(
        shape=input_shape, 
        batch_size=batch_size, 
        dtype=dtype, 
        name='PREPROCESS_INPUT')
    encode_in = layers.Lambda(
        lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams, normalize=True), 
        trainable=False, 
        dtype=dtype, 
        name="LOG_MEL_SPEC")(input)
    encode_in = layers.Lambda(
        lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), 
        trainable=False, 
        dtype=dtype, 
        name="DOWNSAMPLE")(encode_in)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    # Note: the inference flag is in the NORMALIZE layer if Layer Normalization used in training since it attaches parameters to fixed time length
    # and in inference, we want variable time
    # encode_in = layers.LayerNormalization(axis=-2, name="NORMALIZE", trainable=False, center=not_(inference), scale=not_(inference))(encode_in)

    encode = layers.LSTM(
        num_lstm_units, 
        return_sequences=True, 
        stateful=stateful,
        kernel_regularizer=regularizers.L2(l2=0.0001),
        recurrent_regularizer=regularizers.L2(l2=0.0001),
        name="LSTM_0")(encode_in)
    # encode_in = layers.Dense(num_lstm_units, activation="tanh", kernel_regularizer=regularizers.L2(l2=0.0001))(encode_in)
    # encode = layers.Add()([encode, encode_in])
    encode = TimeReduction(reduction_factor=2, batch_size=batch_size, name="TIME_REDUCTION_E")(encode)
    encode = layers.LSTM(
        num_lstm_units, 
        return_sequences=True, 
        stateful=stateful, 
        kernel_regularizer=regularizers.L2(l2=0.0001),
        recurrent_regularizer=regularizers.L2(l2=0.0001),
        name="LSTM_1")(encode)

    # SA Layer
    sa_output = SelfAttentionMechanismFn(
        num_self_att_hops, 
        num_self_att_units, 
        encode, 
        name="SA_0")

    output_proj = layers.Dense(num_dense_units, kernel_regularizer=regularizers.L2(l2=0.0001), activation="sigmoid", name="DENSE_0")(sa_output)

    # TEST LAYER FOR LINEAR SEPARABILITY
    # output_proj = layers.Dense(2, activation="relu", name="TEST_PROJECTION")(output_proj)

    # Output logits layers
    output = layers.Dense(num_classes, activation=None, kernel_regularizer=regularizers.L2(l2=0.0001), name="DENSE_OUT")(output_proj)
    
    # Output activation layer
    # output = layers.Activation(activation="sigmoid", name="SIGMOID_OUT")(output)
    # output = layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output, output_proj], name="speaker_rec_model_att_like")

    
    return model


def speaker_rec_model_att_like_simpleRNN(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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
    encode_in = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False, dtype=dtype)(input)
    # output = layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP, pad_end=True), name="WINDOW")(input)
    # output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    encode_in = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False, dtype=dtype)(encode_in)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    encode_in = layers.LayerNormalization(axis=-2, name="NORMALIZE")(encode_in)
    # output = layers.Dropout(0.2)(output)
    # output = layers.Conv1D(128, 3, strides=2 ,activation="relu", use_bias=True)(output)
    # output = layers.AveragePooling1D(pool_size=2, strides=2)(output)
    # focus = SelfAttentionMechanismFn(1, num_dense_units, encode_in)
    # focus = tf.repeat(focus, tf.shape(encode_in)[1], 1)
    # focus = layers.Reshape([133, 320])(focus)
    # encode_in = layers.Add()([encode_in, focus])
    # encode_in = layers.LayerNormalization(axis=-2)(encode_in)
    # output = layers.Concatenate(axis=-1, name="ADD_FOCUS")([output, focus])
    # encode = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0", stateful=stateful)(encode_in)
    encode = layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_0", stateful=stateful)(encode_in)
    encode = layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_1", stateful=stateful)(encode)
    encode = layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_2", stateful=stateful)(encode)
    encode = layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_3", stateful=stateful)(encode)
    encode = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION_E")(encode)
    # output = layers.Dropout(0.2)(output)
    # encode, final_memory_state, final_carry_state = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_1", stateful=stateful, return_state=True)(encode)
    encode = layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_4", stateful=stateful)(encode)
    encode = layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_5", stateful=stateful)(encode)
    encode = layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_6", stateful=stateful)(encode)
    encode, final_memory_state = layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_7", stateful=stateful, return_state=True)(encode)

    # Attention layer (try across ts)
    # Let query be last ts output from encode
    # encode = tf.transpose(encode, perm=[0, 2, 1])
    
    # query = encode[:,-1,:]
    # query = tf.expand_dims(query, axis=1)
    # query = tf.repeat(query, tf.shape(encode)[1], 1)
    # query = encode
    # inp = encode[:,1,:]
    # inp = layers.Reshape((1, num_lstm_units))(inp)
    # value = layers.Concatenate(axis=1)([inp, encode[:,1:,:]])
    # output = layers.AdditiveAttention()([query, value])
    # output = layers.Concatenate(axis=-1, name="CONCAT")([encode, encode_2, encode_3, encode_4])

    # decode = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_2", stateful=stateful)(encode)
    # # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION_P")(output)
    # decode = layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_3", stateful=stateful)(output)

    # query = decode[:,-1,:]
    # query = tf.expand_dims(query, axis=1)
    # query = tf.repeat(query, tf.shape(encode)[1], 1)
    # query = encode
    # value = encode
    # output = layers.Attention(causal=True)([query, value])

    # SA Layer
    output = SelfAttentionMechanismFn(1, num_dense_units, encode)

    # output = GlobalWeightedMaxPooling1D(name="MAX_POOL")(output)

    # output = decode[:,-1,:]
    output = layers.Reshape([num_lstm_units])(output)

    output = layers.Dense(num_classes, activation=None, name="DENSE_1")(output)
    
    output = layers.Activation(activation="sigmoid", name="SIGMOID_OUT")(output)
    # output = layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="speaker_rec_model_att_like_simpleRNN")
    return model


def speaker_rec_model_rnnt_like(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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
    output = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False, dtype=dtype)(input)
    # output = layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP, pad_end=True), name="WINDOW")(input)
    output = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False, dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    # output = layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
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
    output = layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)

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


def fc_model(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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


def vgg_m(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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

    output = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(output) # Maybe need to conv over features, not ts

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


def speaker_rec_model_w_pooling(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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


def speaker_rec_model_convlstm(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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
def vggvox_model(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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

    output = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(output) # Maybe need to conv over features, not ts

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

def vgg_nang_et_al(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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
    log_mel_spec = layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # input = layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP), name="WINDOW")(input)
    norm = layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(log_mel_spec)
    # input = layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(input)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    # input = layers.LayerNormalization(axis=-2, name="NORMALIZE")(input)

    # Expand for Conv layer compatibility
    expanded = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name="EXPAND")(norm)

    # Begin VGG from Nang et al
    # L1_1
    l1_1 = layers.Conv2D(64, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l1_1")(expanded)
    l1_1 = layers.BatchNormalization(name="bn_l1_1")(l1_1)
    l1_1 = layers.Activation(activation="relu", name="relu_l1_1")(l1_1)
    
    # L1_2
    l1_2 = layers.Conv2D(64, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l1_2")(l1_1)
    l1_2 = layers.BatchNormalization(name="bn_l1_2")(l1_2)
    l1_2 = layers.Activation(activation="relu", name="relu_l1_2")(l1_2)

    # L2
    l2 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name="mpool_l2")(l1_2)

    # L3
    # L3_1
    l3_1 = layers.Conv2D(128, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l3_1")(l2)
    l3_1 = layers.BatchNormalization(name="bn_l3_1")(l3_1)
    l3_1 = layers.Activation(activation="relu", name="relu_l3_1")(l3_1)
    
    # L3_2
    l3_2 = layers.Conv2D(128, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l3_2")(l3_1)
    l3_2 = layers.BatchNormalization(name="bn_l3_2")(l3_2)
    l3_2 = layers.Activation(activation="relu", name="relu_l3_2")(l3_2)

    # L4
    l4 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name="mpool_l4")(l3_2)

    # L5
    # L5_1
    l5_1 = layers.Conv2D(256, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l5_1")(l4)
    l5_1 = layers.BatchNormalization(name="bn_l5_1")(l5_1)
    l5_1 = layers.Activation(activation="relu", name="relu_l5_1")(l5_1)
    
    # L5_2
    l5_2 = layers.Conv2D(256, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l5_2")(l5_1)
    l5_2 = layers.BatchNormalization(name="bn_l5_2")(l5_2)
    l5_2 = layers.Activation(activation="relu", name="relu_l5_2")(l5_2)

    # L6
    l6 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name="mpool_l6")(l5_2)

    # L7
    # L7_1
    l7_1 = layers.Conv2D(512, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l7_1")(l6)
    l7_1 = layers.BatchNormalization(name="bn_l7_1")(l7_1)
    l7_1 = layers.Activation(activation="relu", name="relu_l7_1")(l7_1)
    
    # L7_2
    l7_2 = layers.Conv2D(512, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l7_2")(l7_1)
    l7_2 = layers.BatchNormalization(name="bn_l7_2")(l7_2)
    l7_2 = layers.Activation(activation="relu", name="relu_l7_2")(l7_2)

    # L8
    l8 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name="mpool_l8")(l7_2)

    # L9 - Self-Attention Mechanism
    # matmul1
    shp = K.int_shape(l8)
    T = shp[1]
    n_h = shp[2]*shp[3]
    n_c = shp[3]
    n_k = 4
    l9_in = layers.Reshape((T, n_h), input_shape=shp[1:], name="concat_hiddens_l9")(l8) # combine hidden vectors
    l9_matmul1 = layers.Dense(n_c, activation="tanh", use_bias=False, name="matmul1_l9")(l9_in)

    # matmul2
    l9_matmul2 = layers.Dense(n_k, activation="softmax", use_bias=False, name="matmul2_l9")(l9_matmul1)

    # transpose
    l9_trans = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]), name="transpose_l9")(l9_matmul2)
    
    # matmul3
    l9_matmul3 = layers.Dot(axes=[2,1], name="matmul3_l9")([l9_trans, l9_in])

    # L10
    l10 = layers.GlobalAveragePooling1D(name="avgpool_l10")(l9_matmul3)

    # L11
    l11 = layers.Dense(256, activation="relu", use_bias=True, name="dense_l11")(l10)

    # L12
    l12 = layers.Dense(num_classes, activation=None, use_bias=True, name="dense_l12")(l11)

    output = layers.Softmax(name="SOFTMAX_OUT")(l12)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = keras.Model(inputs=[input],
                        outputs=[output], name="vgg_nang_et_al")
    return model