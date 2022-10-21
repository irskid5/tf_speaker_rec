from numpy import concatenate, std
from operator import not_
import tensorflow as tf
import keras.backend as K

from data.voxceleb import *
from config import *

# MODEL UTILITY FUNCTIONS
from utils.model_utils import *

# QUANTIZATION FUNCTIONS
from quantization import *


def get_model(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
    # CHOOSE MODEL -----------------------------------------------
    model = quantized_speaker_rec_model_IRNN_att_like(hparams, num_classes, stateful, dtype, inference)
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
    output = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = tf.keras.layers.LayerNormalization(axis=1, name="NORMALIZE")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Conv1D(128, 3, strides=2 ,activation="relu", use_bias=True)(output)
    # output = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(output)
    output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0", stateful=stateful)(output)
    output = TimeReduction(reduction_factor=2, batch_size=batch_size, name="TIME_REDUCTION")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=False, name="LSTM_1", stateful=stateful)(output)

    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    output = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False, dtype=dtype)(input)
    # output = tf.keras.layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP, pad_end=True), name="WINDOW")(input)
    # output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False, dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)

    # Time-based lstm
    t_lstm = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="T_LSTM_0", stateful=stateful)(output)
    t_lstm = TimeReduction(reduction_factor=2, batch_size=batch_size, name="TIME_REDUCTION")(t_lstm)
    t_lstm = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="T_LSTM_1", stateful=stateful)(t_lstm)

    # Freq-based lstm
    f_lstm = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]), name="TRANSPOSE")(output)
    f_lstm = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="F_LSTM_0", stateful=stateful)(f_lstm)
    f_lstm = TimeReduction(reduction_factor=2, batch_size=batch_size, name="FREQ_REDUCTION")(f_lstm)
    f_lstm = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="F_LSTM_1", stateful=stateful)(f_lstm)
    

    output = tf.keras.layers.Concatenate(axis=-1, name="CONCAT_T_AND_F")([t_lstm[:,-1, :], f_lstm[:, -1, :]])

    # output = GlobalWeightedAveragePooling1D(name="VOTE_AVG")(output)

    output = tf.keras.layers.Dense(num_classes, activation="relu", name="DENSE_0")(output)

    # output = tf.keras.layers.Lambda(lambda x: tf.nn.sigmoid(x), name="SIGMOID_OUT")(output)
    output = tf.keras.layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)

    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    
    # output = tf.keras.layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    input = tf.keras.Input(
        shape=input_shape, 
        batch_size=batch_size, 
        dtype=dtype, 
        name='PREPROCESS_INPUT')
    encode_in = tf.keras.layers.Lambda(
        lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams, normalize=True), 
        trainable=False, 
        dtype=dtype, 
        name="LOG_MEL_SPEC")(input)
    encode_in = tf.keras.layers.Lambda(
        lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), 
        trainable=False, 
        dtype=dtype, 
        name="DOWNSAMPLE")(encode_in)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    # Note: the inference flag is in the NORMALIZE layer if Layer Normalization used in training since it attaches parameters to fixed time length
    # and in inference, we want variable time
    # encode_in = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE", trainable=False, center=not_(inference), scale=not_(inference))(encode_in)

    encode = tf.keras.layers.LSTM(
        num_lstm_units, 
        # activation="relu",
        # recurrent_activation="relu",
        return_sequences=True, 
        stateful=stateful,
        kernel_regularizer=keras.regularizers.L2(l2=0.0001),
        recurrent_regularizer=keras.regularizers.L2(l2=0.0001),
        # recurrent_initializer=keras.initializers.Identity(),
        # kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.001, seed=1997),
        name="LSTM_0")(encode_in)
    # encode_in = tf.keras.layers.Dense(num_lstm_units, activation="tanh", kernel_regularizer=keras.regularizers.L2(l2=0.0001))(encode_in)
    # encode = tf.keras.layers.Add()([encode, encode_in])
    encode = TimeReduction(reduction_factor=2, batch_size=batch_size, name="TIME_REDUCTION_E")(encode)
    encode = tf.keras.layers.LSTM(
        num_lstm_units, 
        # activation="relu",
        # recurrent_activation="relu",
        return_sequences=True, 
        stateful=stateful, 
        kernel_regularizer=keras.regularizers.L2(l2=0.0001),
        recurrent_regularizer=keras.regularizers.L2(l2=0.0001),
        # recurrent_initializer=keras.initializers.Identity(),
        # kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.001, seed=1997),
        name="LSTM_1")(encode)

    encode = BreakpointLayerForDebug()(encode)

    # SA Layer
    sa_output = SelfAttentionMechanismFn(
        num_self_att_hops, 
        num_self_att_units, 
        encode, 
        name="SA_0")

    # sa_stacked_out = tf.keras.layers.Reshape([num_self_att_hops, num_lstm_units], name="SA_1_INPUT")(sa_output)

    # sa_output = SelfAttentionMechanismFn(
    #     1, 
    #     num_self_att_units, 
    #     sa_stacked_out, 
    #     name="SA_1")

    output_proj = tf.keras.layers.Dense(num_dense_units, kernel_regularizer=keras.regularizers.L2(l2=0.0001), activation="relu", name="DENSE_0")(sa_output)

    # TEST LAYER FOR LINEAR SEPARABILITY
    # output_proj = tf.keras.layers.Dense(2, activation="relu", name="TEST_PROJECTION")(output_proj)

    # Output logits layers
    output = tf.keras.layers.Dense(num_classes, activation=None, kernel_regularizer=keras.regularizers.L2(l2=0.0001), name="DENSE_OUT")(output_proj)
    
    # Output activation layer
    # output = tf.keras.layers.Activation(activation="sigmoid", name="SIGMOID_OUT")(output)
    # output = tf.keras.layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
                        outputs=[output, output_proj], name="speaker_rec_model_att_like")

    
    return model


def quantized_speaker_rec_model_IRNN_att_like(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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
    irnn_stack_size = hparams[HP_IRNN_STACK_SIZE.name]
    irnn_identity_scale = hparams[HP_IRNN_IDENTITY_SCALE.name]
    
    # Activation functions
    activation_irnn = "relu"
    activation_dense = "relu"

    # Activation fns (quantized)
    # activation_irnn = sign_with_htanh_deriv
    # activation_dense = sign_with_htanh_deriv

    # Regularizers (None for quantization I believe)
    # kernel_regularizer = AdaptiveBinaryTernaryRegularizer(lm=1e-1) # TernaryEuclideanRegularizer(l2=0.00001, beta=4) # tf.keras.regularizers.L2(0.0001)
    # recurrent_regularizer = AdaptiveBinaryTernaryRegularizer(lm=1e-1) # BinaryEuclideanRegularizer(l2=1e-6) #TernaryEuclideanRegularizer(l2=0.00001, beta=4) # tf.keras.regularizers.L2(0.0001) # TernaryEuclideanRegularizer(l2=0.00001, beta=4)
    bias_regularizer = None

    # Quantization functions (General)
    kernel_quantizer = stochastic_ternary(alpha="auto_po2") # quantized_bits(bits=2, integer=2, symmetric=1, keep_negative=True)
    recurrent_quantizer = stochastic_ternary(alpha="auto_po2") # quantized_bits(bits=2, integer=2, symmetric=1, keep_negative=True)
    bias_quantizer = None # quantized_bits(bits=2, integer=2, symmetric=1, keep_negative=True)

    # Debug
    tf.print({
        "activation_irnn": activation_irnn,
        "activation_dense": activation_dense,
        "kernel_regularizer": "AdaptiveBinaryTernaryRegularizer(lm=1e-1)",
        "recurrent_regularizer": "AdaptiveBinaryTernaryRegularizer(lm=1e-1)",
        "bias_regularizer": bias_regularizer,
        "kernel_quantizer": kernel_quantizer,
        "recurrent_quantizer": recurrent_quantizer,
        "bias_quantizer": bias_quantizer
    })

    batch_size = None
    if stateful:
        batch_size = 1

    input_shape = [None] # if inference else [MAX_AUDIO_LENGTH]

    # Define model
    # Preprocessing layers (for on device) -------------------------------------------------
    input = tf.keras.Input(
        shape=input_shape, 
        batch_size=batch_size, 
        dtype=dtype, 
        name='PREPROCESS_INPUT')
    encode_in = tf.keras.layers.Lambda(
        lambda x: tf.stop_gradient(convert_to_log_mel_spec_layer(x, hparams=hparams, normalize=True)), 
        trainable=False, 
        dtype=dtype, 
        name="LOG_MEL_SPEC")(input)
    encode_in = tf.keras.layers.Lambda(
        lambda x: tf.stop_gradient(group_and_downsample_spec_v2_layer(x, hparams=hparams)), 
        trainable=False, 
        dtype=dtype, 
        name="DOWNSAMPLE")(encode_in)
    # encode_in = tf.keras.layers.Lambda(
    #     lambda x: tf.stop_gradient(stochastic_ternary(alpha=1)(x)),
    #     trainable=False,
    #     dtype=dtype,
    #     name="QUANTIZE_STOCHASTIC_TERNARY"
    # )(encode_in)
    encode_in = tf.keras.layers.Lambda(
        lambda x: ternarize_tensor_with_threshold(x, theta=2/3*tf.reduce_mean(tf.abs(x))),
        trainable=False,
        dtype=dtype,
        name="TERNARIZE_WITH_THRESHOLD"
    )(encode_in)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    encode = QIRNN( 
        cell=None,
        units=num_lstm_units, 
        identity_scale=irnn_identity_scale,
        stacking_number=irnn_stack_size,
        kernel_regularizer=None, # daptiveBinaryTernaryRegularizer(lm=1e-1, name="IRNN_0/kernel"),
        recurrent_regularizer=None, # AdaptiveBinaryTernaryRegularizer(lm=1e-1, name="IRNN_0/recurrent"),
        bias_regularizer=bias_regularizer,
        kernel_quantizer=kernel_quantizer,
        recurrent_quantizer=recurrent_quantizer,
        bias_quantizer=bias_quantizer,
        activation=activation_irnn,
        stateful=stateful,
        name="IRNN_0")(encode_in)
    # encode = tf.keras.layers.Lambda(lambda x: tf.debugging.check_numerics(x, message="Found NaN or Inf in "+str(x)), name="CHK_"+"IRNN_0")(encode)
    encode = TimeReduction(reduction_factor=2, batch_size=batch_size, name="TIME_REDUCTION_E")(encode)
    encode = QIRNN(
        cell=None,
        units=num_lstm_units, 
        identity_scale=irnn_identity_scale,
        stacking_number=irnn_stack_size,
        kernel_regularizer=None, # AdaptiveBinaryTernaryRegularizer(lm=1e-1, name="IRNN_1/kernel"),
        recurrent_regularizer=None, # AdaptiveBinaryTernaryRegularizer(lm=1e-1, name="IRNN_1/recurrent"),
        bias_regularizer=bias_regularizer,
        kernel_quantizer=kernel_quantizer,
        recurrent_quantizer=recurrent_quantizer,
        bias_quantizer=bias_quantizer,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        activation=activation_irnn,
        stateful=stateful,
        name="IRNN_1")(encode)
    # encode = tf.keras.layers.Lambda(lambda x: tf.debugging.check_numerics(x, message="Found NaN or Inf"), name="CHK_"+"IRNN_1")(encode)
    # SA Layer
    sa_output = QSelfAttentionMechanismFn(
        num_self_att_hops, 
        num_self_att_units, 
        encode, 
        kernel_regularizer=None,
        bias_regularizer=None,
        kernel_quantizer=kernel_quantizer,
        bias_quantizer=bias_quantizer,
        kernel_initializer=None,
        bias_initializer=None,
        activation=activation_dense,
        name="SA_0")
    # sa_output = tf.keras.layers.Lambda(lambda x: tf.debugging.check_numerics(x, message="Found NaN or Inf"), name="CHK_"+"SA_0_OUTPUT")(sa_output)
    output_proj = QDense(
        num_dense_units, 
        kernel_regularizer=AdaptiveBinaryTernaryRegularizer(lm=1e-1, name="DENSE_0"),
        bias_regularizer=bias_regularizer, 
        kernel_quantizer=kernel_quantizer,
        bias_quantizer=bias_quantizer,
        activation=activation_dense, 
        name="DENSE_0")(sa_output)
    # output_proj = tf.keras.layers.Lambda(lambda x: tf.debugging.check_numerics(x, message="Found NaN or Inf"), name="CHK_"+"DENSE_0")(output_proj)
    # Output logits layers
    output = QDense(
        num_classes, 
        activation=None, 
        kernel_regularizer=AdaptiveBinaryTernaryRegularizer(lm=1e-1, name="DENSE_OUT"),
        bias_regularizer=bias_regularizer, 
        kernel_quantizer=kernel_quantizer,
        bias_quantizer=bias_quantizer,
        name="DENSE_OUT")(output_proj)

    # output = tf.keras.layers.Lambda(lambda x: tf.debugging.check_numerics(x, message="Found NaN or Inf"), name="DENSE_OUT")(output)
    
    # Put model together
    model = tf.keras.Model(inputs=[input],
                        outputs=[output, output_proj], name="quantized_speaker_rec_model_att_like")

    
    return model


def speaker_rec_model_IRNN_att_like(hparams, num_classes, stateful=False, dtype=tf.float32, inference=False):
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
    irnn_stack_size = hparams[HP_IRNN_STACK_SIZE.name]
    irnn_identity_scale = hparams[HP_IRNN_IDENTITY_SCALE.name]
    
    # Activation functions
    activation_irnn = sign_with_ste
    activation_dense = sign_with_ste

    # Activation fns (quantized)
    # activation_irnn = lambda x: custom_relu_quantize_outputs(x, num_bits=8)
    # activation_dense = lambda x: custom_relu_quantize_outputs(x, num_bits=8)

    # Regularizers
    kernel_regularizer = tf.keras.regularizers.L2(l2=0.0001)
    recurrent_regularizer = tf.keras.regularizers.L2(l2=0.0001)
    bias_regularizer = None

    batch_size = None
    if stateful:
        batch_size = 1

    input_shape = [None] # if inference else [MAX_AUDIO_LENGTH]

    # Define model
    # Preprocessing layers (for on device) -------------------------------------------------
    input = tf.keras.Input(
        shape=input_shape, 
        batch_size=batch_size, 
        dtype=dtype, 
        name='PREPROCESS_INPUT')
    encode_in = tf.keras.layers.Lambda(
        lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams, normalize=True), 
        trainable=False, 
        dtype=dtype, 
        name="LOG_MEL_SPEC")(input)
    encode_in = tf.keras.layers.Lambda(
        lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), 
        trainable=False, 
        dtype=dtype, 
        name="DOWNSAMPLE")(encode_in)
    encode_in = tf.keras.layers.Lambda(
        lambda x: ternarize_tensor_with_threshold(x, theta=2/3*tf.reduce_mean(tf.abs(x))),
        trainable=False,
        dtype=dtype,
        name="TERNARIZE_WITH_THRESHOLD"
    )(encode_in)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    # Note: the inference flag is in the NORMALIZE layer if Layer Normalization used in training since it attaches parameters to fixed time length
    # and in inference, we want variable time
    # encode_in = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE", trainable=False, center=not_(inference), scale=not_(inference))(encode_in)
    encode = IRNN( 
        cell=None,
        units=num_lstm_units, 
        # int(num_lstm_units/4),
        identity_scale=irnn_identity_scale,
        stacking_number=irnn_stack_size,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activation=activation_irnn,
        stateful=stateful,
        name="IRNN_0")(encode_in)
    # encode_in = tf.keras.layers.Dense(num_lstm_units, activation="tanh", kernel_regularizer=keras.regularizers.L2(l2=0.0001))(encode_in)
    # encode = tf.keras.layers.Add()([encode, encode_in])
    # encode = tf.keras.layers.Dense(num_lstm_units, kernel_regularizer=keras.regularizers.L2(l2=0.0001), activation=activation_dense, name="IRNN_0_PROJ")(encode)
    encode = TimeReduction(reduction_factor=2, batch_size=batch_size, name="TIME_REDUCTION_E")(encode)
    encode = IRNN(
        cell=None,
        units=num_lstm_units, 
        # int(num_lstm_units/4),
        identity_scale=irnn_identity_scale,
        stacking_number=irnn_stack_size,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activation=activation_irnn,
        stateful=stateful,
        name="IRNN_1")(encode)
    # encode = tf.keras.layers.Dense(num_lstm_units, kernel_regularizer=keras.regularizers.L2(l2=0.0001), activation=activation_dense, name="IRNN_1_PROJ")(encode)

    # SA Layer
    sa_output = SelfAttentionMechanismFn(
        num_self_att_hops, 
        num_self_att_units, 
        encode, 
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activation=activation_dense,
        name="SA_0")

    # sa_stacked_out = tf.keras.layers.Reshape([num_self_att_hops, num_lstm_units], name="SA_1_INPUT")(sa_output)

    # sa_output = SelfAttentionMechanismFn(
    #     1, 
    #     num_self_att_units, 
    #     sa_stacked_out, 
    #     name="SA_1")

    # FOR QUANTIZATION: We need to quantize the attention dot product
    # sa_output = tf.keras.layers.Lambda(lambda x: quantize_over_range(x, num_bits=8), name="SA_0_OUTPUT_QUANTIZE")(sa_output)

    output_proj = tf.keras.layers.Dense(
        num_dense_units, 
        kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer,
        activation=activation_dense, 
        name="DENSE_0")(sa_output)

    # TEST LAYER FOR LINEAR SEPARABILITY
    # output_proj = tf.keras.layers.Dense(2, activation="relu", name="TEST_PROJECTION")(output_proj)

    # Output logits layers
    output = tf.keras.layers.Dense(
        num_classes, 
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activation=None, 
        name="DENSE_OUT")(output_proj)
    
    # Output activation layer
    # output = tf.keras.layers.Activation(activation="sigmoid", name="SIGMOID_OUT")(output)
    # output = tf.keras.layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    encode_in = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False, dtype=dtype)(input)
    # output = tf.keras.layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP, pad_end=True), name="WINDOW")(input)
    # output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    encode_in = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False, dtype=dtype)(encode_in)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    encode_in = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE")(encode_in)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Conv1D(128, 3, strides=2 ,activation="relu", use_bias=True)(output)
    # output = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(output)
    # focus = SelfAttentionMechanismFn(1, num_dense_units, encode_in)
    # focus = tf.repeat(focus, tf.shape(encode_in)[1], 1)
    # focus = tf.keras.layers.Reshape([133, 320])(focus)
    # encode_in = tf.keras.layers.Add()([encode_in, focus])
    # encode_in = tf.keras.layers.LayerNormalization(axis=-2)(encode_in)
    # output = tf.keras.layers.Concatenate(axis=-1, name="ADD_FOCUS")([output, focus])
    # encode = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0", stateful=stateful)(encode_in)
    encode = tf.keras.layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_0", stateful=stateful)(encode_in)
    encode = tf.keras.layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_1", stateful=stateful)(encode)
    encode = tf.keras.layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_2", stateful=stateful)(encode)
    encode = tf.keras.layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_3", stateful=stateful)(encode)
    encode = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION_E")(encode)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # encode, final_memory_state, final_carry_state = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_1", stateful=stateful, return_state=True)(encode)
    encode = tf.keras.layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_4", stateful=stateful)(encode)
    encode = tf.keras.layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_5", stateful=stateful)(encode)
    encode = tf.keras.layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_6", stateful=stateful)(encode)
    encode, final_memory_state = tf.keras.layers.SimpleRNN(num_lstm_units, recurrent_initializer=initializers.Identity(gain=0.01), activation="relu", return_sequences=True, name="RNN_7", stateful=stateful, return_state=True)(encode)

    # Attention layer (try across ts)
    # Let query be last ts output from encode
    # encode = tf.transpose(encode, perm=[0, 2, 1])
    
    # query = encode[:,-1,:]
    # query = tf.expand_dims(query, axis=1)
    # query = tf.repeat(query, tf.shape(encode)[1], 1)
    # query = encode
    # inp = encode[:,1,:]
    # inp = tf.keras.layers.Reshape((1, num_lstm_units))(inp)
    # value = tf.keras.layers.Concatenate(axis=1)([inp, encode[:,1:,:]])
    # output = tf.keras.layers.AdditiveAttention()([query, value])
    # output = tf.keras.layers.Concatenate(axis=-1, name="CONCAT")([encode, encode_2, encode_3, encode_4])

    # decode = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_2", stateful=stateful)(encode)
    # # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION_P")(output)
    # decode = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_3", stateful=stateful)(output)

    # query = decode[:,-1,:]
    # query = tf.expand_dims(query, axis=1)
    # query = tf.repeat(query, tf.shape(encode)[1], 1)
    # query = encode
    # value = encode
    # output = tf.keras.layers.Attention(causal=True)([query, value])

    # SA Layer
    output = SelfAttentionMechanismFn(1, num_dense_units, encode)

    # output = GlobalWeightedMaxPooling1D(name="MAX_POOL")(output)

    # output = decode[:,-1,:]
    output = tf.keras.layers.Reshape([num_lstm_units])(output)

    output = tf.keras.layers.Dense(num_classes, activation=None, name="DENSE_1")(output)
    
    output = tf.keras.layers.Activation(activation="sigmoid", name="SIGMOID_OUT")(output)
    # output = tf.keras.layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    output = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False, dtype=dtype)(input)
    # output = tf.keras.layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP, pad_end=True), name="WINDOW")(input)
    output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False, dtype=dtype)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    # output = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Conv1D(128, 3, strides=2 ,activation="relu", use_bias=True)(output)
    # output = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(output)
    output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0", stateful=stateful)(output)
    output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION_E")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    encode = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_1", stateful=stateful)(output)

    output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_2", stateful=stateful)(encode)
    # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION_P")(output)
    output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_3", stateful=stateful)(output)

    output = tf.keras.layers.Concatenate(axis=-1, name="CONCAT_E_AND_P")([output, encode])

    output = tf.keras.layers.Dense(num_classes, activation="relu", name="DENSE_0")(output)

    output = GlobalWeightedMaxPooling1D(name="VOTE")(output)

    # output = tf.keras.layers.Lambda(lambda x: tf.nn.sigmoid(x), name="SIGMOID_OUT")(output)
    output = tf.keras.layers.Softmax(name="SOFTMAX_OUT", dtype=dtype)(output)

    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    
    # output = tf.keras.layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    output = tf.keras.layers.Lambda(
        lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams, ret_mfcc=True), 
        name="LOG_MEL_SPEC", 
        trainable=False)(input)
    # output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = tf.keras.layers.Lambda(
        lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), 
        name="DOWNSAMPLE", 
        trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_2")(output)
    output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_3")(output)
    output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_4")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    output = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = tf.keras.layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP), name="WINDOW")(input)
    # output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    # output = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)

    output = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(output) # Maybe need to conv over features, not ts

    # Expand for Conv layer compatibility
    output = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name="EXPAND")(output)

    output = tf.keras.layers.Conv2D(96, 7, strides=(2,2), activation="relu", use_bias=True, padding="same", name="conv1")(output)
    output = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name="mpool1")(output)

    output = tf.keras.layers.Conv2D(256, 5, strides=(2,2), activation="relu", use_bias=True, padding="same", name="conv2")(output)
    output = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), name="mpool2")(output)
    
    output = tf.keras.layers.Conv2D(384, 3, strides=(2,2), activation="relu", use_bias=True, padding="same", name="conv3")(output)
    output = tf.keras.layers.Conv2D(256, 3, strides=(1,1), activation="relu", use_bias=True, padding="same", name="conv4")(output)
    output = tf.keras.layers.Conv2D(256, 3, strides=(1,1), activation="relu", use_bias=True, padding="same", name="conv5")(output)
    output = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), name="mpool5")(output)

    shp = K.int_shape(output)
    shp = [shp[1], shp[-1]*shp[-2]]
    output = tf.keras.layers.Reshape(shp, name="SQUEEZE")(output)

    output = tf.keras.layers.Dense(4096, activation="relu", name="fc6")(output)

    output = tf.keras.layers.GlobalAveragePooling1D(name="apool6")(output)

    output = tf.keras.layers.Dense(1024, activation="relu", name="fc7")(output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="fc8")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    output = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_0")(output)
    # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION")(output)
    # output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_1")(output)
    # output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=True, name="LSTM_2")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.LSTM(num_classes, return_sequences=True, name="LSTM_3")(output)
    # output = tf.keras.layers.TimeDistributed(tf.keras.layers.Softmax(), name="TIME_DIST_SOFTMAX_0")(output)
    # output = tf.keras.layers.Lambda(lambda x: tf.math.reduce_prod(x, axis=1), name="MULT_ACROSS")(output)
    output = GlobalWeightedMaxPooling1D()(output)
    output = tf.keras.layers.Softmax()(output)
    # output = tf.keras.layers.Flatten()(output)
    # output = tf.keras.layers.Dense(num_classes, activation="softmax")(output)
    # output = tf.keras.layers.GlobalAveragePooling1D(name="GLOBAL_AVG_POOL")(output)
    # output = tf.keras.layers.Flatten()(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = tf.keras.layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    output = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    output = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    output = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE")(output)
    # output = tf.keras.layers.Reshape((tf.shape(output)[0], tf.shape(output)[1], tf.shape(output)[2], 1))(output)
    output = tf.keras.layers.ConvLSTM1D(64, 3, padding="same", return_sequences=True, name="LSTM_0", data_format="channels_last")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    # output = TimeReduction(reduction_factor=2, batch_size=None, name="TIME_REDUCTION")(output)
    output = tf.keras.layers.LSTM(num_lstm_units, return_sequences=False, name="LSTM_1")(output)
    # output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_lstm_units/2, activation="relu"), name="TIME_DIST_DENSE_0")(output)
    # output = tf.keras.layers.GlobalAveragePooling1D(name="GLOBAL_AVG_POOL")(output)
    # output = tf.keras.layers.Flatten()(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_0")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(num_dense_units, activation="relu", name="DENSE_1")(output)
    # output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="OUT_SOFTMAX")(output)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
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
    input = tf.keras.Input(shape=input_shape, batch_size=batch_size,
                           dtype=dtype, name='INPUT')
    output = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # output = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(output)
    # output = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(output)
    # ---------------------------------------------------------------------------------------

    output = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(output) # Maybe need to conv over features, not ts

    output = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name="EXPAND")(output)

    # Trainable layers ----------------------------------------------------------------------
    output = conv_bn_pool(output,layer_idx=1,conv_filters=96,conv_kernel_size=(7,7),conv_strides=(2,2),conv_pad=(1,1), pool='max',pool_size=(3,3),pool_strides=(2,2))
    output = conv_bn_pool(output,layer_idx=2,conv_filters=256,conv_kernel_size=(5,5),conv_strides=(2,2),conv_pad=(1,1), pool='max',pool_size=(3,3),pool_strides=(2,2))
    output = conv_bn_pool(output,layer_idx=3,conv_filters=384,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
    output = conv_bn_pool(output,layer_idx=4,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
    output = conv_bn_pool(output,layer_idx=5,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1), pool='max',pool_size=(5,3),pool_strides=(3,2))		
    output = conv_bn_dynamic_apool(output,layer_idx=6,conv_filters=4096,conv_kernel_size=(5,1),conv_strides=(1,1),conv_pad=(0,0), conv_layer_prefix='fc')
    output = conv_bn_pool(output,layer_idx=7,conv_filters=1024,conv_kernel_size=(1,1),conv_strides=(1,1),conv_pad=(0,0), conv_layer_prefix='fc')
    output = tf.keras.layers.Lambda(lambda y: K.l2_normalize(y, axis=3), name='norm')(output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name='fc8')(output)
    m = tf.keras.Model(input, output, name='VGGVox')
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
    log_mel_spec = tf.keras.layers.Lambda(lambda x: convert_to_log_mel_spec_layer(x, hparams=hparams), name="LOG_MEL_SPEC", trainable=False)(input)
    # input = tf.keras.layers.Lambda(lambda x: tf.signal.frame(x, FRAME_LENGTH, FRAME_STEP), name="WINDOW")(input)
    norm = tf.keras.layers.Lambda(normalize_log_mel_layer, name="NORMALIZE")(log_mel_spec)
    # input = tf.keras.layers.Lambda(lambda x: group_and_downsample_spec_v2_layer(x, hparams=hparams), name="DOWNSAMPLE", trainable=False)(input)
    # ---------------------------------------------------------------------------------------

    # Trainable layers ----------------------------------------------------------------------
    # input = tf.keras.layers.LayerNormalization(axis=-2, name="NORMALIZE")(input)

    # Expand for Conv layer compatibility
    expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name="EXPAND")(norm)

    # Begin VGG from Nang et al
    # L1_1
    l1_1 = tf.keras.layers.Conv2D(64, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l1_1")(expanded)
    l1_1 = tf.keras.layers.BatchNormalization(name="bn_l1_1")(l1_1)
    l1_1 = tf.keras.layers.Activation(activation="relu", name="relu_l1_1")(l1_1)
    
    # L1_2
    l1_2 = tf.keras.layers.Conv2D(64, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l1_2")(l1_1)
    l1_2 = tf.keras.layers.BatchNormalization(name="bn_l1_2")(l1_2)
    l1_2 = tf.keras.layers.Activation(activation="relu", name="relu_l1_2")(l1_2)

    # L2
    l2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name="mpool_l2")(l1_2)

    # L3
    # L3_1
    l3_1 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l3_1")(l2)
    l3_1 = tf.keras.layers.BatchNormalization(name="bn_l3_1")(l3_1)
    l3_1 = tf.keras.layers.Activation(activation="relu", name="relu_l3_1")(l3_1)
    
    # L3_2
    l3_2 = tf.keras.layers.Conv2D(128, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l3_2")(l3_1)
    l3_2 = tf.keras.layers.BatchNormalization(name="bn_l3_2")(l3_2)
    l3_2 = tf.keras.layers.Activation(activation="relu", name="relu_l3_2")(l3_2)

    # L4
    l4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name="mpool_l4")(l3_2)

    # L5
    # L5_1
    l5_1 = tf.keras.layers.Conv2D(256, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l5_1")(l4)
    l5_1 = tf.keras.layers.BatchNormalization(name="bn_l5_1")(l5_1)
    l5_1 = tf.keras.layers.Activation(activation="relu", name="relu_l5_1")(l5_1)
    
    # L5_2
    l5_2 = tf.keras.layers.Conv2D(256, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l5_2")(l5_1)
    l5_2 = tf.keras.layers.BatchNormalization(name="bn_l5_2")(l5_2)
    l5_2 = tf.keras.layers.Activation(activation="relu", name="relu_l5_2")(l5_2)

    # L6
    l6 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name="mpool_l6")(l5_2)

    # L7
    # L7_1
    l7_1 = tf.keras.layers.Conv2D(512, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l7_1")(l6)
    l7_1 = tf.keras.layers.BatchNormalization(name="bn_l7_1")(l7_1)
    l7_1 = tf.keras.layers.Activation(activation="relu", name="relu_l7_1")(l7_1)
    
    # L7_2
    l7_2 = tf.keras.layers.Conv2D(512, 3, strides=(1,1), use_bias=True, padding="same", name="conv_l7_2")(l7_1)
    l7_2 = tf.keras.layers.BatchNormalization(name="bn_l7_2")(l7_2)
    l7_2 = tf.keras.layers.Activation(activation="relu", name="relu_l7_2")(l7_2)

    # L8
    l8 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same", name="mpool_l8")(l7_2)

    # L9 - Self-Attention Mechanism
    # matmul1
    shp = K.int_shape(l8)
    T = shp[1]
    n_h = shp[2]*shp[3]
    n_c = shp[3]
    n_k = 4
    l9_in = tf.keras.layers.Reshape((T, n_h), input_shape=shp[1:], name="concat_hiddens_l9")(l8) # combine hidden vectors
    l9_matmul1 = tf.keras.layers.Dense(n_c, activation="tanh", use_bias=False, name="matmul1_l9")(l9_in)

    # matmul2
    l9_matmul2 = tf.keras.layers.Dense(n_k, activation="softmax", use_bias=False, name="matmul2_l9")(l9_matmul1)

    # transpose
    l9_trans = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]), name="transpose_l9")(l9_matmul2)
    
    # matmul3
    l9_matmul3 = tf.keras.layers.Dot(axes=[2,1], name="matmul3_l9")([l9_trans, l9_in])

    # L10
    l10 = tf.keras.layers.GlobalAveragePooling1D(name="avgpool_l10")(l9_matmul3)

    # L11
    l11 = tf.keras.layers.Dense(256, activation="relu", use_bias=True, name="dense_l11")(l10)

    # L12
    l12 = tf.keras.layers.Dense(num_classes, activation=None, use_bias=True, name="dense_l12")(l11)

    output = tf.keras.layers.Softmax(name="SOFTMAX_OUT")(l12)
    # ---------------------------------------------------------------------------------------

    # Put model together
    model = tf.keras.Model(inputs=[input],
                        outputs=[output], name="vgg_nang_et_al")
    return model