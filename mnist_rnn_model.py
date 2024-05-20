from qkeras import *

from utils.model_utils import TimeReduction, QDenseWithNorm, QIRNN, ModelWithGradInfo, GeneralActivation, QSelfAttentionMechanismFn
from quantization import ternarize_tensor_with_threshold, LearnedThresholdTernary

SEED = 1997

# Regularizers (None for quantization I believe)
kernel_regularizer = None # NoAccRegularizer(0.001, k=256) # tf.keras.regularizers.L2(0.0001) # tf.keras.regularizers.L1(l1=1e-5) # VarianceRegularizer(l2=2.0) # tf.keras.regularizers.L2(0.0001) # TernaryEuclideanRegularizer(l2=0.00001, beta=4) # tf.keras.regularizers.L2(0.0001)
recurrent_regularizer = None # NoAccRegularizer(0.001, k=256) # tf.keras.regularizers.L2(0.0001) # tf.keras.regularizers.L1(l1=1e-5) # VarianceRegularizer(l2=2.0) # tf.keras.regularizers.L2(0.0001) # BinaryEuclideanRegularizer(l2=1e-6) #TernaryEuclideanRegularizer(l2=0.00001, beta=4) # tf.keras.regularizers.L2(0.0001) # TernaryEuclideanRegularizer(l2=0.00001, beta=4)
bias_regularizer = None
activation_regularizer = None # tf.keras.regularizers.L2(l2=0.0001)

# Quantization functions (General)
kernel_quantizer = None # ternary(alpha=1, threshold=0.05) # binary(alpha=0.5) # stochastic_ternary(alpha=1, threshold=0.01) # ternary(alpha=1, threshold=0.1) # quantized_bits(bits=4, integer=0, symmetric=1, keep_negative=True, alpha=1.0) # ternary(alpha=1, threshold=lambda x: 0.7*tf.reduce_mean(tf.abs(x))) # quantized_bits(bits=2, integer=2, symmetric=1, keep_negative=True)
recurrent_quantizer = None # ternary(alpha=1, threshold=0.05) # binary(alpha=0.5) # stochastic_ternary(alpha=1, threshold=0.02) # ternary(alpha=1, threshold=0.1) # quantized_bits(bits=4, integer=0, symmetric=1, keep_negative=True, alpha=1.0) # ternary(alpha=1, threshold=lambda x: 0.08) # quantized_bits(bits=2, integer=2, symmetric=1, keep_negative=True)
bias_quantizer = None # ternary(alpha=1) # quantized_bits(bits=8, integer=8, symmetric=1, keep_negative=True)

# Initializers
rnn_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=SEED) # "he_normal" is default 
rnn_recurrent_initializer = tf.keras.initializers.Orthogonal(gain=1.0, seed=SEED) # keep as None
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_in", distribution="truncated_normal", seed=SEED) # "he_normal" is default 

def get_ff_model(options):
    input_og = tf.keras.layers.Input(shape=(28, 28, 1))
    input = tf.keras.layers.Lambda(
            lambda x: tf.stop_gradient(ternarize_tensor_with_threshold(x, theta=0.7*tf.reduce_mean(tf.abs(x)))),
            trainable=False,
            dtype=tf.float32,
            name="TERNARIZE_WITH_THRESHOLD"
        )(input_og) if options["tern"] else tf.keras.layers.Lambda(lambda x: x, name="NOOP")(input_og)
    # input = tf.keras.layers.Lambda(
    #     lambda x: tf.stop_gradient(quantized_bits(5, 4, symmetric=True, keep_negative=True)(x)),
    #     trainable=False,
    #     dtype=tf.float32,
    #     name="TERNARIZE_WITH_THRESHOLD"
    # )(input) if options["tern"] else tf.keras.layers.Lambda(lambda x: x, name="NOOP")(input)
    input = tf.keras.layers.Flatten()(input)
    dense_0 = QDenseWithNorm(
        1024, # 1024
        activation=GeneralActivation(activation=options["activation_dense"], name="DENSE_0"), 
        use_bias=False,
        kernel_regularizer=kernel_regularizer, 
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=options["g"]*options["tern_params"]["DENSE_0"], 
            name="DENSE_0") if options["learned_thresh"] else kernel_quantizer, # used 0.7*mean(|w|)
        kernel_initializer=dense_kernel_initializer,
        add_dist_loss=options["add_dist_loss"],
        add_no_acc_reg=options["oar"]["use"],
        no_acc_reg_lm=options["oar"]["lm"],
        no_acc_reg_bits=options["oar"]["precision"],
        s=1,
        name="DENSE_0",)(input)
    dense_1 = QDenseWithNorm(
        1024, # 1024
        activation=GeneralActivation(activation=options["activation_dense"], name="DENSE_1"), 
        use_bias=False,
        kernel_regularizer=kernel_regularizer, 
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=options["g"]*options["tern_params"]["DENSE_1"], 
            name="DENSE_1") if options["learned_thresh"] else kernel_quantizer, # used 0.7*mean(|w|)
        kernel_initializer=dense_kernel_initializer,
        add_dist_loss=options["add_dist_loss"],
        add_no_acc_reg=options["oar"]["use"],
        no_acc_reg_lm=options["oar"]["lm"],
        no_acc_reg_bits=options["oar"]["precision"],
        s=1,
        name="DENSE_1",)(dense_0)
    output = QDenseWithNorm(
        10, # 1024
        activation=GeneralActivation(activation=tf.keras.activations.softmax, name="DENSE_OUT"), 
        use_bias=False,
        kernel_regularizer=kernel_regularizer, 
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=0.7*options["tern_params"]["DENSE_OUT"], 
            name="DENSE_OUT") if options["learned_thresh"] else kernel_quantizer, # used 0.7*mean(|w|)
        kernel_initializer=dense_kernel_initializer,
        add_dist_loss=False,
        add_no_acc_reg=False,
        no_acc_reg_lm=options["oar"]["lm"],
        no_acc_reg_bits=options["oar"]["precision"],
        s=1,
        name="DENSE_OUT",)(dense_1)
    
    model = ModelWithGradInfo(inputs=[input_og],
                        outputs=[output], name="MNIST_FF_1024")

    model.summary()

    return model


def get_model(options, layer_options):
    input = tf.keras.layers.Input(shape=(28, 28, 1))
    input = tf.keras.layers.Lambda(
            lambda x: tf.stop_gradient(ternarize_tensor_with_threshold(x, theta=0.7*tf.reduce_mean(tf.abs(x)))),
            trainable=False,
            dtype=tf.float32,
            name="TERNARIZE_WITH_THRESHOLD"
        )(input) if options["tern"] else tf.keras.layers.Lambda(lambda x: x, name="NOOP")(input)
    # input = tf.keras.layers.Lambda(
    #     lambda x: tf.stop_gradient(quantized_bits(5, 0, symmetric=True, keep_negative=True)(x)),
    #     trainable=False,
    #     dtype=tf.float32,
    #     name="TERNARIZE_WITH_THRESHOLD"
    # )(input) if options["tern"] else tf.keras.layers.Lambda(lambda x: x, name="NOOP")(input)
    input = tf.keras.layers.Reshape(target_shape=(28, 28))(input)
    qrnn_0 = QIRNN(
        cell=None,
        units=128, 
        activation=GeneralActivation(activation=layer_options["QRNN_0"]["activation"], name="QRNN_0"), 
        use_bias=False, 
        return_sequences=True, 
        kernel_regularizer=kernel_regularizer, 
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["QRNN_0"]["tern_quant_thresh"], 
            name="QRNN_0/quantized_kernel") if options["learned_thresh"] else kernel_quantizer, # 0.1
        recurrent_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["QRNN_0"]["tern_quant_thresh"], 
            name="QRNN_0/quantized_recurrent") if options["learned_thresh"] else recurrent_quantizer, # 0.05
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        add_dist_loss=layer_options["QRNN_0"]["add_dist_loss"],
        add_no_acc_reg=layer_options["QRNN_0"]["oar"]["use"],
        no_acc_reg_lm=layer_options["QRNN_0"]["oar"]["lm"],
        no_acc_reg_bits=layer_options["QRNN_0"]["oar"]["precision"],
        s=layer_options["QRNN_0"]["s"],
        name="QRNN_0")(input)
    tr = TimeReduction(reduction_factor=2)(qrnn_0)
    qrnn_1 = QIRNN(
        cell=None,
        units=128,  
        activation=GeneralActivation(activation=layer_options["QRNN_1"]["activation"], name="QRNN_1"), 
        use_bias=False, 
        return_sequences=True, 
        kernel_regularizer=kernel_regularizer, 
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["QRNN_1"]["tern_quant_thresh"], 
            name="QRNN_1/quantized_kernel") if options["learned_thresh"] else kernel_quantizer, # 0.045
        recurrent_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["QRNN_1"]["tern_quant_thresh"], 
            name="QRNN_1/quantized_recurrent") if options["learned_thresh"] else recurrent_quantizer, # 0.06
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        add_dist_loss=layer_options["QRNN_1"],
        add_no_acc_reg=layer_options["QRNN_1"]["oar"]["use"],
        no_acc_reg_lm=layer_options["QRNN_1"]["oar"]["lm"],
        no_acc_reg_bits=layer_options["QRNN_1"]["oar"]["precision"],
        s=layer_options["QRNN_1"]["s"],
        name="QRNN_1")(tr)
    qrnn_1 = tf.keras.layers.Flatten()(qrnn_1)
    # sa_0 = QSelfAttentionMechanismFn(
    #     2, 
    #     128, 
    #     qrnn_1, 
    #     kernel_regularizer=kernel_regularizer, # if kernel_regularizer else None,
    #     bias_regularizer=bias_regularizer, # if bias_regularizer else None,
    #     kernel_quantizer=kernel_quantizer,
    #     bias_quantizer=bias_quantizer,
    #     kernel_initializer=dense_kernel_initializer,
    #     bias_initializer=None,
    #     activation=layer_options["SA_0_QDENSE_0"]["activation"],
    #     use_bias=False,
    #     norm=None,
    #     fold_batch_norm=False,
    #     soft_thresh_tern=False,
    #     learned_thresh=options["learned_thresh"],
    #     add_dist_loss=layer_options["SA_0_QDENSE_0"],
    #     add_no_acc_reg=layer_options["SA_0_QDENSE_0"]["oar"]["use"],
    #     no_acc_reg_lm=layer_options["SA_0_QDENSE_0"]["oar"]["lm"],
    #     no_acc_reg_bits=layer_options["SA_0_QDENSE_0"]["oar"]["precision"],
    #     dropout=False,
    #     s=1,
    #     layer_options=layer_options,
    #     name="SA_0")
    dense_0 = QDenseWithNorm(
        1024, # 1024
        activation=GeneralActivation(activation=layer_options["DENSE_0"]["activation"], name="DENSE_0"), 
        use_bias=False,
        kernel_regularizer=kernel_regularizer, 
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["DENSE_0"]["tern_quant_thresh"], 
            name="DENSE_0") if options["learned_thresh"] else kernel_quantizer, # used 0.7*mean(|w|)
        kernel_initializer=dense_kernel_initializer,
        add_dist_loss=layer_options["DENSE_0"],
        add_no_acc_reg=layer_options["DENSE_0"]["oar"]["use"],
        no_acc_reg_lm=layer_options["DENSE_0"]["oar"]["lm"],
        no_acc_reg_bits=layer_options["DENSE_0"]["oar"]["precision"],
        s=layer_options["DENSE_0"]["s"],
        name="DENSE_0",)(qrnn_1)
    output = QDenseWithNorm(
        10, 
        use_bias=False,
        activation=GeneralActivation(activation=layer_options["DENSE_OUT"]["activation"], name="DENSE_OUT"), 
        kernel_regularizer=kernel_regularizer, 
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["DENSE_OUT"]["tern_quant_thresh"], 
            name="DENSE_OUT") if options["learned_thresh"] else kernel_quantizer, # 0.7*mean(|w|)
        kernel_initializer=dense_kernel_initializer, 
        add_dist_loss=layer_options["DENSE_OUT"],
        add_no_acc_reg=layer_options["DENSE_OUT"]["oar"]["use"],
        no_acc_reg_lm=layer_options["DENSE_OUT"]["oar"]["lm"],
        no_acc_reg_bits=layer_options["DENSE_OUT"]["oar"]["precision"],
        s=layer_options["DENSE_OUT"]["s"],
        name="DENSE_OUT",)(dense_0)
    
    model = ModelWithGradInfo(inputs=[input],
                        outputs=[output], name="MNIST_SIMPLE_RNN")

    model.summary()

    return model