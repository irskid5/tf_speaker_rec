from qkeras import *

from utils.model_utils import TimeReduction, sign_with_tanh_deriv, QDenseWithNorm, QIRNN, custom_sign_with_tanh_deriv_mod_on_inputs, ModelWithGradInfo, GeneralActivation, mod_on_inputs
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

# Optional
soft_thresh_tern = False
learned_thresh = True
tern = True
add_no_acc_reg = True
no_acc_reg_lm = 1e-4
add_dist_loss = False
acc_precision = 6

# Activation functions
# activation_irnn = tf.keras.activations.tanh
# activation_dense = tf.keras.activations.tanh

# Activation fns (quantized)
# activation_irnn = sign_with_tanh_deriv
# activation_dense = sign_with_tanh_deriv

# Activation fns (quantized with mod)
activation_irnn = lambda x: custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=acc_precision)
activation_dense = lambda x: custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=acc_precision)

g = 1.5
layer_options = {
    "QRNN_0": {"tern_quant_thresh": g*0.0732264, "g": g,}, 
    "QRNN_1": {"tern_quant_thresh": g*0.0628442243, "g": g,}, 
    "DENSE_0": {"tern_quant_thresh": g*0.0307561122, "g": g,}, 
    "DENSE_OUT": {"tern_quant_thresh": g*0.0557919517, "g": g,}
} # mean|W| for 20221222-202422 init

# layer_options = {
#     "QRNN_0": {"tern_quant_thresh": g*0.10521546, "g": g,}, 
#     "QRNN_1": {"tern_quant_thresh": g*0.0769046694, "g": g,}, 
#     "DENSE_0": {"tern_quant_thresh": g*0.0381105281, "g": g,}, 
#     "DENSE_OUT": {"tern_quant_thresh": g*0.0680549815, "g": g,}
# } # std W for 20221222-202422 init

# Initializers
rnn_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.5 if soft_thresh_tern else 1.0, mode="fan_avg", distribution="uniform", seed=SEED) # "he_normal" is default 
rnn_recurrent_initializer = tf.keras.initializers.Orthogonal(gain=0.5 if soft_thresh_tern else 1.0, seed=SEED) # keep as None
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1.0 if soft_thresh_tern else 2.0, mode="fan_in", distribution="truncated_normal", seed=SEED) # "he_normal" is default 

def get_model():
    input = tf.keras.layers.Input(shape=(28, 28, 1))
    input = tf.keras.layers.Lambda(
            lambda x: tf.stop_gradient(ternarize_tensor_with_threshold(x, theta=0.7*tf.reduce_mean(tf.abs(x)))),
            trainable=False,
            dtype=tf.float32,
            name="TERNARIZE_WITH_THRESHOLD"
        )(input) if tern else tf.keras.layers.Lambda(lambda x: x, name="NOOP")(input)
    input = tf.keras.layers.Reshape(target_shape=(28, 28))(input)
    qrnn_0 = QIRNN(
        cell=None,
        units=128, 
        activation=GeneralActivation(activation=activation_irnn, name="QRNN_0"), 
        use_bias=False, 
        return_sequences=True, 
        kernel_regularizer=kernel_regularizer, 
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["QRNN_0"]["tern_quant_thresh"], 
            name="QRNN_0/quantized_kernel") if learned_thresh else kernel_quantizer, # 0.1
        recurrent_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["QRNN_0"]["tern_quant_thresh"], 
            name="QRNN_0/quantized_recurrent") if learned_thresh else recurrent_quantizer, # 0.05
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        add_dist_loss=add_dist_loss,
        add_no_acc_reg=add_no_acc_reg,
        no_acc_reg_lm=no_acc_reg_lm,
        no_acc_reg_bits=acc_precision,
        s=4,
        name="QRNN_0")(input)
    tr = TimeReduction(reduction_factor=2)(qrnn_0)
    qrnn_1 = QIRNN(
        cell=None,
        units=128,  
        activation=GeneralActivation(activation=activation_irnn, name="QRNN_1"), 
        use_bias=False, 
        return_sequences=True, 
        kernel_regularizer=kernel_regularizer, 
        recurrent_regularizer=recurrent_regularizer,
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["QRNN_1"]["tern_quant_thresh"], 
            name="QRNN_1/quantized_kernel") if learned_thresh else kernel_quantizer, # 0.045
        recurrent_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["QRNN_1"]["tern_quant_thresh"], 
            name="QRNN_1/quantized_recurrent") if learned_thresh else recurrent_quantizer, # 0.06
        kernel_initializer=rnn_kernel_initializer,
        recurrent_initializer=rnn_recurrent_initializer,
        add_dist_loss=add_dist_loss,
        add_no_acc_reg=add_no_acc_reg,
        no_acc_reg_lm=no_acc_reg_lm,
        no_acc_reg_bits=acc_precision,
        s=4,
        name="QRNN_1")(tr)
    qrnn_1 = tf.keras.layers.Flatten()(qrnn_1)
    dense_0 = QDenseWithNorm(
        1024, 
        activation=GeneralActivation(activation=activation_dense, name="DENSE_0"), 
        use_bias=False,
        kernel_regularizer=kernel_regularizer, 
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["DENSE_0"]["tern_quant_thresh"], 
            name="DENSE_0") if learned_thresh else kernel_quantizer, # used 0.7*mean(|w|)
        kernel_initializer=dense_kernel_initializer,
        add_dist_loss=add_dist_loss,
        add_no_acc_reg=add_no_acc_reg,
        no_acc_reg_lm=no_acc_reg_lm,
        no_acc_reg_bits=acc_precision,
        s=1,
        name="DENSE_0",)(qrnn_1)
    output = QDenseWithNorm(
        10, 
        use_bias=False,
        activation=GeneralActivation(activation=lambda x: tf.keras.activations.softmax(x), name="DENSE_OUT"), 
        kernel_regularizer=tf.keras.regularizers.L1(5e-4), 
        kernel_quantizer=LearnedThresholdTernary(
            scale=1.0, 
            threshold=layer_options["DENSE_OUT"]["tern_quant_thresh"], 
            name="DENSE_OUT") if learned_thresh else kernel_quantizer, # 0.7*mean(|w|)
        kernel_initializer=dense_kernel_initializer, 
        add_dist_loss=add_dist_loss,
        add_no_acc_reg=True,
        no_acc_reg_lm=0,
        no_acc_reg_bits=acc_precision,
        s=1,
        name="DENSE_OUT",)(dense_0)
    
    model = ModelWithGradInfo(inputs=[input],
                        outputs=[output], name="MNIST_SIMPLE_RNN")

    model.summary()

    return model