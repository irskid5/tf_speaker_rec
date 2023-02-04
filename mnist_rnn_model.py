from qkeras import *

from utils.model_utils import TimeReduction, sign_with_tanh_deriv, QDenseWithNorm, QIRNN, custom_sign_with_tanh_deriv_mod_on_inputs
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
add_no_acc_reg = False
add_dist_loss = False
acc_precision = 6

# Activation functions
activation_irnn = tf.keras.activations.tanh
activation_dense = tf.keras.activations.tanh

# Activation fns (quantized)
activation_irnn = sign_with_tanh_deriv
activation_dense = sign_with_tanh_deriv

# Activation fns (quantized with mod)
# activation_irnn = lambda x: custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=acc_precision)
# activation_dense = lambda x: custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=acc_precision)

g = 1.0
layer_options = {
    "QRNN_0": {"tern_quant_thresh": g*0.0755566359, "g": g,}, 
    "QRNN_1": {"tern_quant_thresh": g*0.0443275981, "g": g,}, 
    "DENSE_0": {"tern_quant_thresh": g*0.0225049816, "g": g,}, 
    "DENSE_OUT": {"tern_quant_thresh": g*0.0389710329, "g": g,}
} # std for 20221222-202422 init

# Initializers
rnn_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.5 if soft_thresh_tern else 1.0, mode="fan_avg", distribution="uniform", seed=SEED) # "he_normal" is default 
rnn_recurrent_initializer = tf.keras.initializers.Orthogonal(gain=0.5 if soft_thresh_tern else 1.0, seed=SEED) # keep as None
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1.0 if soft_thresh_tern else 2.0, mode="fan_in", distribution="truncated_normal", seed=SEED) # "he_normal" is default 

def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((28, 28, 1)),
            tf.keras.layers.Lambda(
                lambda x: tf.stop_gradient(ternarize_tensor_with_threshold(x, theta=0.7*tf.reduce_mean(tf.abs(x)))),
                trainable=False,
                dtype=tf.float32,
                name="TERNARIZE_WITH_THRESHOLD"
            ) if tern else tf.keras.layers.Lambda(lambda x: x, name="NOOP"),
            tf.keras.layers.Reshape(target_shape=(28, 28)),
            QIRNN(
                cell=None,
                units=128, 
                activation=activation_irnn, 
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
                no_acc_reg_lm=0.001,
                no_acc_reg_bits=acc_precision,
                s=1,
                name="QRNN_0"),
            TimeReduction(reduction_factor=2),
            QIRNN(
                cell=None,
                units=128,  
                activation=activation_irnn, 
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
                no_acc_reg_lm=0.001,
                no_acc_reg_bits=acc_precision,
                s=1,
                name="QRNN_1"),
            tf.keras.layers.Flatten(),
            QDenseWithNorm(
                1024, 
                activation=activation_dense, 
                use_bias=False,
                kernel_regularizer=kernel_regularizer, 
                kernel_quantizer=LearnedThresholdTernary(
                    scale=1.0, 
                    threshold=layer_options["DENSE_0"]["tern_quant_thresh"], 
                    name="DENSE_0") if learned_thresh else kernel_quantizer, # used 0.7*mean(|w|)
                kernel_initializer=dense_kernel_initializer,
                add_dist_loss=add_dist_loss,
                add_no_acc_reg=add_no_acc_reg,
                no_acc_reg_lm=0.005,
                no_acc_reg_bits=acc_precision,
                s=1,
                name="DENSE_0",),
            QDenseWithNorm(
                10, 
                use_bias=False,
                activation=None, 
                kernel_regularizer=kernel_regularizer, 
                kernel_quantizer=LearnedThresholdTernary(
                    scale=1.0, 
                    threshold=layer_options["DENSE_OUT"]["tern_quant_thresh"], 
                    name="DENSE_OUT") if learned_thresh else kernel_quantizer, # 0.7*mean(|w|)
                kernel_initializer=dense_kernel_initializer, 
                add_dist_loss=add_dist_loss,
                add_no_acc_reg=False,
                s=1,
                name="DENSE_OUT",),
        ],
        name="MNIST_SIMPLE_RNN",
    )

    model.summary()

    return model