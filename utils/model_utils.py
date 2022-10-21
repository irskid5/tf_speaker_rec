import imp
from math import pi
import tensorflow as tf
import keras.backend as K
import numpy as np
# GENERAL UTILS
from utils.general import *

# QUANTIZATION FUNCTIONS
from quantization import *

from qkeras import *

class IRNN(tf.keras.layers.RNN):
    '''
    Model from arXiv:1504.00941v2
    A Simple Way to Initialize Recurrent Networks of
    Rectified Linear Units
    '''
    def __init__(
        self, 
        cell=None, 
        units=256, 
        identity_scale=0.1, 
        stacking_number=1, 
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activation="relu", 
        stateful=False, 
        name="", 
        **kwargs):
        cell = tf.keras.layers.StackedRNNCells([tf.keras.layers.SimpleRNNCell(
                    units,
                    activation=activation,
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.001),
                    recurrent_initializer=tf.keras.initializers.Identity(gain=identity_scale),
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                ) for _ in range(stacking_number)]) if not cell else cell 
            
        super(IRNN, self).__init__(cell, return_sequences=True, stateful=stateful, name=name)
        self.identity_scale = identity_scale
        self.stacking_number = stacking_number
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        

    def get_config(self):
        config = super().get_config()
        config.update({
            "identity_scale": self.identity_scale,
            "stacking_number": self.stacking_number,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "bias_regularizer": self.bias_regularizer
        })
        return config

class QIRNN(tf.keras.layers.RNN):
    '''
    Model from arXiv:1504.00941v2
    A Simple Way to Initialize Recurrent Networks of
    Rectified Linear Units
    '''
    def __init__(
        self, 
        cell=None, 
        units=256, 
        identity_scale=0.1, 
        stacking_number=1, 
        activation="relu", 
        stateful=False, 
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        kernel_quantizer=None,
        recurrent_quantizer=None,
        bias_quantizer=None,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        name="", **kwargs):
        cell = tf.keras.layers.StackedRNNCells([QSimpleRNNCell(
                    units,
                    activation=activation,
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.001),
                    recurrent_initializer=tf.keras.initializers.Identity(gain=identity_scale),
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_quantizer=kernel_quantizer,
                    recurrent_quantizer=recurrent_quantizer,
                    bias_quantizer=bias_quantizer,
                ) for _ in range(stacking_number)]) if not cell else cell 
            
        super(QIRNN, self).__init__(cell, return_sequences=True, stateful=stateful, name=name)
        self.identity_scale = identity_scale
        self.stacking_number = stacking_number
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_quantizer = kernel_quantizer
        self.recurrent_quantizer = recurrent_quantizer
        self.bias_quantizer = bias_quantizer

    def get_config(self):
        config = super().get_config()
        config.update({
            "identity_scale": self.identity_scale,
            "stacking_number": self.stacking_number,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "kernel_quantizer": self.kernel_quantizer,
            "recurrent_quantizer": self.recurrent_quantizer,
            "bias_quantizer": self.bias_quantizer,
        })
        return config

class IRNNWithProjection(tf.keras.layers.RNN):
    '''
    Model from arXiv:1504.00941v2
    A Simple Way to Initialize Recurrent Networks of
    Rectified Linear Units
    +
    Projection layer between individual RNNs
    '''
    def __init__(self, units, projection_units, identity_scale, stacking_number=1, activation="relu", stateful=False, name="", **kwargs):
        cell = [SimpleRNNCellWithProjection(
                    units,
                    projection_units,
                    activation=activation,
                    kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.001),
                    recurrent_initializer=tf.keras.initializers.Identity(gain=identity_scale),
                ) for _ in range(stacking_number)]
            
        super(IRNNWithProjection, self).__init__(cell, return_sequences=True, stateful=stateful, name=name, **kwargs)
        self.identity_scale = identity_scale
        self.stacking_number = stacking_number
        

    def get_config(self):
        config = super().get_config()
        config.update({
            "identity_scale": self.identity_scale,
            "stacking_number": self.stacking_number,
        })
        return config

@tf.custom_gradient
def sign_with_ste(x):
    """
    Compute the signum function in the fwd pass but return STE approximation for grad in bkwd pass
    """
    out = tf.stop_gradient(K.sign(x))
    # out = tf.stop_gradient(tf.where(tf.equal(x, tf.constant(0, tf.float32)), tf.ones_like(x), x))
    def grad(upstream):
        return tf.where(K.less_equal(K.abs(x), 1), upstream, 0*upstream)
    return out, grad

def custom_relu_mod_on_inputs(x, num_bits=8):
    """
    Compute x mod 2**num_bits then run relu.
    Note, no gradient passes from mod op by using tf.stop_gradient
    """
    # inner function to wrap stop gradient around to not take gradient into account
    def _inner_fn(x, num_bits):
        base = 2**num_bits
        half_base = 2**(num_bits-1)

        # Cast to int to do modular reduction
        x_int = tf.cast(x, tf.int32)

        # Perform modular reduction (creating unsigned int)
        modded = tf.math.mod(x_int, base)

        # Sign the int 
        signed = tf.where(tf.greater_equal(modded, half_base), modded-base, modded)

        # Cast back to float
        signed_float = tf.cast(signed, tf.float32)

        return signed_float

    out = tf.stop_gradient(_inner_fn(x, num_bits=num_bits)) # first mod
    out = K.relu(out) # then relu

    # For seeing distribution of input to activation
    # (Note: The try must be kept or else it will fail on the initial tf graphing)
    # try:
    #     plot_histogram_discrete(x, "histogram_of_wxplusb.png")
    #     plot_histogram_discrete(signed_float, "histogram_of_wxplusb_modded.png")
    # except:
    #     return out
    return out

def custom_relu_mod_on_outputs(x, num_bits=8):
    """
    Compute y = relu(x) then y mod 2**num_bits.
    Note, no gradient passes from mod op by using tf.stop_gradient
    """
    # Inner function to wrap stop gradient around to not take gradient into account
    def _inner_fn(x, num_bits):
        # Calculate modulus based on num_bits
        base = 2**num_bits
        half_base = 2**(num_bits-1)

        # Cast to int to do modular reduction
        out = tf.cast(x, tf.int32)

        # Perform modular reduction (creating unsigned int)
        modded = tf.math.mod(out, base)

        # Sign the int 
        out = tf.where(tf.greater_equal(modded, half_base), modded-base, modded)

        # Cast back to float
        out = tf.cast(out, tf.float32)

        return out

    out = K.relu(x) # first relu
    out = tf.stop_gradient(_inner_fn(out, num_bits=num_bits)) # then mod

    return out

def custom_relu_quantize_outputs(x, num_bits=4):
    out = K.relu(x)
    out = tf.stop_gradient(ternarize_tensor(out))
    return out

def custom_relu_with_threshold(x, max_value=6):
    """
    ReLU with specified max_value (i.e, if x < 0: 0; 0 <= x < theshold: x; x >= threshold: threshold)
    """
    out = K.relu(x, max_value=max_value)

    # For seeing distribution of input to activation
    # (Note: The try must be kept or else it will fail on the initial tf graphing)
    # try:
    #     plot_histogram_continous(x, "histogram_of_wxplusb.png")
    #     plot_histogram_continous(out, "histogram_of_relu_wxplusb.png")
    # except:
    #     return out
    return out

def soft_root_sign(x, alpha=1, beta=1):
    """
    From arXiv:2003.00547v1, Soft-Root-Sign Activation Function
    with fixed alpha and beta
    """
    # tf.Assert(alpha > 0 and beta > 0, None, "Either Alpha or Beta in soft_root_sign function are <0")

    x_over_alpha = x / alpha
    x_over_beta = x / beta
    den = x_over_alpha + tf.exp(-x_over_beta)
    out = x / den
    
    return out

def sign_with_htanh_deriv(x):
    out = hard_tanh(x)
    q = tf.math.sign(x)
    q += (1.0 - tf.math.abs(q))
    return out + tf.stop_gradient(-out + q)

def sign_swish(x, beta=5):
    """
    Adapted from arXiv:1812.11800v3 "Regularized Binary Network Training"
    """
    beta_x = beta*x
    temped_sigmoid = tf.sigmoid(beta_x)
    out = 2*temped_sigmoid*(1+beta_x*(1-temped_sigmoid))-1
    return out

class TrainableSignSwish(tf.Module):
    def __init__(self, a_init=None, name=""):
        self.a = tf.Variable(
            initial_value=tf.random.uniform([], minval=1e-4, maxval=1e1),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=1e-4, clip_value_max=1e1),
            name=name+"/SS_act/a"
        )

    def __call__(self, x):
        out = self.a * sign_swish(x)
        return out

    def get_config(self):
        return {'a': float(self.a)}

@tf.keras.utils.register_keras_serializable(package='Custom', name='rb1')
class BinaryManhattanRegularizer(tf.keras.regularizers.Regularizer):
  """
  Adapted from arXiv:1812.11800v3 "Regularized Binary Network Training"
  """
  def __init__(self, l2=0.):
    self.l2 = l2

  def __call__(self, x):
    """
    R1(x) = |1 - |x||
    """
    return self.l2 * tf.math.reduce_sum(tf.math.abs(1-tf.math.abs(x)))

  def get_config(self):
    return {'l2': float(self.l2)}

@tf.keras.utils.register_keras_serializable(package='Custom', name='rt1')
class TernaryManhattanRegularizer(tf.keras.regularizers.Regularizer):
  """
  Based on arXiv:1812.11800v3 "Regularized Binary Network Training"
  This is ternary manhatten regularization with +- 1/2 fixed.
  """
  def __init__(self, l2=0., beta=2.):
    self.l2 = l2
    self.beta = beta

  def __call__(self, x):
    """
    R1(x) = beta * ||x| + |x-1| + |x+1| - |x-1/2| - |x+1/2| - 1|
    """
    regularized_x = tf.math.abs(x) + tf.math.abs(x - 1.0) + tf.math.abs(x + 1.0) - tf.math.abs(x - 0.5) - tf.math.abs(x + 0.5) - 1.0
    return self.l2 * self.beta * tf.math.reduce_sum(tf.math.abs(regularized_x))

  def get_config(self):
    return {'l2': float(self.l2), 'beta': float(self.beta)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='rb2')
class BinaryEuclideanRegularizer(tf.keras.regularizers.Regularizer):
  """
  Adapted from arXiv:1812.11800v3 "Regularized Binary Network Training"
  """
  def __init__(self, l2=0.):
    self.l2 = l2

  def __call__(self, x):
    """
    R2(x) = (1 - |x|)^2
    """
    return self.l2 * tf.math.reduce_sum(tf.math.square(1-tf.math.abs(x)))

  def get_config(self):
    return {'l2': float(self.l2)}

@tf.keras.utils.register_keras_serializable(package='Custom', name='rt2v1')
class TernaryEuclideanRegularizerV1(tf.keras.regularizers.Regularizer):
  """
  Based on arXiv:1812.11800v3 "Regularized Binary Network Training"
  This is polynomial in nature.
  """
  def __init__(self, l2=0., beta=1.):
    self.l2 = l2
    self.beta = beta

  def __call__(self, x):
    """
    R2(x) = beta*|1 - |x|||x|
    """
    return self.l2 * tf.math.reduce_sum(self.beta * tf.math.multiply(tf.math.abs(1-tf.math.abs(x)), tf.math.abs(x)))

  def get_config(self):
    return {'l2': float(self.l2), 'beta': float(self.beta)}

@tf.keras.utils.register_keras_serializable(package='Custom', name='rt2v2')
class TernaryEuclideanRegularizerV2(tf.keras.regularizers.Regularizer):
  """
  Based on arXiv:1812.11800v3 "Regularized Binary Network Training"
  This is ternary manhatten regularization with +- 1/2 fixed as the bad values.
  """
  def __init__(self, l2=0., beta=2.):
    self.l2 = l2
    self.beta = beta

  def __call__(self, x):
    """
    R2(x) = (beta * (|x| + |x-1| + |x+1| - |x-1/2| - |x+1/2| - 1))^2
    """
    x_norm = (x - tf.math.reduce_mean(x)) / (tf.math.reduce_std(x) + K.epsilon())
    regularized_x = tf.math.abs(x_norm) + tf.math.abs(x_norm - 1.0) + tf.math.abs(x_norm + 1.0) - tf.math.abs(x_norm - 0.5) - tf.math.abs(x_norm + 0.5) - 1.0
    regularized_x = tf.math.square(self.beta * regularized_x)
    return self.l2 * tf.math.reduce_sum(regularized_x)

  def get_config(self):
    return {'l2': float(self.l2), 'beta': float(self.beta)}

@tf.keras.utils.register_keras_serializable(package='Custom', name='rtt')
class RectifiedTernaryRegularizer(tf.keras.regularizers.Regularizer):
  """
  This is a rectified ternary regularizer
  """
  def __init__(self, l2=0., alpha=None):
    self.l2 = l2
    self.alpha = alpha

  def __call__(self, x):
    alpha = 2/3*tf.math.reduce_mean(tf.math.abs(x))
    mask_right = tf.cast(x >= alpha, K.floatx())
    mask_left = tf.cast(x <= -alpha, K.floatx())
    return self.l2 * tf.math.reduce_sum(mask_left * tf.square(x+1) + mask_right * tf.square(x-1))

  def get_config(self):
    return {'l2': float(self.l2), 'alpha': float(self.alpha)}

# @tf.keras.utils.register_keras_serializable(package='Custom', name='abtr')
class AdaptiveBinaryTernaryRegularizer(tf.Module):
  """
  Adapted from ArXiv:1909.12205v3 "Adaptive Binary-Ternary Quantization"
  """
  def __init__(self, lm=0., p=2, gamma=0.001, beta_init=(pi/2-1e-7), name=""):
    self.lm = lm
    self.p = p
    self.gamma = gamma
    self.beta = tf.Variable(
        initial_value=tf.random.uniform(shape=[], minval=pi/4, maxval=(pi/2-1e-7)),
        # initial_value=beta_init,
        trainable=True,
        constraint=lambda x: tf.clip_by_value(x, clip_value_min=(pi/4), clip_value_max=(pi/2-1e-7)),
        name=name+"/SQ/beta"
    )


  def __call__(self, x):
    # tf.print(self.beta)

    x = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-7)

    # Three terms in min
    t0 = tf.math.pow(tf.math.abs(tf.math.abs(x) + 1), self.p)
    t1= tf.math.pow(tf.math.abs(tf.math.abs(x) - 1), self.p)
    t2 = tf.math.tan(self.beta) * tf.math.pow(tf.math.abs(x), self.p)

    # Perform elementwise min
    out = tf.math.minimum(t0, t1)
    out = tf.math.minimum(out, t2)

    # Comput lambda
    lmbda = self.lm / (x.shape[0]*x.shape[1])

    out = lmbda * (tf.math.reduce_sum(out) + self.gamma * tf.math.abs(1 / (tf.math.tan(self.beta) + 1e-7)))

    return out

  def get_config(self):
    return {'lm': float(self.l2), "p": self.p, "gamma": self.gamma}

class DoubleSymmetricNormalInitializer(tf.keras.initializers.Initializer):

  def __init__(self, mean, stddev):
    self.mean = mean
    self.stddev = stddev

  def __call__(self, shape, dtype=None, **kwargs):
    return tf.random.normal(
        shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

  def get_config(self):  # To support serialization
    return {"mean": self.mean, "stddev": self.stddev}

class BreakpointLayerForDebug(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BreakpointLayerForDebug, self).__init__(**kwargs)

    def call(self, inputs):
        vele = "ferus" # place breakpoint here
        # inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        return inputs


class CenterLoss(tf.keras.losses.Loss):
    def __init__(self, ratio=0.0001, alpha=0.5, num_classes=1251, num_features=2048, one_hot=True, 
                from_logits=False,
                reduction=tf.keras.losses.Reduction.AUTO,
                name='center_loss'):
        super().__init__(reduction=reduction, name=name)
        self.ratio = ratio
        self.alpha = alpha
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.from_logits = from_logits
        self.centers = tf.Variable(
                initial_value=tf.constant(0, shape=[num_classes, num_features], dtype=tf.float32),
                trainable=False,
                name="centers"
            )

    @tf.function
    def call(self, y_true, y_pred):
        # print("y_pred shape = ", y_pred.get_shape())

        center_loss = self.center_loss_fn(
            features=y_pred, 
            labels=y_true, 
            alpha=self.alpha, num_classes=self.num_classes, one_hot=self.one_hot)

        center_loss = self.ratio * center_loss
        
        return center_loss


    # adapted from https://github.com/EncodeTS/TensorFlow_Center_Loss/blob/master/mnist_sample_code/mnist_with_center_loss.ipynb
    @tf.function
    def center_loss_fn(self, features, labels, alpha, num_classes, one_hot):
        # Init variables (if necessary) ---------------------------------------------------
        num_classes = labels.get_shape()[-1] if one_hot else num_classes

        # Calculate center loss -----------------------------------------------------------

        # Get sparse labels
        sparse_labels = tf.argmax(labels, axis=-1)
        
        # Get center vectors based on current label (indicies given by sparse labels)
        centers_batch = tf.gather(self.centers, sparse_labels)

        # Check for and convert NaNs to zeros
        features = tf.where(tf.math.is_nan(features), tf.zeros_like(features), features)

        # Calculate loss based on current centers (note division by two included)
        loss_diff = features - centers_batch
        loss = tf.nn.l2_loss(loss_diff)
        
        # Calculate centers updates -------------------------------------------------------

        # Get difference for grad (note it's the opposite since sign flips due to 
        # derivative)
        diff = centers_batch - features
        
        # Get count of each current label
        _, unique_idx, unique_count = tf.unique_with_counts(sparse_labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        
        # Get final derivative contribution from each feature row
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        
        # Update centers by subtracting row derivs from associated center row
        centers_update = tf.tensor_scatter_nd_sub(
            self.centers, 
            tf.reshape(sparse_labels, [-1, 1]), 
            diff)
        self.centers.assign(centers_update)
        
        return loss


class JointSoftmaxCenterLoss(tf.keras.losses.Loss):
    def __init__(self, ratio=5, alpha=0.5, num_classes=1251, one_hot=True, 
                from_logits=False,
                reduction=tf.keras.losses.Reduction.AUTO,
                name='joint_softmax_center_loss'):
        super().__init__(reduction=reduction, name=name)
        self.ratio = ratio
        self.alpha = alpha
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        print("y_pred shape = ", y_pred.get_shape())
        center_loss = self.center_loss_fn(
            features=y_pred[1], 
            labels=y_true, 
            alpha=self.alpha, num_classes=self.num_classes, one_hot=self.one_hot)

        softmax_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, 
            logits=y_pred[0])

        total_loss = softmax_loss + self.ratio * center_loss
        
        return total_loss


    # adapted from https://github.com/EncodeTS/TensorFlow_Center_Loss/blob/master/mnist_sample_code/mnist_with_center_loss.ipynb
    def center_loss_fn(self, features, labels, alpha, num_classes, one_hot):
        """center loss
        
        Arguments:
            features: Tensor, shape[batch_size, feature_length].
            labels: Tensor, label, one-hot, shape = [batch_size, num_classes].
            alpha: 0-1.
            num_classes.
        
        Return:
            loss: Tensor, center loss
        """

        # Init variables (if necessary) ---------------------------------------------------

        len_features = features.get_shape()[-1]
        print("Features shape = ", features.get_shape())
        print("Labels shape = ", labels.get_shape())
        num_classes = labels.get_shape()[-1] if one_hot else num_classes

        # Create centers variable if doesn't exist
        centers_shape = [num_classes, len_features]
        if not hasattr(self, "centers"):
            self.centers = tf.Variable(
                initial_value=tf.constant(0, shape=centers_shape, dtype=tf.float32),
                trainable=False,
                name="centers"
            )

        print(centers_shape)

        # Calculate center loss -----------------------------------------------------------

        # Get sparse labels
        sparse_labels = tf.argmax(labels, axis=-1)
        
        # Get center vectors based on current label (indicies given by sparse labels)
        centers_batch = tf.gather(self.centers, sparse_labels)

        # Calculate loss based on current centers (note division by two included)
        loss_diff = features - centers_batch
        loss = tf.nn.l2_loss(loss_diff)
        
        # Calculate centers updates -------------------------------------------------------

        # Get difference for grad (note it's the opposite since sign flips due to 
        # derivative)
        diff = centers_batch - features
        
        # Get count of each current label
        _, unique_idx, unique_count = tf.unique_with_counts(sparse_labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        
        # Get final derivative contribution from each feature row
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        
        # Update centers by subtracting row derivs from associated center row
        centers_update = tf.tensor_scatter_nd_sub(self.centers, sparse_labels, diff)
        self.centers.assign(centers_update)
        
        return loss

"""
The lr scheduler from "Attention is all you need"
"""
class AttentionLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, warmup_steps, output_depth):
        self.warmup_steps = warmup_steps
        self.output_depth = output_depth

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        one = 1 / tf.sqrt(tf.constant(self.output_depth, dtype=tf.float32))
        two = 1 / tf.sqrt(step + 1)
        three = (step + 1) / tf.sqrt(tf.pow(tf.constant(self.warmup_steps, dtype=tf.float32), tf.constant(3, dtype=tf.float32))) 
        cur_lr = one * tf.minimum(two, three)
        return cur_lr


class TimeReduction(tf.keras.layers.Layer):

    def __init__(self,
                 reduction_factor,
                 batch_size=None,
                 **kwargs):

        super(TimeReduction, self).__init__(**kwargs)

        self.reduction_factor = reduction_factor
        self.batch_size = batch_size

    def compute_output_shape(self, input_shape):
        max_time = input_shape[1]
        num_units = input_shape[2]
        if max_time != None: # For time variance
            extra_timestep = tf.math.floormod(max_time, self.reduction_factor)
            reduced_size = tf.math.floordiv(max_time, self.reduction_factor) + extra_timestep
        else: 
            reduced_size = None
        return [input_shape[0], reduced_size, num_units*self.reduction_factor]

    def call(self, inputs):

        input_shape = K.int_shape(inputs)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = input_shape[0]

        outputs = inputs

        if input_shape[1] != None:
            max_time = input_shape[1]
            extra_timestep = tf.math.floormod(max_time, self.reduction_factor)

            paddings = [[0, 0], [0, extra_timestep], [0, 0]]
            outputs = tf.pad(outputs, paddings)

        else:
            outputs = tf.signal.frame(outputs, self.reduction_factor, self.reduction_factor, pad_end=False, axis=1)

        # Necessary to let tf know correct output shape
        out_shape = self.compute_output_shape(input_shape)
        out_shape_tuple = tuple(-1 if s is None else s for s in out_shape)

        return tf.reshape(outputs, out_shape_tuple)

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_factor": self.reduction_factor,
            "batch_size": self.batch_size,
        })
        return config

def QSelfAttentionMechanismFn(num_hops, hidden_size, inputs, kernel_regularizer, bias_regularizer, kernel_quantizer, bias_quantizer, kernel_initializer, bias_initializer, activation, name):

    shp = K.int_shape(inputs)
    T = shp[1]
    n_h = shp[2]
    n_c = hidden_size
    n_k = num_hops

    #matmul1
    sa_matmul1 = QDense(
        n_c, activation=activation, use_bias=False, 
        kernel_regularizer=AdaptiveBinaryTernaryRegularizer(lm=1e-1, name=name+"_QDENSE_0"),
        bias_regularizer=bias_regularizer,
        kernel_quantizer=kernel_quantizer,
        bias_quantizer=bias_quantizer,
        name=name+"_QDENSE_0"
        )(inputs)

    # matmul2
    sa_matmul2 = QDense(
        n_k, activation=activation, use_bias=False, 
        kernel_regularizer=AdaptiveBinaryTernaryRegularizer(lm=1e-1, name=name+"_QDENSE_1"),
        bias_regularizer=bias_regularizer,
        kernel_quantizer=kernel_quantizer,
        bias_quantizer=bias_quantizer,
        name=name+"_QDENSE_1"
        )(sa_matmul1)

    # transpose
    sa_trans = tf.keras.layers.Lambda(
        lambda x: tf.transpose(x, perm=[0, 2, 1]),
        trainable=False,
        name=name+"_TRANSPOSE")(sa_matmul2)
    
    # matmul3
    sa_matmul3 = tf.keras.layers.Dot(axes=[2,1], name=name+"_DOT")([sa_trans, inputs])

    # sa_avg = tf.keras.layers.GlobalAveragePooling1D(name=name+"_TAP")(sa_matmul3)

    sa_output = tf.keras.layers.Reshape([n_h*n_k], name=name+"_OUTPUT")(sa_matmul3)
    # sa_output = tf.keras.layers.Reshape([n_h], name=name+"_OUTPUT")(sa_avg)

    return sa_output

def SelfAttentionMechanismFn(num_hops, hidden_size, inputs, kernel_regularizer, bias_regularizer, activation, name):

    shp = K.int_shape(inputs)
    T = shp[1]
    n_h = shp[2]
    n_c = hidden_size
    n_k = num_hops

    #matmul1
    sa_matmul1 = tf.keras.layers.Dense(
        n_c, activation=activation, use_bias=False, 
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        name=name+"_DENSE_0"
        )(inputs)
    
    # matmul2
    sa_matmul2 = tf.keras.layers.Dense(
        n_k, activation=activation, use_bias=False, 
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        name=name+"_DENSE_1"
        )(sa_matmul1)
    
    # transpose
    sa_trans = tf.keras.layers.Lambda(
        lambda x: tf.transpose(x, perm=[0, 2, 1]),
        trainable=False,
        name=name+"_TRANSPOSE")(sa_matmul2)
    
    # matmul3
    sa_matmul3 = tf.keras.layers.Dot(axes=[2,1], name=name+"_DOT")([sa_trans, inputs])

    # sa_avg = tf.keras.layers.GlobalAveragePooling1D(name=name+"_TAP")(sa_matmul3)

    sa_output = tf.keras.layers.Reshape([n_h*n_k], name=name+"_OUTPUT")(sa_matmul3)
    # sa_output = tf.keras.layers.Reshape([n_h], name=name+"_OUTPUT")(sa_avg)

    return sa_output

class SimpleRNNCellWithProjection(tf.keras.layers.SimpleRNNCell):
    def __init__(self, units, projection_units, **kwargs):
        super().__init__(units, **kwargs)
        self.projection_units = projection_units
        self.state_size = self.projection_units
        self.output_size = self.projection_units

    def build(self, input_shape):
        super().build(input_shape)
        self.recurrent_kernel = self.add_weight(
            shape=(self.projection_units, self.units),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.projection_kernel = self.add_weight(
            shape=(self.units, self.projection_units),
            name="projection_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )

    def call(self, inputs, states, training=None):
        prev_output = states[0] if tf.nest.is_nested(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training
        )

        # h = activation(inputs*kernel + prev_output*recurrent_kernel + bias)
        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        h = h + K.dot(prev_output, self.recurrent_kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)
        if self.activation is not None:
            h = self.activation(h)

        # r = h*projection_kernel
        output = K.dot(h, self.projection_kernel) # Projection layer

        new_state = [output] if tf.nest.is_nested(states) else output
        return output, new_state

    def get_config(self):
        return super().get_config().update({"projection_units": self.projection_units})
    

"""
# class SelfAttentionMechanism(tf.keras.layers.Layer):
#     def __init__(self,
#                  num_hops,
#                  hidden_size,
#                  **kwargs):

#         super(SelfAttentionMechanism, self).__init__(**kwargs)

#         self.num_hops = num_hops
#         self.hidden_size = hidden_size

#     def compute_output_shape(self, input_shape):
#         return [input_shape[0], self.num_hops, self.hidden_size]

#     def call(self, inputs):

#         # TODO: Switch out layers for actual variables and matmuls

#         shp = K.int_shape(inputs)
#         T = shp[1]
#         n_h = shp[2]
#         n_c = self.hidden_size
#         n_k = self.num_hops

#         #matmul1
#         sa_matmul1 = tf.keras.layers.Dense(n_c, activation="tanh", use_bias=False, name="matmul1_sa")(inputs)
        
#         # matmul2
#         sa_matmul2 = tf.keras.layers.Dense(n_k, activation="softmax", use_bias=False, name="matmul2_sa")(sa_matmul1)
        
#         # transpose
#         sa_trans = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]), name="transpose_sa")(sa_matmul2)
        
#         # matmul3
#         sa_matmul3 = tf.keras.layers.Dot(axes=[2,1], name="matmul3_sa")([sa_trans, inputs])

#         return sa_matmul3

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "num_hops": self.num_hops,
#             "hidden_size": self.hidden_size,
#         })
#         return config


# # https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
# class GlobalWeightedAveragePooling1D(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         tf.keras.layers.Layer.__init__(self, **kwargs)

#     def build(self, input_shape):
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], 1),
#                                       initializer='ones',
#                                       trainable=True)
#         tf.keras.layers.Layer.build(self, input_shape)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[2],

#     def call(self, x):

#         x = x*self.kernel
#         x = K.mean(x, axis=1)

#         return x


# # https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
# class GlobalWeightedMaxPooling1D(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         tf.keras.layers.Layer.__init__(self, **kwargs)

#     def build(self, input_shape):
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], 1),
#                                       initializer='ones',
#                                       trainable=True)
#         tf.keras.layers.Layer.build(self, input_shape)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[2],

#     def call(self, x):

#         x = x*self.kernel
#         x = K.max(x, axis=1)

#         return x


# # https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
# class GlobalWeightedAveragePooling2D(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         tf.keras.layers.Layer.__init__(self, **kwargs)

#     def build(self, input_shape):
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], input_shape[2], 1),
#                                       initializer='ones',
#                                       trainable=True)
#         tf.keras.layers.Layer.build(self, input_shape)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[3],

#     def call(self, x):
        
#         x = x*self.kernel
#         x = K.mean(x, axis=(1, 2))

#         return x


# # https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
# class GlobalWeightedMaxPooling2D(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         tf.keras.layers.Layer.__init__(self, **kwargs)

#     def build(self, input_shape):
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], input_shape[2], 1),
#                                       initializer='ones',
#                                       trainable=True)
#         tf.keras.layers.Layer.build(self, input_shape)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[3],

#     def call(self, x):

#         x = x*self.kernel
#         x = K.max(x, axis=(1, 2))

#         return x

# # https://github.com/linhdvu14/vggvox-speaker-identification/blob/master/model.py
# # Block of layers: Conv --> BatchNorm --> ReLU --> Pool
# def conv_bn_pool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,pool='',pool_size=(2, 2),pool_strides=None,
#     conv_layer_prefix='conv'):
# 	x = tf.keras.layers.ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
# 	x = tf.keras.layers.Conv2D(
#         filters=conv_filters,
#         kernel_size=conv_kernel_size, 
#         strides=conv_strides, 
#         padding='valid', 
#         name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
# 	x = tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
# 	x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
# 	if pool == 'max':
# 		x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,strides=pool_strides,name='mpool{}'.format(layer_idx))(x)
# 	elif pool == 'avg':
# 		x = tf.keras.layers.AveragePooling2D(pool_size=pool_size,strides=pool_strides,name='apool{}'.format(layer_idx))(x)
# 	return x


# # https://github.com/linhdvu14/vggvox-speaker-identification/blob/master/model.py
# # Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
# def conv_bn_dynamic_apool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,conv_layer_prefix='conv'):
# 	x = tf.keras.layers.ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
# 	x = tf.keras.layers.Conv2D(
#         filters=conv_filters,
#         kernel_size=conv_kernel_size, 
#         strides=conv_strides, 
#         padding='valid', 
#         name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
# 	x = tf.keras.layers.BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
# 	x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
# 	x = tf.keras.layers.GlobalAveragePooling2D(name='gapool{}'.format(layer_idx))(x)
# 	x = tf.keras.layers.Reshape((1,1,conv_filters),name='reshape{}'.format(layer_idx))(x)
# 	return x
"""
