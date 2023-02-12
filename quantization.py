import tensorflow as tf
import keras.backend as K
import tensorflow_model_optimization as tfmot
from qkeras import *


from tensorflow_model_optimization.python.core.quantization.keras.quantizers import *

def log_quantize_layer_lognet(x, num_bits=4, max=10, min=-10, hparams=None):
    """
    The way lognet does logarithmic quantization
    """
    sign = tf.sign(x)
    # delta_lsb = (max-min) / 2**num_bits
    fsr = tf.experimental.numpy.log2(tf.constant(max - min, dtype=tf.float32))
    logged = tf.experimental.numpy.log2(tf.math.abs(x))
    rounded = tf.round(logged)
    clipped = tf.clip_by_value(rounded, fsr - 2**num_bits, fsr - 1)
    out = sign * 2**clipped
    # quantized = tf.cast(quantized, tf.float32)
    return out

def log_quantize_layer_vele(x, num_bits=4, max=1, min=-1, hparams=None):
    """
    Vele's version of log quant (in newer book notes)
    """
    sign = tf.sign(x)
    # delta_lsb = (max-min) / 2**num_bits
    # fsr = tf.experimental.numpy.log2(tf.constant(max - min, dtype=tf.float32))
    logged = tf.experimental.numpy.log10(tf.math.abs(x))
    rounded = logged + tf.stop_gradient(-logged + tf.round(logged))
    bias = tf.reduce_mean(rounded)
    bias = bias + tf.stop_gradient(-bias + tf.round(bias))
    translated = rounded + tf.abs(bias)
    out = translated + tf.stop_gradient(-translated + tf.clip_by_value(translated, 0, max))
    out = sign * out
    # quantized = tf.cast(quantized, tf.float32)
    return out

def binarize_tensor(x, tau=1, hparams=None):
    """
    Computes the sign(x) function with 0 -> 1
    """
    quantized = tf.math.sign(x)
    quantized = tf.where(tf.equal(quantized, 0), tf.ones_like(quantized), quantized)
    return quantized

def ternarize_tensor(x, tau=1, hparams=None):
    """
    Quantize tensor into {-1, 0, 1}, but first multiplying it by tau and rounding.
    """
    rounded = tf.round(x*tau)
    quantized = tf.math.sign(rounded)
    return quantized

def binarize_tensor_with_threshold(x, theta=1, hparams=None):
    """
    Ternary quantizer where 
    x = -1 if x <= -threshold,
    x = 1  if -threshold < x < threshold,
    x = 1  if x > -threshold
    """
    quantized = tf.where(tf.less(x, -theta), -tf.ones_like(x), x)
    quantized = tf.where(tf.less_equal(tf.math.abs(quantized), theta), tf.ones_like(quantized), quantized)
    quantized = tf.where(tf.greater(quantized, theta), tf.ones_like(quantized), quantized)
    return quantized

def ternarize_tensor_with_threshold(x, theta=1, mean=0., hparams=None):
    """
    Ternary quantizer where 
    x = -1 if x-mean <= -threshold,
    x = 0  if -threshold < x-mean < threshold,
    x = 1  if x-mean >= threshold.

    Includes mean correction whici is 0 by default.
    """
    q = K.cast(tf.abs(x-mean) >= theta, K.floatx()) * tf.sign(x)
    # q = K.cast(tf.abs(x) >= theta, K.floatx()) * x
    return q

def twn_ternarize_with_threshold(x, hparams=None):
    """
    Adapted from ArXiv:1605.04711v2 "Ternary Weight Networks"
    """
    thres = 0.7 * tf.reduce_mean(tf.abs(x))
    return ternarize_tensor_with_threshold(x, theta=thres)

def symmetric_qauntize_tensor(x, num_bits=4, max_range=1e-8, hparams=None):
    """
    Symmetric quantizer where Xq = round( (2**(num_bits-1)-1) / max|x| * x )
    """
    max = tf.reduce_max(tf.abs(x))
    max = tf.where(max > max_range, max, max_range)
    q = (pow(2, num_bits-1)-1) / (max + 1e-8)
    quantized = tf.round(q*x)
    return quantized

def quantize_over_range(x, num_bits=4):
    """
    Asymmetric quantizer where Xq = round( (2**(num_bits-1)-1) / (max(x) - min(x)) * x )
    """
    range = tf.reduce_max(x) - tf.reduce_min(x)
    s = range / (2**(num_bits-1)-1)
    quantized = tf.round(x / s)
    return quantized

class LearnedThresholdTernary(BaseQuantizer):
    def __init__(
        self, 
        scale=0.7, 
        threshold=None,
        mean=0.0,
        qnoise_factor=1.0,
        var_name=None,
        use_ste=True,
        use_variables=False, 
        name=""
        ):

        super(LearnedThresholdTernary, self).__init__()
        self.bits = 2
        self.threshold = threshold
        self.mean = mean
        # self.threshold = tf.Variable(
        #     initial_value=tf.constant(0, dtype=tf.float32) if threshold is None else threshold,
        #     trainable=False,
        #     name=name+"/tern_threshold"
        # )
        self.scale = scale
        self.initialized = False
        self.qnoise_factor = qnoise_factor
        self.use_ste = use_ste
        self.var_name = var_name
        self.use_variables = use_variables
    
    def __call__(self, x):
        if not self.built:
            self.build(var_name=self.var_name, use_variables=self.use_variables)
            # self.threshold.assign(0.7 * tf.reduce_mean(tf.abs(x)))
            # sorted = tf.sort(tf.reshape(tf.abs(x), shape=[-1]))
            # t = sorted[int(np.ceil(0.33*len(sorted)))]
            # self.threshold.assign(self.scale * t)
            self.initialized = True
        
        xq = ternarize_tensor_with_threshold(x, theta=self.threshold, mean=self.mean)

        if self.use_ste:
            return x + tf.stop_gradient(self.qnoise_factor * (-x + xq))
            # out = tf.math.tanh(x)
            # return out + tf.stop_gradient(-out + xq)
        else:
            return (1 - self.qnoise_factor) * x + tf.stop_gradient(
                self.qnoise_factor * xq)

    def max(self):
        """Get the maximum value that ternary can respresent."""
        return 1.0

    def min(self):
        """Get the minimum value that ternary can respresent."""
        return -1.0

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "scale": self.scale,
            "threshold": self.threshold,
        }
        return config

class CustomLogarithmic(BaseQuantizer):
    def __init__(
        self,
        max=1.0,
        min=-1.0, 
        scale=0.7, 
        qnoise_factor=1.0,
        var_name=None,
        use_ste=False,
        use_variables=False, 
        name=""
        ):

        super(CustomLogarithmic, self).__init__()
        self.bits = 4
        self.maximum = max
        self.minimum = min
        self.scale = scale
        self.initialized = False
        self.qnoise_factor = qnoise_factor
        self.use_ste = use_ste
        self.var_name = var_name
        self.use_variables = use_variables
    
    def __call__(self, x):
        if not self.built:
            # self.build(var_name=self.var_name, use_variables=self.use_variables)
            self.initialized = True
        
        xq = log_quantize_layer_vele(x, max=self.maximum, min=self.minimum)

        return xq

    def max(self):
        """Get the maximum value that ternary can respresent."""
        return self.maximum + tf.abs(self.minimum)

    def min(self):
        """Get the minimum value that ternary can respresent."""
        return -self.maximum - tf.abs(self.minimum)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            "scale": self.scale,
        }
        return config

class BinaryQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
  """Quantizer which forces outputs to be -1 or 1."""

  def build(self, tensor_shape, name, layer):
    # Not needed. No new TensorFlow variables needed.
    return {}

  def __call__(self, inputs, training, weights, **kwargs):
    return binarize_tensor(inputs)

  def get_config(self):
    # Not needed. No __init__ parameters to serialize.
    return {}

class TernaryQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
  """Quantizer which forces outputs to be -1, 0, or 1."""

  def build(self, tensor_shape, name, layer):
    # Not needed. No new TensorFlow variables needed.
    return {}

  def __call__(self, inputs, training, weights, **kwargs):
    return ternarize_tensor(inputs)

  def get_config(self):
    # Not needed. No __init__ parameters to serialize.
    return {}


class CustomQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    """
    Custom quantization config for our main model for QAT
    """
    # List all of your weights
    weights = {
        # "kernel": AllValuesQuantizer(num_bits=4, per_axis=False, symmetric=True, narrow_range=True),
        # "recurrent_kernel": AllValuesQuantizer(num_bits=4, per_axis=False, symmetric=True, narrow_range=True),
        # "bias": AllValuesQuantizer(num_bits=4, per_axis=False, symmetric=False, narrow_range=True),
        "kernel": BinaryQuantizer(),
        "recurrent_kernel": BinaryQuantizer()
    }

    # List of all your activations
    activations = {
        # "activation": AllValuesQuantizer(num_bits=4, per_axis=False, symmetric=True, narrow_range=True)
        # "activation": BinaryQuantizer()
    }

    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        output = []
        
        for attribute, quantizer in self.weights.items():
            if hasattr(layer, attribute) and (getattr(layer, attribute) is not None):
                output.append((getattr(layer, attribute), quantizer))
        
        # For IRNN 
        if hasattr(layer, "cell") and (getattr(layer, "cell") is not None):
            for cell in layer.cell.cells.layers:
                for attribute, quantizer in self.weights.items():
                    if hasattr(cell, attribute) and (getattr(cell, attribute) is not None):
                        output.append((getattr(cell, attribute), quantizer))

        return output

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        output = []
        for attribute, quantizer in self.activations.items():
            if hasattr(layer, attribute) and (getattr(layer, attribute) is not None):
                output.append((getattr(layer, attribute), quantizer))
        
        # For IRNN 
        if hasattr(layer, "cell") and (getattr(layer, "cell") is not None):
            for cell in layer.cell.cells.layers:
                for attribute, quantizer in self.activations.items():
                    if hasattr(cell, attribute) and (getattr(cell, attribute) is not None):
                        output.append((getattr(cell, attribute), quantizer))

        return output

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order

        count = 0
        for attribute in self.weights.keys():
            if hasattr(layer, attribute) and (getattr(layer, attribute) is not None):
                setattr(layer, attribute, quantize_weights[count])
                count += 1
        
        # For IRNN
        if hasattr(layer, "cell") and (getattr(layer, "cell") is not None):
            for cell in layer.cell.cells.layers:
                for attribute in self.weights.keys():
                    if hasattr(cell, attribute) and (getattr(cell, attribute) is not None):
                        setattr(cell, attribute, quantize_weights[count])
                        count += 1

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        count = 0
        for attribute in self.activations.keys():
            if hasattr(layer, attribute) and (getattr(layer, attribute) is not None):
                setattr(layer, attribute, quantize_activations[count])
                count += 1
        
        # For IRNn
        if hasattr(layer, "cell") and (getattr(layer, "cell") is not None):
            for cell in layer.cell.cells.layers:
                for attribute in self.activations.keys():
                    if hasattr(cell, attribute) and (getattr(cell, attribute) is not None):
                        setattr(cell, attribute, quantize_activations[count])
                        count += 1
        
    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}