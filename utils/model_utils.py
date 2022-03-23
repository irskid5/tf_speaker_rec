import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K


class TimeReduction(layers.Layer):

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
        extra_timestep = tf.math.floormod(max_time, self.reduction_factor)
        reduced_size = tf.math.floordiv(max_time, self.reduction_factor) + extra_timestep
        return [input_shape[0], reduced_size, num_units*self.reduction_factor]

    def call(self, inputs):

        input_shape = K.int_shape(inputs)

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = input_shape[0]

        max_time = input_shape[1]
        extra_timestep = tf.math.floormod(max_time, self.reduction_factor)

        outputs = inputs

        paddings = [[0, 0], [0, extra_timestep], [0, 0]]
        outputs = tf.pad(outputs, paddings)

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


# https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
class GlobalWeightedAveragePooling1D(layers.Layer):
    def __init__(self, **kwargs):
        layers.Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='ones',
                                      trainable=True)
        layers.Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2],

    def call(self, x):

        x = x*self.kernel
        x = K.mean(x, axis=1)

        return x


# https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
class GlobalWeightedMaxPooling1D(layers.Layer):
    def __init__(self, **kwargs):
        layers.Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='ones',
                                      trainable=True)
        layers.Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2],

    def call(self, x):

        x = x*self.kernel
        x = K.max(x, axis=1)

        return x


# https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
class GlobalWeightedAveragePooling2D(layers.Layer):
    def __init__(self, **kwargs):
        layers.Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2], 1),
                                      initializer='ones',
                                      trainable=True)
        layers.Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3],

    def call(self, x):
        
        x = x*self.kernel
        x = K.mean(x, axis=(1, 2))

        return x


# https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
class GlobalWeightedMaxPooling2D(layers.Layer):
    def __init__(self, **kwargs):
        layers.Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], input_shape[2], 1),
                                      initializer='ones',
                                      trainable=True)
        layers.Layer.build(self, input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3],

    def call(self, x):

        x = x*self.kernel
        x = K.max(x, axis=(1, 2))

        return x

# https://github.com/linhdvu14/vggvox-speaker-identification/blob/master/model.py
# Block of layers: Conv --> BatchNorm --> ReLU --> Pool
def conv_bn_pool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,pool='',pool_size=(2, 2),pool_strides=None,
    conv_layer_prefix='conv'):
	x = layers.ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
	x = layers.Conv2D(
        filters=conv_filters,
        kernel_size=conv_kernel_size, 
        strides=conv_strides, 
        padding='valid', 
        name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
	x = layers.BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
	x = layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
	if pool == 'max':
		x = layers.MaxPooling2D(pool_size=pool_size,strides=pool_strides,name='mpool{}'.format(layer_idx))(x)
	elif pool == 'avg':
		x = layers.AveragePooling2D(pool_size=pool_size,strides=pool_strides,name='apool{}'.format(layer_idx))(x)
	return x


# https://github.com/linhdvu14/vggvox-speaker-identification/blob/master/model.py
# Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
def conv_bn_dynamic_apool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,conv_layer_prefix='conv'):
	x = layers.ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
	x = layers.Conv2D(
        filters=conv_filters,
        kernel_size=conv_kernel_size, 
        strides=conv_strides, 
        padding='valid', 
        name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
	x = layers.BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
	x = layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
	x = layers.GlobalAveragePooling2D(name='gapool{}'.format(layer_idx))(x)
	x = layers.Reshape((1,1,conv_filters),name='reshape{}'.format(layer_idx))(x)
	return x