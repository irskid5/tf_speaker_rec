from math import pi
import tensorflow as tf
import tensorflow_addons as tfa
import keras.backend as K
from keras.engine import data_adapter
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
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer=None,
        kernel_norm=None,
        recurrent_norm=None,
        use_bias=False,
        add_dist_loss=False,
        fold_batch_norm=False,
        soft_thresh_tern=False,
        add_no_acc_reg=False,
        no_acc_reg_lm=0,
        no_acc_reg_bits=32,
        s=1,
        unroll=False,
        name="", **kwargs):
        
        # Which init are we using?
        k_init = kernel_initializer # 'glorot_uniform'
        rk_init = recurrent_initializer # 'orthogonal'
        # k_init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.001) if not kernel_initializer else kernel_initializer
        # rk_init = tf.keras.initializers.Identity(gain=identity_scale) if not recurrent_initializer else recurrent_initializer

        # Normalizers
        if kernel_norm == "batch_norm":
            self.kernel_norm = tf.keras.layers.BatchNormalization()
        elif kernel_norm == "layer_norm":
            self.kernel_norm = tf.keras.layers.LayerNormalization()
        else:
            self.kernel_norm = None

        if recurrent_norm == "batch_norm":
            self.recurrent_norm = tf.keras.layers.BatchNormalization()
        elif recurrent_norm == "layer_norm":
            self.recurrent_norm = tf.keras.layers.LayerNormalization()
        else:
            self.recurrent_norm = None

        self.add_dist_loss = add_dist_loss
        self.fold_batch_norm = fold_batch_norm
        self.soft_thresh_tern = soft_thresh_tern
        self.add_no_acc_reg = add_no_acc_reg
        self.no_acc_reg_lm = no_acc_reg_lm
        self.no_acc_reg_bits = no_acc_reg_bits
        self.s = s

        # These are flags that require rnn unrolling
        to_unroll = unroll or add_dist_loss or add_no_acc_reg
        
        cell = QSimpleRNNCellWithNorm(
                    units,
                    activation=activation,
                    kernel_initializer=k_init,
                    recurrent_initializer=rk_init,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_quantizer=kernel_quantizer,
                    recurrent_quantizer=recurrent_quantizer,
                    bias_quantizer=bias_quantizer,
                    kernel_norm=self.kernel_norm,
                    recurrent_norm=self.recurrent_norm,
                    use_bias=use_bias,
                    add_dist_loss=add_dist_loss,
                    fold_batch_norm=fold_batch_norm,
                    soft_thresh_tern=soft_thresh_tern,
                    add_no_acc_reg=add_no_acc_reg,
                    no_acc_reg_lm=no_acc_reg_lm,
                    no_acc_reg_bits=no_acc_reg_bits,
                    s=s,
                    name=name,
                ) if not cell else cell 
            
        super(QIRNN, self).__init__(cell, return_sequences=True, stateful=stateful, unroll=to_unroll, name=name)
        
        # IRNN Specific
        self.identity_scale = identity_scale
        self.stacking_number = stacking_number
        
        # Initializers
        self.kernel_initializer = k_init
        self.recurrent_initializer = rk_init

        # Quantizers (specfically for QNoiseScheduler)
        self.quantizers = self.get_quantizers()

    # def build(self, input_shape):
    #     super(QIRNN, self).build(input_shape)
    #     if self.kernel_norm:
    #         self.kernel_norm.build(input_shape)
    #     if self.recurrent_norm:
    #         self.recurrent_norm.build(input_shape)

    def get_quantizers(self):
        return self.cell.quantizers

    def get_prunable_weights(self):
        return [self.cell.kernel, self.cell.recurrent_kernel]

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def kernel_quantizer_internal(self):
        return self.cell.kernel_quantizer_internal

    @property
    def recurrent_quantizer_internal(self):
        return self.cell.recurrent_quantizer_internal

    @property
    def bias_quantizer_internal(self):
        return self.cell.bias_quantizer_internal

    @property
    def state_quantizer_internal(self):
        return self.cell.state_quantizer_internal

    @property
    def kernel_quantizer(self):
        return self.cell.kernel_quantizer

    @property
    def recurrent_quantizer(self):
        return self.cell.recurrent_quantizer

    @property
    def bias_quantizer(self):
        return self.cell.bias_quantizer

    @property
    def state_quantizer(self):
        return self.cell.state_quantizer

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = super().get_config()
        config.update({
            "identity_scale": self.identity_scale,
            "stacking_number": self.stacking_number,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
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

class QSimpleRNNCellWithNorm(QSimpleRNNCell):
    """
    Cell class for the QSimpleRNNCell layer with added Norm function for Wx.
    """
    def __init__(self,
                units,
                activation='quantized_tanh',
                use_bias=False,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                kernel_quantizer=None,
                recurrent_quantizer=None,
                bias_quantizer=None,
                state_quantizer=None,
                dropout=0.,
                recurrent_dropout=0.,
                kernel_norm=None,
                recurrent_norm=None,
                add_dist_loss=False,
                fold_batch_norm=False,
                soft_thresh_tern=False,
                add_no_acc_reg=False,
                no_acc_reg_lm=0,
                no_acc_reg_bits=32,
                s=1,
                **kwargs):

        super(QSimpleRNNCellWithNorm, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            kernel_quantizer=kernel_quantizer,
            recurrent_quantizer=recurrent_quantizer,
            bias_quantizer=bias_quantizer,
            state_quantizer=state_quantizer,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            **kwargs
        )
        
        self.kernel_norm = kernel_norm
        self.recurrent_norm = recurrent_norm
        
        self.fold_batch_norm = fold_batch_norm

        # If folding, freeze BN
        if self.kernel_norm is not None:
            if self.fold_batch_norm:
                self.kernel_norm.trainable = False
        if self.recurrent_norm is not None:
            if self.fold_batch_norm:
                self.recurrent_norm.trainable = False

        # Add distribution loss
        self.dist_loss = None
        if add_dist_loss:
            self.dist_loss = DistributionLossLayer()
        
        # Soft Threshold Ternarization
        self.soft_thresh_tern = soft_thresh_tern

        # No Accumulation Activity Regularization (must unroll rnn)
        self.no_acc_reg = None
        self.no_acc_reg_lm = no_acc_reg_lm
        self.no_acc_reg_bits = no_acc_reg_bits
        if add_no_acc_reg:
            self.no_acc_reg = NoAccRegularizerV2(lm=no_acc_reg_lm, k=2**no_acc_reg_bits, name=kwargs["name"])

        # Gradient scale
        self.s = s
  
    def build(self, input_shape):
        super(QSimpleRNNCellWithNorm, self).build(input_shape)
        if self.kernel_norm is not None:
            norm_input_shape = list(input_shape)
            norm_input_shape[-1] = self.kernel.shape[-1] # since norm is applied to Wx, not x
            self.kernel_norm.build(norm_input_shape)
        if self.recurrent_norm is not None:
            norm_input_shape = list(input_shape)
            norm_input_shape[-1] = self.recurrent_kernel.shape[-1] # since norm is applied to Wx, not x
            self.recurrent_norm.build(norm_input_shape)
        
        # For debugging to see quantized kernels
        if self.kernel_quantizer:
            self.quantized_kernel = self.add_weight(
                name="quantized_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False
            )
        if self.recurrent_quantizer:
            self.quantized_recurrent_kernel = self.add_weight(
                name="quantized_recurrent_kernel",
                shape=self.recurrent_kernel.shape,
                dtype=self.recurrent_kernel.dtype,
                initializer="zeros",
                trainable=False
            )
        if self.kernel_norm is not None:
            if self.fold_batch_norm:
                self.folded_kernel = self.add_weight(
                    name="folded_kernel",
                    shape=self.kernel.shape,
                    dtype=self.kernel.dtype,
                    initializer="zeros",
                    trainable=False
                )
            self.delta_kernel_norm = self.add_weight(
                name="delta_kernel_norm",
                shape=[self.units],
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=True
            )
        if self.recurrent_norm is not None:
            if self.fold_batch_norm:
                self.folded_recurrent_kernel = self.add_weight(
                    name="folded_recurrent_kernel",
                    shape=self.recurrent_kernel.shape,
                    dtype=self.recurrent_kernel.dtype,
                    initializer="zeros",
                    trainable=False
                )
            self.delta_recurrent_kernel_norm = self.add_weight(
                name="delta_recurrent_kernel_norm",
                shape=[self.units],
                dtype=self.recurrent_kernel.dtype,
                initializer="zeros",
                trainable=True
            )

        self.wx = self.add_weight(
            name="wx",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False
        )

        self.wh = self.add_weight(
            name="wh",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False
        )

        if self.soft_thresh_tern:
            self.kernel_2 = self.add_weight(
                name="kernel_2",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            self.kernel_2.assign(tf.random.shuffle(self.kernel_2))
            self.recurrent_kernel_2 = self.add_weight(
                name="recurrent_kernel_2",
                shape=self.recurrent_kernel.shape,
                dtype=self.recurrent_kernel.dtype,
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                trainable=True
            )
            self.recurrent_kernel_2.assign(tf.random.shuffle(self.recurrent_kernel_2))
            if self.kernel_quantizer:
                self.quantized_kernel_2 = self.add_weight(
                    name="quantized_kernel_2",
                    shape=self.kernel.shape,
                    dtype=self.kernel.dtype,
                    initializer="zeros",
                    trainable=False
                )
            if self.recurrent_quantizer:
                self.quantized_recurrent_kernel_2 = self.add_weight(
                    name="quantized_recurrent_kernel_2",
                    shape=self.recurrent_kernel.shape,
                    dtype=self.recurrent_kernel.dtype,
                    initializer="zeros",
                    trainable=False
                )
            self.combined_kernel = self.add_weight(
                name="combined_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False
            )
            self.combined_recurrent_kernel = self.add_weight(
                name="combined_recurrent_kernel",
                shape=self.recurrent_kernel.shape,
                dtype=self.recurrent_kernel.dtype,
                initializer="zeros",
                trainable=False
            )

    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_sequence(states) else states

        # Dropout mask
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)

        # Quantize the state
        if self.state_quantizer:
            quantized_prev_output = self.state_quantizer_internal(prev_output)
        else:
            quantized_prev_output = prev_output

        # Fold frozen BN into weights (this won't work with self.soft_thresh_tern on)
        folded_kernel = self.kernel
        if self.kernel_norm is not None:
            if self.fold_batch_norm:
                folded_kernel = folded_kernel * self.kernel_norm.gamma / tf.math.sqrt(self.kernel_norm.moving_variance + self.kernel_norm.epsilon)
                # For debugging to see folded kernel
                self.folded_kernel.assign(folded_kernel)
        folded_recurrent_kernel = self.recurrent_kernel
        if self.recurrent_norm is not None:
            if self.fold_batch_norm:
                folded_recurrent_kernel = folded_recurrent_kernel * self.recurrent_norm.gamma / tf.math.sqrt(self.recurrent_norm.moving_variance + self.recurrent_norm.epsilon)
                # For debugging to see folded kernel
                self.folded_recurrent_kernel.assign(folded_recurrent_kernel)

        # Quantize the kernel(s)
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(folded_kernel)
            # For debugging to see quantized kernel
            self.quantized_kernel.assign(quantized_kernel)
            
            # Quantize the second kernel for soft threshold ternarization
            if self.soft_thresh_tern:
                quantized_kernel_2 = self.kernel_quantizer_internal(self.kernel_2)
                self.quantized_kernel_2.assign(quantized_kernel_2)
                quantized_kernel -= quantized_kernel_2
                self.combined_kernel.assign(quantized_kernel)
        else:
            quantized_kernel = folded_kernel
            
            # Combine both kernels for soft threshold ternarization
            if self.soft_thresh_tern:
                quantized_kernel_2 = self.kernel_2
                quantized_kernel -= quantized_kernel_2
                self.combined_kernel.assign(quantized_kernel)

        # Add dropout mask if not none
        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, quantized_kernel)
        else:
            h = K.dot(inputs, quantized_kernel)
        if rec_dp_mask is not None:
            quantized_prev_output = quantized_prev_output * rec_dp_mask

        # Quantize recurrent kernel
        if self.recurrent_quantizer:
            quantized_recurrent = self.recurrent_quantizer_internal(folded_recurrent_kernel)
            # For debugging to see quantized kernel
            self.quantized_recurrent_kernel.assign(quantized_recurrent)

            # Quantize the second kernel for soft threshold ternarization and combine both
            if self.soft_thresh_tern:
                quantized_recurrent_2 = self.recurrent_quantizer_internal(self.recurrent_kernel_2)
                self.quantized_recurrent_kernel_2.assign(quantized_recurrent_2)
                quantized_recurrent -= quantized_recurrent_2
                self.combined_recurrent_kernel.assign(quantized_recurrent)
        else:
            quantized_recurrent = folded_recurrent_kernel
            
            # Combine both kernels for soft threshold ternarization
            if self.soft_thresh_tern:
                quantized_recurrent_2 = self.recurrent_kernel_2
                quantized_recurrent -= quantized_recurrent_2
                self.combined_recurrent_kernel.assign(quantized_recurrent)

        # Calculate h_2
        h_2 = K.dot(quantized_prev_output, quantized_recurrent)

        # Update stats
        self.wx.assign(0.99*self.wx + 0.01*tf.reduce_mean(h, axis=0))
        self.wh.assign(0.99*self.wh + 0.01*tf.reduce_mean(h_2, axis=0))

        # Compute norm if applicable
        h_norm = h
        h_2_norm = h_2
        if self.kernel_norm:
            if not self.fold_batch_norm:
                # h_norm = self.kernel_norm(h)
                # Record the diff between normed and not normed 
                # self.delta_kernel_norm.assign(0.99*self.delta_kernel_norm + 0.01*(tf.reduce_mean(h, axis=0) - tf.reduce_mean(h_norm, axis=0)))
                h_norm = h - self.delta_kernel_norm + tf.stop_gradient(self.delta_kernel_norm-2000*self.delta_kernel_norm)
        if self.recurrent_norm:
            if not self.fold_batch_norm:
                # h_2_norm = self.recurrent_norm(h_2)
                # Record the diff between normed and not normed 
                # self.delta_recurrent_kernel_norm.assign(0.99*self.delta_recurrent_kernel_norm + 0.01*(tf.reduce_mean(h_2, axis=0) - tf.reduce_mean(h_2_norm, axis=0)))
                h_2_norm = h_2 - self.delta_recurrent_kernel_norm + tf.stop_gradient(self.delta_recurrent_kernel_norm-2000*self.delta_recurrent_kernel_norm)

        # Add two dot products (W_x*x + W_h*h)
        # Division is to control the gradient but output is h_norm + h_2_norm
        s = self.s # 1000 worked for quant
        output = h_norm/s + h_2_norm/s + tf.stop_gradient(-h_norm/s - h_2_norm/s + h_norm + h_2_norm)

        # Add bias if applicable 
        if self.bias is not None:
            # Fold the bias if folding BN
            # TODO
            # Quantize the bias
            if self.bias_quantizer:
                quantized_bias = self.bias_quantizer_internal(self.bias)
            else:
                quantized_bias = self.bias

            output = K.bias_add(output, quantized_bias)
        else:   
            # Add the beta terms from both norms, or either norm
            if (self.kernel_norm is not None) or (self.recurrent_norm is not None):
                if self.fold_batch_norm:
                    # Add folded beta to output if folding batch norm
                    folded_kernel_bias = 0
                    folded_recurrent_bias = 0
                    if self.kernel_norm is not None:
                            folded_kernel_bias = self.kernel_norm.beta - self.kernel_norm.moving_mean * self.kernel_norm.gamma / tf.math.sqrt(self.kernel_norm.moving_variance + self.kernel_norm.epsilon)
                    if self.recurrent_norm is not None:
                            folded_recurrent_bias = self.recurrent_norm.beta - self.recurrent_norm.moving_mean * self.recurrent_norm.gamma / tf.math.sqrt(self.recurrent_norm.moving_variance + self.recurrent_norm.epsilon)

                    # Combine folded biases
                    folded_bias = folded_kernel_bias + folded_recurrent_bias
                    
                    if self.bias_quantizer:
                        quantized_bias = self.bias_quantizer_internal(folded_bias)
                    else:
                        quantized_bias = folded_bias

                    output = K.bias_add(output, quantized_bias)

        # Finally compute the activation
        if self.activation is not None:
            # No acc regularizer
            if self.no_acc_reg is not None:
                output = self.no_acc_reg(output)
            # Add distribution loss if applicable
            if self.dist_loss is not None:
                output = self.dist_loss(output)
            output = self.activation(output)

        return output, [output]

class QDenseWithNorm(QDense):
    """Implements a quantized Dense layer WITH NORMALIZATION."""

    def __init__(self,
                units,
                activation=None,
                use_bias=True,
                kernel_initializer="he_normal",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                kernel_quantizer=None,
                bias_quantizer=None,
                kernel_range=None,
                bias_range=None,
                norm=None,
                add_dist_loss=False,
                fold_batch_norm=False,
                soft_thresh_tern=False,
                add_no_acc_reg=False,
                no_acc_reg_lm=0,
                no_acc_reg_bits=32,
                s=1,
                **kwargs):

        self.norm = None
        if norm == "batch_norm":
            self.norm = tf.keras.layers.BatchNormalization()
        
        self.dist_loss = None
        if add_dist_loss:
            self.dist_loss = DistributionLossLayer()

        self.fold_batch_norm = fold_batch_norm

        # If folding, freeze BN
        if self.norm is not None:
            if self.fold_batch_norm:
                self.norm.trainable = False

        # Soft Threshold Ternarization
        self.soft_thresh_tern = soft_thresh_tern

        # No Accumulation Activity Regularization
        self.no_acc_reg = None
        self.no_acc_reg_lm = no_acc_reg_lm
        self.no_acc_reg_bits = no_acc_reg_bits
        if add_no_acc_reg:
            self.no_acc_reg = NoAccRegularizerV2(lm=no_acc_reg_lm, k=2**no_acc_reg_bits, name=kwargs["name"])

        # Gradient scale
        self.s = s

        super(QDenseWithNorm, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            kernel_quantizer=kernel_quantizer,
            bias_quantizer=bias_quantizer,
            kernel_range=kernel_range,
            bias_range=bias_range,
            **kwargs)

    def build(self, input_shape):
        super(QDenseWithNorm, self).build(input_shape)
        if self.norm is not None:
            norm_input_shape = input_shape.as_list() 
            norm_input_shape[-1]= self.kernel.shape[-1] # since norm is applied to Wx, not x
            self.norm.build(norm_input_shape)
        
        # For debugging to see quantized kernel
        if self.kernel_quantizer:
            self.quantized_kernel = self.add_weight(
                name="quantized_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False
            )

        if self.soft_thresh_tern:
            self.kernel_2 = self.add_weight(
                name="kernel_2",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )
            self.kernel_2.assign(tf.random.shuffle(self.kernel_2))

            if self.kernel_quantizer:
                self.quantized_kernel_2 = self.add_weight(
                    name="quantized_kernel_2",
                    shape=self.kernel.shape,
                    dtype=self.kernel.dtype,
                    initializer="zeros",
                    trainable=False
                )
            self.combined_kernel = self.add_weight(
                name="combined_kernel",
                shape=self.kernel.shape,
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=False
            )
        
        if self.norm is not None:
            if self.fold_batch_norm:
                self.folded_kernel = self.add_weight(
                    name="folded_kernel",
                    shape=self.kernel.shape,
                    dtype=self.kernel.dtype,
                    initializer="zeros",
                    trainable=False
                )
            self.delta_kernel_norm = self.add_weight(
                name="delta_kernel_norm",
                shape=[self.units],
                dtype=self.kernel.dtype,
                initializer="zeros",
                trainable=True
            )

        self.wx = self.add_weight(
            name="wx",
            shape=[self.units],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False
        )

        self.x_input = self.add_weight(
            name="x",
            shape=[input_shape[-1]],
            dtype=self.kernel.dtype,
            initializer="zeros",
            trainable=False
        )

    
    def call(self, inputs):
        # Write input stats (mainly for SA mult)
        reduce_dims = tf.range(0, tf.rank(inputs)-1)
        self.x_input.assign(0.99*self.x_input + 0.01*tf.reduce_mean(inputs, axis=reduce_dims)) 
        
        # Fold frozen BN into weights
        folded_kernel = self.kernel
        if self.norm is not None:
            if self.fold_batch_norm:
                folded_kernel = folded_kernel * self.norm.gamma / tf.math.sqrt(self.norm.moving_variance + self.norm.epsilon)
                # For debugging to see folded kernel
                self.folded_kernel.assign(folded_kernel)

        # Quantize the kernel
        if self.kernel_quantizer:
            quantized_kernel = self.kernel_quantizer_internal(folded_kernel)
            # For debugging to see quantized kernel
            self.quantized_kernel.assign(quantized_kernel)

            # Quantize the second kernel for soft threshold ternarization
            if self.soft_thresh_tern:
                quantized_kernel_2 = self.kernel_quantizer_internal(self.kernel_2)
                self.quantized_kernel_2.assign(quantized_kernel_2)
                quantized_kernel -= quantized_kernel_2
                self.combined_kernel.assign(quantized_kernel)
        else:
            quantized_kernel = folded_kernel

            # Combine the kernels for soft threshold ternarization
            if self.soft_thresh_tern:
                quantized_kernel_2 = self.kernel_2
                quantized_kernel -= quantized_kernel_2
                self.combined_kernel.assign(quantized_kernel)

        # Calculate Wx
        h = tf.keras.backend.dot(inputs, quantized_kernel)

        # Scale gradients
        s = self.s # 1000 worked for quant
        h = h/s + tf.stop_gradient(-h/s + h)

        # Update stats
        reduce_dims = tf.range(0, tf.rank(h)-1)
        self.wx.assign(0.99*self.wx + 0.01*tf.reduce_mean(h, axis=reduce_dims)) 
        
        output = h
        # Add bias if using bias
        if self.use_bias:
            # Fold the bias if folding BN
            folded_bias = self.bias
            if self.norm is not None:
                if self.fold_batch_norm:
                    folded_bias = self.norm.gamma * (folded_bias - self.norm.moving_mean) / tf.math.sqrt(self.norm.moving_variance + self.norm.epsilon) + self.norm.beta
            
            # Quantize bias
            if self.bias_quantizer:
                quantized_bias = self.bias_quantizer_internal(folded_bias)
            else:
                quantized_bias = folded_bias

            quantized_bias = quantized_bias + tf.stop_gradient(-quantized_bias + 10000*quantized_bias)
            
            # Add bias to output (Wx + b)
            output = tf.keras.backend.bias_add(output, quantized_bias,
                                            data_format="channels_last")
        else:   
            # Add folded beta to output if folding batch norm
            if self.norm is not None:
                if self.fold_batch_norm:
                    folded_bias = self.norm.beta - self.norm.moving_mean * self.norm.gamma / tf.math.sqrt(self.norm.moving_variance + self.norm.epsilon)
                    # Quantize bias
                    if self.bias_quantizer:
                        quantized_bias = self.bias_quantizer_internal(folded_bias)
                    else:
                        quantized_bias = folded_bias
                    
                    # Add bias to output (Wx + b)
                    output = tf.keras.backend.bias_add(output, quantized_bias,
                                                    data_format="channels_last")

        # Normalize if no batch norm folding
        if self.norm is not None:
            if not self.fold_batch_norm:
                output = self.norm(output)
                # Record the diff between normed and not normed 
                # reduceDims = tf.range(0, tf.rank(h)-1)
                # self.delta_kernel_norm.assign(0.99*self.delta_kernel_norm + 0.01*(tf.reduce_mean(h, axis=reduceDims) - tf.reduce_mean(output, axis=reduceDims)))
                # output = output - self.delta_kernel_norm + tf.stop_gradient(self.delta_kernel_norm-2000*self.delta_kernel_norm)

        if self.activation is not None:
            # No acc regularizer
            if self.no_acc_reg is not None:
                output = self.no_acc_reg(output)
            # Add distribution loss
            if self.dist_loss is not None:
                output = self.dist_loss(output)
            # Compute activation
            output = self.activation(output)
        return output

def sign_with_ste(x):
    """
    Compute the signum function in the fwd pass but return STE approximation for grad in bkwd pass
    """
    out = x
    q = tf.math.sign(x)
    q += (1.0 - tf.math.abs(q))
    return out + tf.stop_gradient(-out + q)

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

def sign_with_tanh_deriv(x):
    out = tf.keras.activations.tanh(x)
    q = tf.math.sign(x)
    q += (1.0 - tf.math.abs(q))
    return out + tf.stop_gradient(-out + q)

def sign_with_htanh_deriv(x):
    out = hard_tanh(x)
    q = tf.math.sign(x)
    q += (1.0 - tf.math.abs(q))
    return out + tf.stop_gradient(-out + q)

def sign_swish(x, beta=1):
    """
    Adapted from arXiv:1812.11800v3 "Regularized Binary Network Training"
    """
    beta_x = beta*x
    temped_sigmoid = tf.sigmoid(beta_x)
    out = 2*temped_sigmoid*(1+beta_x*(1-temped_sigmoid))-1
    return out

def sign_with_ss_deriv(x):
    """
    Compute the signum function in the fwd pass but return STE approximation for grad in bkwd pass
    """
    out = sign_swish(x, beta=1)
    q = tf.math.sign(x)
    q += (1.0 - tf.math.abs(q))
    return out + tf.stop_gradient(-out + q)

def custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=8):
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

    out = sign_with_tanh_deriv(x) + tf.stop_gradient(-sign_with_tanh_deriv(x) + sign_with_tanh_deriv(_inner_fn(x, num_bits=num_bits)))

    # For seeing distribution of input to activation
    # (Note: The try must be kept or else it will fail on the initial tf graphing)
    # try:
    #     plot_histogram_discrete(x, "histogram_of_wxplusb.png")
    #     plot_histogram_discrete(signed_float, "histogram_of_wxplusb_modded.png")
    # except:
    #     return out
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

class GeneralActivation(tf.keras.layers.Layer):
    """
    Adapted from ArXiv:1805.06085 "PACT: Parameterized Clipping Activation for Quantized Neural Networks"
    """
    def __init__(self, beta_ss=10, regularizer=None, normalizer=None, activation=None, add_dist_loss=True, name=""):
        super(GeneralActivation, self).__init__()
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.activation = activation
        self.dist_loss = None 
        
        if add_dist_loss:
            self.dist_loss = DistributionLossLayer() 
        """
        self.alpha = self.add_weight(
            name=name+"/alpha", 
            shape=[], 
            dtype=tf.float32, 
            initializer=tf.keras.initializers.constant(self.alpha_init), 
            regularizer=self.regularizer,
            trainable=True,
        )
        self.inp_moving_mean = self.add_weight(
            name=name+"/inp_moving_mean", 
            shape=[], 
            dtype=tf.float32, 
            initializer=tf.keras.initializers.constant(0), 
            trainable=False,
        )
        self.inp_moving_std = self.add_weight(
            name=name+"/inp_moving_std", 
            shape=[], 
            dtype=tf.float32, 
            initializer=tf.keras.initializers.constant(0), 
            trainable=False,
        )
        self.out_moving_mean = self.add_weight(
            name=name+"/out_moving_mean", 
            shape=[], 
            dtype=tf.float32, 
            initializer=tf.keras.initializers.constant(0), 
            trainable=False,
        )
        self.out_moving_std = self.add_weight(
            name=name+"/out_moving_std", 
            shape=[], 
            dtype=tf.float32, 
            initializer=tf.keras.initializers.constant(0), 
            trainable=False,
        )
        """
        
        # Get normalizer
        if self.normalizer == "batch_norm":
            self.norm = tf.keras.layers.BatchNormalization()
        elif self.normalizer == "layer_norm":
            self.norm = tf.keras.layers.LayerNormalization()
        else:
            self.norm = None

        # self.beta_ss = self.add_weight(
        #     name=name+"/beta_ss",
        #     shape=[],
        #     dtype=tf.float32,
        #     initializer=tf.keras.initializers.Constant(beta_ss),
        #     trainable=False,
        # )

        self.built = False


    def build(self, input_shape):
        super(GeneralActivation, self).build(input_shape)
        
        # Build normalizer
        if self.norm:
            self.norm.build(input_shape)
        
        new_shape = [128, input_shape[-1]]

        # Build shape dependent stats
        # self.input_dist = self.add_weight(
        #     name=self.name+"/inputs_to_act", 
        #     shape=new_shape, 
        #     dtype=tf.float32, 
        #     initializer="zeros", 
        #     trainable=False,
        # )
        # self.output_dist = self.add_weight(
        #     name=self.name+"/outputs_from_act", 
        #     shape=new_shape, 
        #     dtype=tf.float32, 
        #     initializer="zeros", 
        #     trainable=False,
        # )

        self.built = True

    def __call__(self, inputs):
        input_shape = inputs.get_shape()
        size_shape = len(input_shape.as_list())
        if not self.built:
            self.build(input_shape)
        
        # self.beta_ss.assign(self.beta_ss)

        # Calc act
        # pact = 1/2 * (tf.abs(inputs) - tf.abs(inputs-self.alpha) + self.alpha)

        # Normalize
        out = inputs
        if self.norm:
            out = self.norm(out) # self.layer_norm(inputs)

        # Run activation
        if self.activation:
            # Calculate Distribution Loss from ArXiv:1904.02823
            # if self.dist_loss:
            #     out = self.dist_loss(out)
            out = self.activation(out)

        # Compute stats
        # inputs_for_stats = tf.zeros_like(inputs) + inputs
        # outputs_for_stats = tf.zeros_like(out) + out
        # if (size_shape == 3):
        #     inputs_for_stats = tf.reduce_mean(inputs_for_stats, 1)
        #     outputs_for_stats = tf.reduce_mean(outputs_for_stats, 1)

        # # Calculate input stats
        # self.input_dist.assign(0.01 * inputs_for_stats + 0.99 * self.input_dist)
        
        # # Calculate output stats
        # self.output_dist.assign(0.01 * outputs_for_stats + 0.99 * self.output_dist)

        return out

    def get_config(self):
        return { 
            "alpha_init": self.alpha_init, 
        }

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

@tf.keras.utils.register_keras_serializable(package='Custom', name='var_reg')
class VarianceRegularizer(tf.keras.regularizers.Regularizer):
  """
  This is regularizer that punishes weights with variance other than one.
  """
  def __init__(self, l2=0.):
    self.l2 = l2

  def __call__(self, x):
    out = tf.square(x)
    out = tf.reduce_sum(out)
    out = tf.math.divide(1.0, tf.size(x, out_type=tf.float32)) * out
    out = 1 - out
    out = tf.square(out)
    return self.l2 * out

  def get_config(self):
    return {'l2': float(self.l2)}

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

    # x = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-7)

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
    return {'lm': float(self.lm), "p": self.p, "gamma": self.gamma}

@tf.function
def no_acc_reg_trig_fn(t, b):
    b2 = b**2
    cos_t = tf.cos(t)
    num = 1 + b2
    den = 1 + b2*tf.square(cos_t)
    
    return -1/2*tf.sqrt(num/den)*cos_t

class NoAccRegularizerV1(tf.keras.layers.Layer):
  """
  This regularizer penalizes results of matrix multiplications that lie in bad regions of the domain
  after a modulo operation is applied.

  Ex. if K (modulus) = 8, then a result of Wx_i mod 8 = 4 mod 8 = -4 instead of +4 which can lead a 
  sign activation function outputting wrong -1 instead of +1. This is important for binary/ternary
  networks.
  """
  def __init__(self, lm=0.001, k=128, a=3/8., name=""):
    super(NoAccRegularizerV1, self).__init__()
    self.lm = lm
    self.k = k
    self.a = a
    self.b = a * k
    self.g_0 = tf.abs(no_acc_reg_trig_fn(2*pi/k * (-k/4 + 1/2), self.b))

  
  def __call__(self, x):
    # Map inputs
    t_x = 2*pi/self.k * (tf.abs(x) - self.k/4 + 1/2)

    # Run function
    loss = tf.nn.relu(no_acc_reg_trig_fn(t_x, self.b) + self.g_0)

    self.add_loss(self.lm * tf.reduce_sum(loss))
    return x

  def get_config(self):
    return {"lm": float(self.lm), "k": float(self.k), "a": float(self.a), "b": float(self.b), "g_0": float(self.g_0)}

@tf.function
def no_acc_reg_hat_fn(x, k, a):
    t_x = 1/k*(tf.abs(x)-(3/4*k-1/2))
    mod = tf.math.mod(t_x, 1)
    abs = tf.abs(2*mod-1)
    out = 2*abs-1
    return a*tf.nn.relu(out)

@tf.function
def no_acc_reg_hat_metric_fn(x, k, a):
    """
    A function that calculates the ratio between values in correct regions and all values.
    To be used as input into metric function (ex. tf.keras.metrics.Mean()).

    Takes the sign of the no_acc_reg function which gives you whether a value is in a wrong
    region (+1) or correct region (0). Adding all the values in the tensor and dividing by the
    total number of values is effectively the count of wrong values over all values. This is 
    also more easily represented as the mean of the tensor. 1 - mean(wrongs) gives the ratio
    of correct values over all values.

    k is the modulus of quantization.
    a is the height of no_acc_reg_hat_fn.

    Important: this implementation only to be used with no_acc_reg_hat_fn!
    """
    wrongs = tf.sign(no_acc_reg_hat_fn(x, k=k, a=a))
    rights_ratio = 1 - tf.reduce_mean(wrongs, axis=[-1])
    return rights_ratio

class NoAccRegularizerV2(tf.keras.layers.Layer):
  """
  This regularizer penalizes results of matrix multiplications that lie in incorrectly-signed regions of the domain
  after a modulo operation is applied.

  Ex. if K (modulus) = 8, then a result of Wx_i mod 8 = 4 mod 8 = -4 instead of +4 which can lead a 
  sign activation function outputting wrong -1 instead of +1. This is important for binary/ternary
  networks.

  This specific version implements a hat function (ex. __/\__/\__). This is supposed to provide a constant
  derivative in the incorrect regions to move them to the correct regions.
  """
  def __init__(self, lm=1e-3, k=2**8, a=1., name=""):
    super(NoAccRegularizerV2, self).__init__()
    self.lm = lm
    self.k = k
    self.a = a
    self.no_acc_metric = tf.keras.metrics.Mean(name='no_acc_metric/'+name)

  def __call__(self, x):
    loss = tf.square(no_acc_reg_hat_fn(x=x, k=self.k, a=self.a))
    accuracy = no_acc_reg_hat_metric_fn(x, k=self.k, a=self.a)

    self.add_loss(self.lm * tf.reduce_sum(loss))
    self.add_metric(self.no_acc_metric(accuracy))

    return x

  def get_config(self):
    return {'lm': float(self.lm), 'k': int(self.k), 'a': float(self.a)}

class TernaryNormalInitializer(tf.keras.initializers.Initializer):

  def __init__(self, stddev=1/60):
    self.stddev = stddev

  def __call__(self, shape, dtype=None, **kwargs):
    n0 = tf.random.normal(
        shape, mean=0, stddev=self.stddev, dtype=dtype)
    n1 = tf.random.normal(
        shape, mean=-1, stddev=self.stddev, dtype=dtype)
    n2 = tf.random.normal(
        shape, mean=1, stddev=self.stddev, dtype=dtype)
    pick = tf.random.uniform(shape, minval=0, maxval=3, dtype=tf.int32)
    out0 = tf.where(tf.equal(pick, 0), n0, tf.zeros(shape))
    out1 = tf.where(tf.equal(pick, 1), n1, tf.zeros(shape))
    out2 = tf.where(tf.equal(pick, 2), n2, tf.zeros(shape))
    return tf.stop_gradient(out0 + out1 + out2)

  def get_config(self):  # To support serialization
    return {"stddev": self.stddev}

class BreakpointLayerForDebug(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BreakpointLayerForDebug, self).__init__(**kwargs)

    def call(self, inputs):
        vele = "ferus" # place breakpoint here
        # inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        return inputs

@tf.function
def distribution_loss(pre_acts, size_shape, k_d=1, k_s=0.25, k_m=0.25):
    
    mean = None
    std = None
    if size_shape == 3:
        mean = tf.math.reduce_mean(pre_acts, [1, 2])
        std = tf.math.reduce_std(pre_acts, [1, 2])
    if size_shape == 2:
        mean = tf.math.reduce_mean(pre_acts, [-1])
        std = tf.math.reduce_std(pre_acts, [-1])
    
    # Calculate absolute value of mean
    mean_abs = tf.math.abs(mean)

    l_d = tf.math.square(tf.keras.activations.relu(mean_abs - k_d*std))
    l_s = tf.math.square(tf.keras.activations.relu(k_s*std - 1))
    l_m = tf.math.square(tf.keras.activations.relu(1 - mean_abs - k_m*std))

    loss = l_d + 0*l_s + 0*l_m # CHANGE BACK

    return loss


class DistributionLossLayer(tf.keras.layers.Layer):
    """
    Adapted from ArXiv:1904.02823 "Regularizing Activation Distribution for Training Binarized Deep Networks"
    """
    def __init__(self, rate=0.001, k_d=0., k_s=0., k_m=0.): # k_d=1, k_s=0.25, k_m=0.25 are default
        super(DistributionLossLayer, self).__init__()
        self.rate = rate
        self.k_d = k_d
        self.k_s = k_s
        self.k_m = k_m

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(distribution_loss(inputs, len(inputs.get_shape()), self.k_d, self.k_s, self.k_m)))
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

def QSelfAttentionMechanismFn(num_hops, hidden_size, inputs, kernel_regularizer, bias_regularizer, kernel_quantizer, bias_quantizer, kernel_initializer, bias_initializer, activation, use_bias, norm, fold_batch_norm, add_dist_loss, soft_thresh_tern, learned_thresh, add_no_acc_reg, no_acc_reg_lm, no_acc_reg_bits, s, name):

    shp = K.int_shape(inputs)
    T = shp[1]
    n_h = shp[2]
    n_c = hidden_size
    n_k = num_hops

    #matmul1
    sa_matmul1 = QDenseWithNorm(
        n_c, activation=GeneralActivation(activation=activation, name=name+"_QDENSE_0"), 
        use_bias=use_bias, 
        kernel_regularizer=kernel_regularizer, # if kernel_regularizer else AdaptiveBinaryTernaryRegularizer(lm=1e-1, name=name+"_QDENSE_0"),
        bias_regularizer=bias_regularizer,
        kernel_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.02, name=name+"_QDENSE_0") if learned_thresh else kernel_quantizer, # 0.0032
        bias_quantizer=bias_quantizer,
        kernel_initializer=kernel_initializer,
        norm=norm,
        fold_batch_norm=fold_batch_norm,
        add_dist_loss=add_dist_loss, # add_dist_loss, CHANGE BACK
        soft_thresh_tern=soft_thresh_tern,
        add_no_acc_reg=add_no_acc_reg,
        no_acc_reg_lm=no_acc_reg_lm,
        no_acc_reg_bits=no_acc_reg_bits,
        s=s,
        name=name+"_QDENSE_0")(inputs)

    # matmul2
    sa_matmul2 = QDenseWithNorm(
        n_k, activation=GeneralActivation(activation=activation, name=name+"_QDENSE_1"),  
        use_bias=use_bias, 
        kernel_regularizer=kernel_regularizer, # if kernel_regularizer else AdaptiveBinaryTernaryRegularizer(lm=1e-1, name=name+"_QDENSE_1"),
        bias_regularizer=bias_regularizer,
        kernel_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.02, name=name+"_QDENSE_1") if learned_thresh else kernel_quantizer, # 0.00055
        bias_quantizer=bias_quantizer,
        kernel_initializer=kernel_initializer,
        norm=norm,
        fold_batch_norm=fold_batch_norm,
        add_dist_loss=add_dist_loss, # add_dist_loss, CHANGE BACK
        soft_thresh_tern=soft_thresh_tern,
        add_no_acc_reg=add_no_acc_reg,
        no_acc_reg_lm=no_acc_reg_lm,
        no_acc_reg_bits=no_acc_reg_bits,
        s=s,
        name=name+"_QDENSE_1")(sa_matmul1)

    # transpose
    sa_trans = tf.keras.layers.Lambda(
        lambda x: tf.transpose(x, perm=[0, 2, 1]),
        trainable=False,
        name=name+"_TRANSPOSE")(sa_matmul2)
    
    # matmul3
    sa_matmul3 = tf.keras.layers.Dot(axes=[2,1], name=name+"_DOT")([sa_trans, inputs])

    # ADDED THIS TO SEE HOW IT WOULD WORK
    if add_no_acc_reg:
        sa_matmul3 = NoAccRegularizerV2(lm=no_acc_reg_lm, k=2**no_acc_reg_bits, name=name+"_DOT")(sa_matmul3)
    sa_matmul3 = GeneralActivation(activation=activation, name=name+"_act_after_dot")(sa_matmul3)

    # sa_avg = tf.keras.layers.GlobalAveragePooling1D(name=name+"_TAP")(sa_matmul3)

    sa_output = tf.keras.layers.Reshape([n_h*n_k], name=name+"_OUTPUT")(sa_matmul3)

    # bias_correction = tf.Variable(
    #     initial_value="zeros",
    #     trainable=True,
    #     shape=[n_h*n_k],
    # )
    # sa_output = tf.keras.layers.Add()([sa_output, bias_correction])
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

 
class CustomModelWithGradHistograms(tf.keras.models.Model):
    def train_step(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happens in fit](
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.

        This method should contain the mathematical logic for one step of
        training.  This typically includes the forward pass, loss calculation,
        backpropagation, and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function`
        and `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

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
