import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers, losses
import tensorflow.keras.backend as K
import math


class CenterLoss(losses.Loss):
    def __init__(self, ratio=0.0001, alpha=0.5, num_classes=1251, one_hot=True, 
                from_logits=False,
                reduction=losses.Reduction.AUTO,
                name='center_loss'):
        super().__init__(reduction=reduction, name=name)
        self.ratio = ratio
        self.alpha = alpha
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # print("y_pred shape = ", y_pred.get_shape())
        center_loss = self.center_loss_fn(
            features=y_pred, 
            labels=y_true, 
            alpha=self.alpha, num_classes=self.num_classes, one_hot=self.one_hot)

        center_loss = self.ratio * center_loss
        
        return center_loss


    # adapted from https://github.com/EncodeTS/TensorFlow_Center_Loss/blob/master/mnist_sample_code/mnist_with_center_loss.ipynb
    def center_loss_fn(self, features, labels, alpha, num_classes, one_hot):
        # Init variables (if necessary) ---------------------------------------------------

        len_features = features.get_shape()[-1]
        # print("Features shape = ", features.get_shape())
        # print("Labels shape = ", labels.get_shape())
        num_classes = labels.get_shape()[-1] if one_hot else num_classes

        # Create centers variable if doesn't exist
        centers_shape = [num_classes, len_features]
        if not hasattr(self, "centers"):
            self.centers = tf.Variable(
                initial_value=tf.constant(0, shape=centers_shape, dtype=tf.float32),
                trainable=False,
                name="centers"
            )

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
        centers_update = tf.tensor_scatter_nd_sub(
            self.centers, 
            tf.reshape(sparse_labels, [-1, 1]), 
            diff)
        self.centers.assign(centers_update)
        
        return loss


class JointSoftmaxCenterLoss(losses.Loss):
    def __init__(self, ratio=5, alpha=0.5, num_classes=1251, one_hot=True, 
                from_logits=False,
                reduction=losses.Reduction.AUTO,
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
class AttentionLRScheduler(optimizers.schedules.LearningRateSchedule):

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


def SelfAttentionMechanismFn(num_hops, hidden_size, inputs, name):

    shp = K.int_shape(inputs)
    T = shp[1]
    n_h = shp[2]
    n_c = hidden_size
    n_k = num_hops

    #matmul1
    sa_matmul1 = layers.Dense(
        n_c, activation="tanh", use_bias=False, 
        kernel_regularizer=regularizers.L2(l2=0.0001),
        name=name+"_DENSE_0"
        )(inputs)
    
    # matmul2
    sa_matmul2 = layers.Dense(
        n_k, activation="sigmoid", use_bias=False, 
        kernel_regularizer=regularizers.L2(l2=0.0001),
        name=name+"_DENSE_1"
        )(sa_matmul1)
    
    # transpose
    sa_trans = layers.Lambda(
        lambda x: tf.transpose(x, perm=[0, 2, 1]),
        trainable=False,
        name=name+"_TRANSPOSE")(sa_matmul2)
    
    # matmul3
    sa_matmul3 = layers.Dot(axes=[2,1], name=name+"_DOT")([sa_trans, inputs])

    sa_output = layers.Reshape([n_h*n_k], name=name+"_OUTPUT")(sa_matmul3)

    return sa_output


# class SelfAttentionMechanism(layers.Layer):
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
#         sa_matmul1 = layers.Dense(n_c, activation="tanh", use_bias=False, name="matmul1_sa")(inputs)
        
#         # matmul2
#         sa_matmul2 = layers.Dense(n_k, activation="softmax", use_bias=False, name="matmul2_sa")(sa_matmul1)
        
#         # transpose
#         sa_trans = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]), name="transpose_sa")(sa_matmul2)
        
#         # matmul3
#         sa_matmul3 = layers.Dot(axes=[2,1], name="matmul3_sa")([sa_trans, inputs])

#         return sa_matmul3

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "num_hops": self.num_hops,
#             "hidden_size": self.hidden_size,
#         })
#         return config


# # https://github.com/csvance/keras-global-weighted-pooling/blob/master/gwp.py
# class GlobalWeightedAveragePooling1D(layers.Layer):
#     def __init__(self, **kwargs):
#         layers.Layer.__init__(self, **kwargs)

#     def build(self, input_shape):
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], 1),
#                                       initializer='ones',
#                                       trainable=True)
#         layers.Layer.build