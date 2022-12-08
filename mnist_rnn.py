import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit(True)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from qkeras import *

from utils.model_utils import TimeReduction, sign_with_tanh_deriv, QDenseWithNorm, QIRNN
from quantization import ternarize_tensor_with_threshold, LearnedThresholdTernary

SEED = 1997
RUNS_DIR = "/home/vele/Documents/masters/mnist_rnn/runs/"
TB_LOGS_DIR = "logs/tensorboard/"
CKPT_DIR = "checkpoints/"
BACKUP_DIR = "tmp/backup"
RECORD_CKPTS = True

now = datetime.now()
RUN_DIR = RUNS_DIR + now.strftime("%Y%m") + "/" + now.strftime("%Y%m%d-%H%M%S") + "/"

# Make sure we don't get any GPU errors
# Helper functions
def configure_environment(gpu_names, fp16_run=False, multi_strategy=False):
    # Set dtype
    dtype = tf.float32

    # ----------------- TO IMPLEMENT -------------------------
    if fp16_run:
        print('Using 16-bit float precision.')
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        # dtype = tf.float16
    # --------------------------------------------------------

    # Get GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpu_names is not None and len(gpu_names) > 0:
        gpus = [x for x in gpus if x.name[len('/physical_device:'):] in gpu_names]

    # Init gpus
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            logging.warning(str(e))

    # Set how many GPUs you want running
    # gpus = gpus[0:2]

    # Set training strategy (mirrored for multi, single for one, depending on availability
    # and multi_strategy flag)
    if multi_strategy and len(gpus) > 1:
        gpu_names = [x.name[len('/physical_device:'):] for x in gpus]
        print('Running multi gpu: {}'.format(', '.join(gpu_names)))
        strategy = tf.distribute.MirroredStrategy(
            devices=gpu_names)
        num_workers = len(gpus)
    else:
        device = gpus[0].name[len('/physical_device:'):]
        print('Running single gpu: {}'.format(device))
        strategy = tf.distribute.OneDeviceStrategy(
            device=device)
        num_workers = 1

    return strategy, dtype, num_workers

strategy, _, num_workers = configure_environment(None, False, True)

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


def augment(image, label):
    # if tf.random.uniform((), minval=0, maxval=1) < 0.1:
    #     image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64*num_workers

# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)

# Activation functions
# activation_irnn = tf.keras.activations.tanh
# activation_dense = tf.keras.activations.tanh

# Activation fns (quantized)
activation_irnn = sign_with_tanh_deriv
activation_dense = sign_with_tanh_deriv

# Regularizers (None for quantization I believe)
kernel_regularizer = None # NoAccRegularizer(0.001, k=256) # tf.keras.regularizers.L2(0.0001) # tf.keras.regularizers.L1(l1=1e-5) # VarianceRegularizer(l2=2.0) # tf.keras.regularizers.L2(0.0001) # TernaryEuclideanRegularizer(l2=0.00001, beta=4) # tf.keras.regularizers.L2(0.0001)
recurrent_regularizer = None # NoAccRegularizer(0.001, k=256) # tf.keras.regularizers.L2(0.0001) # tf.keras.regularizers.L1(l1=1e-5) # VarianceRegularizer(l2=2.0) # tf.keras.regularizers.L2(0.0001) # BinaryEuclideanRegularizer(l2=1e-6) #TernaryEuclideanRegularizer(l2=0.00001, beta=4) # tf.keras.regularizers.L2(0.0001) # TernaryEuclideanRegularizer(l2=0.00001, beta=4)
bias_regularizer = None
activation_regularizer = None # tf.keras.regularizers.L2(l2=0.0001)

# Quantization functions (General)
kernel_quantizer = ternary(alpha=1, threshold=0.1) # binary(alpha=0.5) # stochastic_ternary(alpha=1, threshold=0.01) # ternary(alpha=1, threshold=0.1) # quantized_bits(bits=4, integer=0, symmetric=1, keep_negative=True, alpha=1.0) # ternary(alpha=1, threshold=lambda x: 0.7*tf.reduce_mean(tf.abs(x))) # quantized_bits(bits=2, integer=2, symmetric=1, keep_negative=True)
recurrent_quantizer = ternary(alpha=1, threshold=0.1) # binary(alpha=0.5) # stochastic_ternary(alpha=1, threshold=0.02) # ternary(alpha=1, threshold=0.1) # quantized_bits(bits=4, integer=0, symmetric=1, keep_negative=True, alpha=1.0) # ternary(alpha=1, threshold=lambda x: 0.08) # quantized_bits(bits=2, integer=2, symmetric=1, keep_negative=True)
bias_quantizer = None # ternary(alpha=1) # quantized_bits(bits=8, integer=8, symmetric=1, keep_negative=True)

# Optional
soft_thresh_tern = False
learned_thresh = False
tern = True

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
                kernel_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.1, name="QRNN_0/quantized_kernel") if learned_thresh else kernel_quantizer,
                recurrent_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.05, name="QRNN_0/quantized_recurrent") if learned_thresh else recurrent_quantizer,
                kernel_initializer=rnn_kernel_initializer,
                recurrent_initializer=rnn_recurrent_initializer,
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
                kernel_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.045, name="QRNN_1/quantized_kernel") if learned_thresh else kernel_quantizer,
                recurrent_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.06, name="QRNN_1/quantized_recurrent") if learned_thresh else recurrent_quantizer,
                kernel_initializer=rnn_kernel_initializer,
                recurrent_initializer=rnn_recurrent_initializer,
                name="QRNN_1"),
            tf.keras.layers.Flatten(),
            QDenseWithNorm(
                1024, 
                activation=activation_dense, 
                use_bias=False,
                kernel_regularizer=kernel_regularizer, 
                kernel_quantizer=LearnedThresholdTernary(scale=1.0, name="DENSE_0/quantized_kernel") if learned_thresh else kernel_quantizer,
                kernel_initializer=dense_kernel_initializer,
                name="DENSE_0",),
            QDenseWithNorm(
                10, 
                use_bias=False,
                activation=activation_dense, 
                kernel_regularizer=kernel_regularizer, 
                kernel_quantizer=LearnedThresholdTernary(scale=1.0, name="DENSE_OUT/quantized_kernel") if learned_thresh else kernel_quantizer,
                kernel_initializer=dense_kernel_initializer,
                name="DENSE_OUT",),
        ]
    )

    model.summary()

    return model

# Specify checkpoint dir
checkpoint_dir = None
if checkpoint_dir:
    run_dir = checkpoint_dir[0:-12]

# Specify location of pretrained weights
pretrained_weights = None

# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-134607/checkpoints/" # Original, tanh, default init, Adam(1e-4), L2(1e-4), 99%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-140942/checkpoints/" # Original, sign_with_tanh, 20221208-134607 init, Adam(1e-5, clipnorm=1), 98%
pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-150638/checkpoints/" # tern input, sign_with_tanh, 20221208-140942 init, Adam(1e-5, clipnorm=1), 99%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-095905/checkpoints/" # tern input, sign_with_tanh, 20221208-150638 init, Adam(1e-5, clipnorm=1), 

with strategy.scope():
    model = get_model()

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        logging.info('Restored pretrained weights from {}.'.format(pretrained_weights))

        # Load weights if we starting from checkpoint
    if checkpoint_dir is not None:
        model.load_weights(checkpoint_dir)
        logging.info('Restored checkpoint weights from {}.'.format(checkpoint_dir))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

# Define callbacks
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=(RUN_DIR + TB_LOGS_DIR), histogram_freq=1, update_freq="epoch", # profile_batch='100,200'
)

if RECORD_CKPTS:
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(RUN_DIR + CKPT_DIR),
                                                save_weights_only=True,
                                                save_best_only=True,
                                                monitor="val_accuracy",
                                                mode="max",
                                                verbose=1)
else:
    ckpt_callback = None

model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_test,
    callbacks=[tb_callback, ckpt_callback],
    verbose=1,
)
