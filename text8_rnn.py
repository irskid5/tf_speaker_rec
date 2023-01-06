import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

import io
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit(True)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from qkeras import *
import larq

from utils.model_utils import TimeReduction, sign_with_tanh_deriv, QDenseWithNorm, QIRNN, custom_sign_with_tanh_deriv_mod_on_inputs
from quantization import ternarize_tensor_with_threshold, LearnedThresholdTernary

SEED = 1997
RUNS_DIR = "/home/vele/Documents/masters/text8_rnn/runs/"
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

strategy, _, num_workers = configure_environment(gpu_names=None, fp16_run=False, multi_strategy=True)

# IMPORT DATASET

from data.creators import text8_creator

ds_train, ds_val, ds_test, ids_from_chars, chars_from_ids = text8_creator.text8_creator()
num_features = len(ids_from_chars.get_vocabulary())

def split_input_target_and_one_hot(sequence):
    one_hot_sequence = tf.keras.layers.CategoryEncoding(num_tokens=num_features, output_mode="one_hot")(sequence)
    input_text = one_hot_sequence[:-1]
    target_text = one_hot_sequence[1:]
    return input_text, target_text

SEQ_LEN = 50
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128*num_workers

# Setup for train dataset
ds_train = ds_train.batch(SEQ_LEN+1, drop_remainder=True)
ds_train = ds_train.map(split_input_target_and_one_hot, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(buffer_size=1000000)
ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for validation dataset
ds_val = ds_val.batch(SEQ_LEN+1, drop_remainder=True)
ds_val = ds_val.map(split_input_target_and_one_hot, num_parallel_calls=AUTOTUNE)
ds_val = ds_val.cache()
ds_val = ds_val.batch(BATCH_SIZE, drop_remainder=True)
ds_val = ds_val.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_test.batch(SEQ_LEN+1, drop_remainder=True)
ds_test = ds_test.map(split_input_target_and_one_hot, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=True)
ds_test = ds_test.prefetch(AUTOTUNE)

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
learned_thresh = False
tern = False
add_no_acc_reg = False
add_dist_loss = False
acc_precision = 6

# Activation functions
activation_irnn = tf.keras.activations.relu
activation_dense = tf.keras.activations.relu

# Activation fns (quantized)
# activation_irnn = sign_with_tanh_deriv
# activation_dense = sign_with_tanh_deriv

# Activation fns (quantized with mod)
# activation_irnn = lambda x: custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=acc_precision)
# activation_dense = lambda x: custom_sign_with_tanh_deriv_mod_on_inputs(x, num_bits=acc_precision)

# Initializers
rnn_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.5 if soft_thresh_tern else 1.0, mode="fan_avg", distribution="uniform", seed=SEED) # "he_normal" is default 
rnn_recurrent_initializer = tf.keras.initializers.Identity()
dense_kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.5 if soft_thresh_tern else 1.0, mode="fan_avg", distribution="uniform", seed=SEED) # "he_normal" is default 


def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((SEQ_LEN,num_features)),
            QIRNN(
                cell=None,
                units=2048, 
                activation=activation_irnn, 
                use_bias=False, 
                return_sequences=True, 
                kernel_regularizer=kernel_regularizer, 
                recurrent_regularizer=recurrent_regularizer,
                kernel_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.1, name="QRNN_0/quantized_kernel") if learned_thresh else kernel_quantizer,
                recurrent_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.05, name="QRNN_0/quantized_recurrent") if learned_thresh else recurrent_quantizer,
                kernel_initializer=rnn_kernel_initializer,
                recurrent_initializer=rnn_recurrent_initializer,
                add_dist_loss=add_dist_loss,
                add_no_acc_reg=add_no_acc_reg,
                no_acc_reg_lm=0.001,
                no_acc_reg_bits=acc_precision,
                s=1,
                name="QRNN_0"),
            # QIRNN(
            #     cell=None,
            #     units=256,  
            #     activation=activation_irnn, 
            #     use_bias=False, 
            #     return_sequences=True, 
            #     kernel_regularizer=kernel_regularizer, 
            #     recurrent_regularizer=recurrent_regularizer,
            #     kernel_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.045, name="QRNN_1/quantized_kernel") if learned_thresh else kernel_quantizer,
            #     recurrent_quantizer=LearnedThresholdTernary(scale=1.0, threshold=0.06, name="QRNN_1/quantized_recurrent") if learned_thresh else recurrent_quantizer,
            #     kernel_initializer=rnn_kernel_initializer,
            #     recurrent_initializer=rnn_recurrent_initializer,
            #     add_dist_loss=add_dist_loss,
            #     add_no_acc_reg=add_no_acc_reg,
            #     no_acc_reg_lm=0.001,
            #     no_acc_reg_bits=acc_precision,
            #     s=1,
            #     name="QRNN_1"),
            # QDenseWithNorm(
            #     1024, 
            #     activation=activation_dense, 
            #     use_bias=False,
            #     kernel_regularizer=kernel_regularizer, 
            #     kernel_quantizer=LearnedThresholdTernary(scale=1.0, name="DENSE_0/quantized_kernel") if learned_thresh else kernel_quantizer,
            #     kernel_initializer=dense_kernel_initializer,
            #     add_dist_loss=add_dist_loss,
            #     add_no_acc_reg=add_no_acc_reg,
            #     no_acc_reg_lm=0.005,
            #     no_acc_reg_bits=acc_precision,
            #     s=1,
            #     name="DENSE_0",),
            QDenseWithNorm(
                num_features, 
                use_bias=False,
                activation=tf.keras.activations.softmax, 
                kernel_regularizer=kernel_regularizer, 
                kernel_quantizer=LearnedThresholdTernary(scale=1.0, name="DENSE_OUT/quantized_kernel") if learned_thresh else kernel_quantizer,
                kernel_initializer=dense_kernel_initializer, 
                add_dist_loss=add_dist_loss,
                add_no_acc_reg=False,
                s=1,
                name="DENSE_OUT",),
        ],
        name="TEXT8_BASIC_RNN"
    )

    model.summary()

    return model

# Specify checkpoint dir
checkpoint_dir = None
if checkpoint_dir:
    run_dir = checkpoint_dir[0:-12]

# Specify location of pretrained weights
pretrained_weights = None

# pretrained_weights = "/home/vele/Documents/masters/text8_rnn/runs/202301/20230106-143250/checkpoints/"

with strategy.scope():
    model = get_model()

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
        print('Restored pretrained weights from {}.'.format(pretrained_weights))

    # Load weights if we starting from checkpoint
    if checkpoint_dir is not None:
        model.load_weights(checkpoint_dir)
        print('Restored checkpoint weights from {}.'.format(checkpoint_dir))

    # Reset the stat variables
    weights = model.get_weights()
    for i in range(len(weights)):
        if "/w" in model.weights[i].name or "/x" in model.weights[i].name:
            weights[i] = 0*weights[i]
    model.set_weights(weights)

    model.compile(
        # optimizer=larq.optimizers.Bop(threshold=1e-8, gamma=1e-4), # first ever binary optimizer (flips weights based on grads)
        optimizer=tf.keras.optimizers.Adam(),
        loss={"DENSE_OUT": tf.keras.losses.CategoricalCrossentropy(from_logits=False)},
        loss_weights={"DENSE_OUT": np.log2(np.e)},
        metrics=["accuracy"],
    )

# model.run_eagerly = True

# Define callbacks
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=(RUN_DIR + TB_LOGS_DIR), histogram_freq=1, update_freq="epoch", # profile_batch='1000,1200'
)

# Add a lr decay callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(
    tf.keras.optimizers.schedules.CosineDecay(1e-4, 500, alpha=0.1),
    verbose=0,
)

if RECORD_CKPTS:
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(RUN_DIR + CKPT_DIR),
                                                save_weights_only=True,
                                                save_best_only=True,
                                                monitor="val_loss",
                                                mode="min",
                                                verbose=1)
else:
    ckpt_callback = None

train = True 
test = False

if train:
    try:
        model.fit(
            ds_train,
            epochs=1000,
            validation_data=ds_val,
            callbacks=[tb_callback, ckpt_callback, lr_callback],
            verbose=1,
        )
    except:
        pass

if test:
    try:
        for _ in range(10):
            model.evaluate(
                x=ds_test,
                verbose=1,
            )
    except:
        pass

print("End!")



