import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_jit("autoclustering")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from mnist_rnn_model import get_model

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
            print(str(e))

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

strategy, _, num_workers = configure_environment(gpu_names=None, fp16_run=False, multi_strategy=False)

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
BATCH_SIZE = 512*num_workers

# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=True)
ds_test = ds_test.prefetch(AUTOTUNE)

# Specify checkpoint dir
checkpoint_dir = None
if checkpoint_dir:
    run_dir = checkpoint_dir[0:-12]

# Specify location of pretrained weights
pretrained_weights = None

# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-134607/checkpoints/" # Original, tanh, default init, Adam(1e-4), L2(1e-4), 99%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-140942/checkpoints/" # Original, sign_with_tanh, 20221208-134607 init, Adam(1e-5, clipnorm=1), 98%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-150638/checkpoints/" # tern input, sign_with_tanh, 20221208-140942 init, Adam(1e-5, clipnorm=1), 99%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221208-154453/checkpoints/" # tern input, sign_with_tanh, 20221208-150638 init, tern(0.1) for all, NoAccRegV2(lm=0,k=8) -> QRNN_0:0.63,QRNN_1:0.54,DENSE_0:0.59, Adam(1e-5, clipnorm=1), 93%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221221-232808/checkpoints/" # tern input, sign_with_tanh, 20221208-154453 init, tern(0.1) for all, NoAccRegV2(lm=5e-4,k=8) -> QRNN_0:1.0,QRNN_1:0.89,DENSE_0:0.79, Adam(CosineDecay(1e-4, 500, alpha=0.1), clipnorm=1), 85%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221221-232819/checkpoints/" # tern input, sign_with_tanh, 20221208-154453 init, tern(0.1) for all, NoAccRegV2(lm=1e-3,k=8) -> QRNN_0:1.0,QRNN_1:0.95,DENSE_0:0.87, Adam(CosineDecay(1e-4, 500, alpha=0.1), clipnorm=1), 73%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221221-193404/checkpoints/" # tern input, sign_with_tanh, 20221208-154453 init, tern(0.1) for all, NoAccRegV2(lm=1e-4,k=8) -> QRNN_0:0.99,QRNN_1:0.64,DENSE_0:0.64, Adam(CosineDecay(1e-4, 500, alpha=0.1), clipnorm=1), 92%

# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221216-184318/checkpoints/" # Original, sign_with_tanh, 20221208-134607 init, NoAccRegV2(lm=5e-6,k=4) -> QRNN_0:1.0,QRNN_1:0.99,DENSE_0:0.95, Adam(CosineDecay(1e-4, 100, alpha=0.1), clipnorm=1), 98%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221220-123439/checkpoints/" # tern input, sign_with_tanh, 20221216-184318 init, NoAccRegV2(lm=5e-6,k=4) -> QRNN_0:1.0,QRNN_1:0.99,DENSE_0:0.95, Adam(CosineDecay(1e-4, 100, alpha=0.1), clipnorm=1), 98%

# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221222-193832/checkpoints/" # og
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221222-195050/checkpoints/" # og + sign
pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221222-202422/checkpoints/" # og + sign + tern
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221223-150643/checkpoints/" # og + sign + tern + quant, 92%

# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221229-123124/checkpoints/" # og + sign + tern + quant + NoAccReg1(0.001 all, 0.005 Dense0), 20221223-150643 init, 71%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221229-144111/checkpoints/" # og + sign + tern + quant + NoAccReg2(0.001 all, 0.005 Dense0), 20221223-150643 init, 82%
# pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202212/20221229-182840/checkpoints/" # og + sign + tern + (quant + NoAccReg2(0.001 all, 0.005 Dense0)), 20221222-202422 init, 84.48%

pretrained_weights = "/home/vele/Documents/masters/mnist_rnn/runs/202302/20230205-190617/checkpoints/"

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

    # Calculate total weight stats
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            all_weights = tf.concat([tf.reshape(x, shape=[-1]) for x in layer.trainable_weights], axis=-1)
            tf.print("std   W  in ", layer.name, 1.0*tf.math.reduce_std(all_weights))
            tf.print("mean  W  in ", layer.name, 1.0*tf.math.reduce_mean(all_weights))
            tf.print("mean |W| in ", layer.name, 1.0*tf.math.reduce_mean(tf.abs(all_weights)))
            tf.print("")
    all_weights = tf.concat([tf.reshape(x, shape=[-1]) for x in model.trainable_weights], axis=-1)
    tf.print("std   W  all weights in ", model.name, 1.0*tf.math.reduce_std(all_weights))
    tf.print("mean  W  all weights in ", model.name, 1.0*tf.math.reduce_mean(all_weights))
    tf.print("mean |W| all weights in ", model.name, 1.0*tf.math.reduce_mean(tf.abs(all_weights)))

    model.compile(
        # optimizer=larq.optimizers.Bop(threshold=1e-8, gamma=1e-4), # first ever binary optimizer (flips weights based on grads)
        optimizer=tf.keras.optimizers.Adam(clipnorm=1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

# model.run_eagerly = True

# Define callbacks
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=(RUN_DIR + TB_LOGS_DIR), histogram_freq=1, update_freq="epoch", # profile_batch='100,200'
)

# Add a lr decay callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(
    tf.keras.optimizers.schedules.CosineDecay(1e-4, 100, alpha=0.1), #500
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
            validation_data=ds_test,
            callbacks=[tb_callback, ckpt_callback, lr_callback],
            verbose=1,
        )
    except Exception as e:
        print(e)

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



