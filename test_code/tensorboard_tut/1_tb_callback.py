import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from qkeras import *
from utils.model_utils import TimeReduction

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
    "cifar10",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


def augment(image, label):
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32*num_workers

# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.cache()
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_test.cache()
ds_test = ds_train.prefetch(AUTOTUNE)

class_names = [
    "Airplane",
    "Autmobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

def get_model():
    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.Input((32, 32, 3)),
    #         tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
    #         tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    #         tf.keras.layers.MaxPooling2D((2, 2)),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(64, activation="relu"),
    #         tf.keras.layers.Dropout(0.1),
    #         tf.keras.layers.Dense(10),
    #     ]
    # )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((32, 32, 3)),
            tf.keras.layers.Reshape(target_shape=[32, 32*3]),
            tf.keras.layers.SimpleRNN(128, use_bias=False, return_sequences=True),
            TimeReduction(reduction_factor=2, batch_size=int(BATCH_SIZE/4)),
            tf.keras.layers.SimpleRNN(128, use_bias=False, return_sequences=True),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2048, use_bias=False),
            tf.keras.layers.Dense(10),
        ]
    )

    return model

# CUSTOM TRAINING LOOP

# num_epochs = 1
# loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = tf.keras.optimizers.Adam(lr=0.001)
# acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
# train_writer = tf.summary.create_file_writer("logs/train/")
# test_writer = tf.summary.create_file_writer("logs/test/")
# train_step = test_step = 0
#
# for epoch in range(num_epochs):
#     for batch_idx, (x, y) in enumerate(ds_train):
#         with tf.GradientTape() as tape:
#             y_pred = model(x, training=True)
#             loss = loss_fn(y, y_pred)
#         gradients = tape.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#         acc_metric.update_state(y, y_pred)
#
#     acc_metric.reset_states()
#
#     for batch_idx, (x, y) in enumerate(ds_test):
#         y_pred = model(x, training=False)
#         loss = loss_fn(y, y_pred)
#         acc_metric.update_state(y, y_pred)
#
#     acc_metric.reset_states()

with strategy.scope():
    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="tb_callback_dir", histogram_freq=1,
)

model.fit(
    ds_train,
    epochs=100,
    validation_data=ds_test,
    callbacks=[tensorboard_callback],
    verbose=1,
)
