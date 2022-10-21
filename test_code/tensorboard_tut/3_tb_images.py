import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from test_code.tensorboard_tut.utils import plot_to_image, image_grid

# Make sure we don't get any GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


def augment(image, label):
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)

    # matplotlib wants [0,1] values
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

    return image, label


# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
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
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((32, 32, 3)),
            tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(10),
        ]
    )

    return model


model = get_model()
num_epochs = 1
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
writer = tf.summary.create_file_writer("logs/train/")
step = 0


for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(ds_train):
        figure = image_grid(x, y, class_names)

        with writer.as_default():
            tf.summary.image(
                "Visualize Images", plot_to_image(figure), step=step,
            )
            step += 1