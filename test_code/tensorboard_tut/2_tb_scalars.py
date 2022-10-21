import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

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
num_epochs = 10
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_writer = tf.summary.create_file_writer("logs/train/")
test_writer = tf.summary.create_file_writer("logs/test/")
train_step = test_step = 0


for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    train_step = test_step = 0
    train_writer = tf.summary.create_file_writer("logs/train/" + str(lr))
    test_writer = tf.summary.create_file_writer("logs/test/" + str(lr))
    model = get_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for epoch in range(num_epochs):
        # Iterate through training set
        for batch_idx, (x, y) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_fn(y, y_pred)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            acc_metric.update_state(y, y_pred)

        with train_writer.as_default():
            tf.summary.scalar("Loss", loss, step=epoch+1)
            tf.summary.scalar(
                "Accuracy", acc_metric.result(), step=epoch+1,
            )
            train_step += 1

        # Reset accuracy in between epochs (and for testing and test)
        acc_metric.reset_states()

        # Iterate through test set
        for batch_idx, (x, y) in enumerate(ds_test):
            y_pred = model(x, training=False)
            loss = loss_fn(y, y_pred)
            acc_metric.update_state(y, y_pred)

        with test_writer.as_default():
            tf.summary.scalar("Loss", loss, step=epoch+1)
            tf.summary.scalar(
                "Accuracy", acc_metric.result(), step=epoch+1,
            )
            test_step += 1

        acc_metric.reset_states()

    # Reset accuracy in between epochs (and for testing and test)
    acc_metric.reset_states()