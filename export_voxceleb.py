"""
This file prepares the VoxCeleb1 dataset test set for TFHE inference using Concrete Core + Concrete GPU.
Since the libraries are written in Rust, we want Rust to contain arrays of data the same way
TF does. 

This file preprocesses the test dataset, normalizes the vectors and saves two .npy + one .npz file 
that zips together two arrays:
    1) x: matrix of [n, 28, 28] ternary values {-1, 0, 1}
    2) y_true: matrix of [n, 1] integers in [0, 9] depicting the 10 class labels

The "x" matrix is the batch of 28x28 pixel (normalized, ternarized) test images.

The ternarization is defined by a threshold theta = 0.7*tf.reduce_mean(tf.abs(batch))) as in
Ternary Weight Networks (TWN).
"""

import os
import tensorflow as tf
import keras.backend as K
import tensorflow_datasets as tfds
import numpy as np

# DATA
import data
from preprocessing import import_dataset
from utils.audio_processing import *
# MODEL
from model import *
# HELPER FUNCTIONS
from main import setup_hparams
# PARAMS
from config import *

strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
num_workers = 1
dtype = tf.float32

# Directory with hparams (need for dataset params)
checkpoint_dir = "/home/vele/Documents/masters/tf_speaker_rec_runs/runs/202302/20230208-192733/checkpoints/"

# Init hparams
hparams, tb_hparams = setup_hparams(
    log_dir=checkpoint_dir+"../"+TB_LOGS_DIR,
    hparam_dir=checkpoint_dir+"../"+TB_LOGS_DIR
)

# Load VoxCeleb
_, _, ds_test, ds_info = import_dataset("voxceleb", VOXCELEB_DIR)
print("Loaded VoxCeleb test set")


def ternarize_tensor_with_threshold(x, theta=1, hparams=None):
    """
    Ternary quantizer where 
    x = -1 if x <= -threshold,
    x = 0  if -threshold < x < threshold,
    x = 1  if x >= threshold
    """
    q = K.cast(tf.abs(x) >= theta, K.floatx()) * tf.sign(x)
    # q = K.cast(tf.abs(x) >= theta, K.floatx()) * x
    return q


def cast_tensor_to_float32(tensor):
    return tf.cast(tensor, tf.float32)

def ternarize(data):
    return ternarize_tensor_with_threshold(data, theta=0.7*tf.reduce_mean(tf.abs(data)))

def add_batch_dim(x):
    return tf.expand_dims(x, axis=0)

def remove_batch_dim(x):
    return tf.squeeze(x, axis=0)

def cast_tensor_to_int8(tensor):
    return tf.cast(tensor, tf.int8)

def cast_tensor_to_int32(tensor):
    return tf.cast(tensor, tf.int32)

def make_even_by_dropping(data):
    if tf.math.floormod(tf.shape(data)[0], 2) == 1:
        return data[:-1, :]
    return data


def prepare_for_tfhe(x, y):
    x = cast_tensor_to_float32(x)
    x = convert_to_log_mel_spec(x, hparams=hparams, normalize=True)
    x = group_and_downsample_spec_v2(x, hparams=hparams)
    x = make_even_by_dropping(x)
    x = ternarize(x)
    x = cast_tensor_to_int8(x)
    y = cast_tensor_to_int32(y-1)
    return x, y


# Preprocess the data
AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_test = ds_test.filter(lambda x, y: tf.shape(x)[0] < 64000)
ds_test = ds_test.map(prepare_for_tfhe, num_parallel_calls=AUTOTUNE)

print("Preprocessed VOXCELEB for TFHE")

# Iterate over the dataset and append the elements to the list
x = []
y = []
for i, example in enumerate(ds_test):
    x.append(example[0].numpy())
    y.append(example[1].numpy())
    if i < 10:
        print("x_len = %d, y_val = %d" % (x[i].shape[0], y[i]))
y = np.array(y)

# More pythonic lol
# X, y = zip(*tfds.as_numpy(ds_test))

print("Converted to numpy")

# Save files
directory = './voxceleb_preprocessed'
if not os.path.exists(directory):
    os.makedirs(directory)
np.savez(directory+"/voxceleb_tern.npz", *x)
np.save(directory+"/voxceleb_labels.npy", y)

print("Saved NPY files of preprocessed VOXCELEB Images and Labels")
