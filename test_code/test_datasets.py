from scipy.io.wavfile import write
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Import datasets
import tensorflow_datasets as tfds

# Ausio processing functions
from utils.audio_processing import *
from data.voxceleb import *

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

datasets = tfds.list_builders()

# Test voxceleb
(ds_train, ds_val, ds_test), ds_info = tfds.load('voxceleb',
                                                 split=[
                                                     'train', 'validation', 'test'],
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True,
                                                 data_dir="/media/vele/Data/Documents/University Files/Masters/Thesis/dev/datasets/VoxCeleb/VoxCeleb1")
assert isinstance(ds_train, tf.data.Dataset)
assert isinstance(ds_val, tf.data.Dataset)
assert isinstance(ds_test, tf.data.Dataset)

print(ds_info)

# Preprocessing testing
# ds_train = ds_train
# ds_val = ds_val
# ds_test = ds_test

# Dataset testing
# ds_train_test = ds_train.take(10)  # take examples

# example is `{'audio': tf.Tensor, 'label': tf.Tensor}`
# for audio, label in ds_train_test:
#     print(audio, label)

# Dataframe testing
# df_train_test = tfds.as_dataframe(ds_train_test, ds_info)
# print(df_train_test)

# ds_* is in tuple format: (audio, label) where audio is list of int64's, label is int64, both in tensors


# ----------------------------------- PREPROCESSING TESTING ---------------------------------------


# -------------------------------- RAW AUDIO VISUALIZATION --------------------------------

# Extract sample
example = ds_train.take(1)
sample = None
sample_label = None
for a, l in example:
    # Copy tensor to global
    sample = tf.identity(a)
    sample_label = tf.identity(l)
print("Audio tensor Def.: ", sample)

# Convert to NP
a_np = sample.numpy()

# Visualize audio
print("-------------- Visualize Audio Sample ----------------")

print("Audio tensor as NP: ", a_np)
x = np.arange(0, len(a_np), 1)
fig, ax = plt.subplots()
ax.plot(x, a_np)
ax.set(xlabel='Sample No.', ylabel='Amplitude (int64)',
       title='Visualization of Random Audio Sample (sr=16kHz)')
ax.grid()
fig.savefig("rand_aud_sample.png")
# plt.show()

print("---------------- Save Scaled Audio Sample ------------------")

# Scales raw audio, creates quantized arrays. Quantizations <= 8 are limited within their
# quantiziation bit-level but represented in uint8 arrays

# int32
a_scaled_int32 = np.int32(a_np/np.max(np.abs(a_np)) * 2147483647)
write('rand_aud_samp_16kHz_int32.wav', 16000, a_scaled_int32)

# int16
a_scaled_int16 = np.int16(a_np/np.max(np.abs(a_np)) * 32767)
write('rand_aud_samp_16kHz_int16.wav', 16000, a_scaled_int16)

# uint8
a_scaled_uint8 = np.uint8(a_np/np.max(np.abs(a_np)) * 127 + 127)
write('rand_aud_samp_16kHz_uint8.wav', 16000, a_scaled_uint8)

# uint6
a_scaled_uint6 = np.uint8(a_np/np.max(np.abs(a_np)) * 31 + 31)
write('rand_aud_samp_16kHz_uint6.wav', 16000, a_scaled_uint6)

# uint5
a_scaled_uint5 = np.uint8(a_np/np.max(np.abs(a_np)) * 15 + 15)
write('rand_aud_samp_16kHz_uint5.wav', 16000, a_scaled_uint5)

# uint4
a_scaled_uint4 = np.uint8(a_np/np.max(np.abs(a_np)) * 7 + 7)
write('rand_aud_samp_16kHz_uint4.wav', 16000, a_scaled_uint4)

# uint1
a_scaled_uint1 = np.uint8(np.heaviside(a_np/np.max(np.abs(a_np)), 0))
write('rand_aud_samp_16kHz_uint1.wav', 16000, a_scaled_uint1)

print("-------------- Visualize Scaled Audio Sample ----------------")

# Shows a figure of multiple plots, how their quantized versions of the raw audio relate

list_scaled_samples = [a_scaled_int32, a_scaled_int16, a_scaled_uint8,
                       a_scaled_uint6, a_scaled_uint5, a_scaled_uint4, a_scaled_uint1]
list_scales = ["int32", "int16", "uint8", "uint6", "uint5", "uint4", "uint1"]

slice = -1  # No. of samples to show
x = np.arange(0, len(a_np), 1)[:slice]
fig, ax = plt.subplots(len(list_scaled_samples))
fig.suptitle('Visualization of Random Audio Sample (Scaled, sr=16kHz)')
for i, scaled in enumerate(list_scaled_samples):
    ax[i].plot(x, scaled[:slice])
    ax[i].set_ylabel(list_scales[i], rotation=0,
                     loc='center', labelpad=20, fontsize=8)
    ax[i].tick_params(axis='both', labelsize=6)
    if i != len(list_scaled_samples)-1:
        ax[i].set_xticks([])
ax[-1].set(xlabel='Sample Number')
fig.savefig("scaled_aud_sample.png", dpi=500)

# -------------------------------------------------------------------------------------------

# Inspiration from https://github.com/noahchalifour/rnnt-speech-recognition/blob/master/utils/preprocessing.py
# and https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms

# MFCC Tests
BATCH_SIZE = 64
NUM_BATCHES = 1
SAMPLE_RATE = 16000.0
MAX_NUM_FRAMES = 203  # for 2s waveform segments
WINDOW_SIZE = 400  # 25ms @ 16kHz
STEP_SIZE = 160  # 10ms @ 16kHz

FFT_LENGTH = 1024
UPPER_EDGE_HERTZ = 7600.0
LOWER_EDGE_HERTZ = 80.0
NUM_MEL_BINS = 80

max_audio = MAX_NUM_FRAMES * STEP_SIZE + WINDOW_SIZE - STEP_SIZE

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Cast audio to float32
sample = tf.cast(sample, tf.float32)

# Pad the audio if smaller than desired length
sample = pad_audio(sample, max_audio)

# Take random segment of audio
sample = take_random_segment(sample)

batch = True
if batch:
    sample_np = sample.numpy()
    sample_batch = tf.reshape(sample, [1, tf.shape(sample)[0]])
    for i in range(BATCH_SIZE-1):
        # sample_batch = tf.map_fn(lambda x: tf.concat((x, sample_np), axis=0), sample_batch)
        sample_batch = tf.concat((sample_batch, tf.reshape(sample, [1, tf.shape(sample)[0]])), axis=0)

    # Get log_mel spectrogram of sample
    sample_batch = convert_to_log_mel_spec_layer(sample_batch, sr=SAMPLE_RATE, num_mel_bins=NUM_MEL_BINS,
                                           window_size=WINDOW_SIZE, step_size=STEP_SIZE, low_hertz=LOWER_EDGE_HERTZ,
                                           upper_hertz=UPPER_EDGE_HERTZ)

    sample_batch = normalize_log_mel_layer(sample_batch)

    # Group timesteps and downsample log-mel spectrogram
    sample_batch = group_and_downsample_spec_v2_layer(sample_batch)

# Get log_mel spectrogram of sample
log_mel_spec = convert_to_log_mel_spec(sample, sr=SAMPLE_RATE, num_mel_bins=NUM_MEL_BINS,
                                       window_size=WINDOW_SIZE, step_size=STEP_SIZE, low_hertz=LOWER_EDGE_HERTZ, upper_hertz=UPPER_EDGE_HERTZ)

log_mel_spec = normalize_log_mel(log_mel_spec)

# Group timesteps and downsample log-mel spectrogram
ds_log_mel_spec = group_and_downsample_spec_v2(log_mel_spec)

print("End")
