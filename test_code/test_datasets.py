from scipy.io.wavfile import write
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
MAX_NUM_FRAMES = 200  # for 2s waveform segments
WINDOW_SIZE = 400  # 25ms @ 16kHz
STEP_SIZE = 160  # 10ms @ 16kHz

FFT_LENGTH = 1024
UPPER_EDGE_HERTZ = 7600.0
LOWER_EDGE_HERTZ = 80.0
NUM_MEL_BINS = 80

max_audio = MAX_NUM_FRAMES * STEP_SIZE + WINDOW_SIZE - STEP_SIZE


def pad_audio(audio, min_audio_size):
    """Pads audio that is too short to process

    Pads the audio sample with zeros to the right until length of audio is equal 
    to min_audio_size

    Args:
        audio ([tf.Tensor]): [Raw audio]
        min_audio_size ([type]): [Size that audio must be]

    Returns:
        [tf.Tensor]: [Correctly sized audio]
    """

    # audio is tensor, max_audio is int
    reshaped = audio
    audio_size = audio.shape[0]
    if(audio_size < min_audio_size):
        reshaped = tf.pad(audio, [[0, min_audio_size-audio_size]])
    return reshaped


def convert_to_log_mel_spec(sample,
                            sr=16000.0,
                            num_mel_bins=80,
                            window_size=400,
                            step_size=160,
                            upper_hertz=7600.0,
                            low_hertz=80.0):
    """Generate log-mel spectrogram of raw audio sample

    Args:
        sample ([tf.Tensor]): [Raw audio sample]
        sr (int, optional): [Sample rate]. Defaults to 16000.0.
        num_mel_bins (int, optional): [Number of mel scale bins]. Defaults to 80.
        window_size (int, optional): [Number of samples in 1 window]. Defaults to 400.
        step_size (int, optional): [Number of samples in a stride]. Defaults to 160.
        upper_hertz (float, optional): [Upper hertz energy]. Defaults to 7600.0.
        low_hertz (float, optional): [Lower hertz energy]. Defaults to 80.0.

    Returns:
        [tf.Tensor]: [Normalized log-mel spectrogram]
    """

    # Perform short-time discrete fourier transform to get spectrograms
    stfts = tf.signal.stft(sample,
                           frame_length=window_size,
                           frame_step=step_size)
    specs = tf.abs(stfts)

    # Get mel spectrograms
    num_spec_bins = tf.shape(specs)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins, num_spectrogram_bins=num_spec_bins,
        sample_rate=sr,
        lower_edge_hertz=low_hertz,
        upper_edge_hertz=upper_hertz)
    mel_specs = tf.tensordot(specs, linear_to_mel_weight_matrix, 1)
    mel_specs.set_shape(specs.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Get log-mel spectrograms
    log_mel_specs = tf.math.log(mel_specs + 1e-6)

    # Normalize
    norm_log_mel_specs = log_mel_specs - tf.reduce_mean(log_mel_specs)
    norm_log_mel_specs /= (tf.math.reduce_std(log_mel_specs) + 1e-8)

    return norm_log_mel_specs


def group_and_downsample_spec(mel_spec, n=3, stack_size=4):
    """Downsamples log-mel spectrogram by n with a stride.

    Group features into stack_size every n timesteps. 

    Args:
        mel_spec ([tf.Tensor]): [Log-Mel Spectrogram]
        n (int, optional): [Downsample factor or stride]. Defaults to 3.
        stack_size (int, optional): [Number of features to group]. Defaults to 4.

    Returns:
        [tf.Tensor]: [Downsampled log-mel spectrogram]
    """
    spec_length = mel_spec.shape[0]

    # Trim input to get equal sized groups for every n timesteps
    trimmed_length = spec_length - (spec_length - stack_size) % n
    trimmed_spec = mel_spec[:trimmed_length]

    # Get indicies in trimmed input of beginning elements of each group
    stack_idxs = tf.range(0, trimmed_length -
                          stack_size + 1, n, dtype=tf.float32)

    # Group and downsample
    downsampled = tf.map_fn(
        lambda i: tf.reshape(trimmed_spec[int(i):int(i)+stack_size], [-1]), stack_idxs)

    return downsampled


# Cast audio to float32
sample_float32 = tf.cast(sample, tf.float32)

# Pad the audio if smaller than desired length
sample_float32 = pad_audio(sample_float32, max_audio)

# Take random segment of audio
rand_start_idx = np.random.randint(0, sample_float32.shape[0]-max_audio-1)
sample_seg = sample_float32[rand_start_idx:(rand_start_idx+max_audio)]

# Get log_mel spectrogram of sample
log_mel_spec = convert_to_log_mel_spec(sample_seg, sr=SAMPLE_RATE, num_mel_bins=NUM_MEL_BINS,
                                       window_size=WINDOW_SIZE, step_size=STEP_SIZE, low_hertz=LOWER_EDGE_HERTZ, upper_hertz=UPPER_EDGE_HERTZ)

# Group timesteps and downsample log-mel spectrogram
ds_log_mel_spec = group_and_downsample_spec(log_mel_spec)

print("End")
