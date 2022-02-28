import numpy as np
import tensorflow as tf

# Import datasets
import tensorflow_datasets as tfds

# Audio processing functions
from utils.audio_processing import *

# Import configuration
from config import *

# Spectrogram settings
NUM_MEL_BINS = HP_NUM_MEL_BINS.domain.values[0]
UPPER_HERTZ = HP_UPPER_HERTZ.domain.values[0]
LOWER_HERTZ = HP_LOWER_HERTZ.domain.values[0]
FFT_LENGTH = HP_FFT_LENGTH.domain.values[0]

# Downsampling settings
DOWNSAMPLE_FACTOR = HP_DOWNSAMPLE_FACTOR.domain.values[0]
STACK_SIZE = HP_STACK_SIZE.domain.values[0]


def convert_to_float32(data):
    return tf.cast(data, dtype=tf.float32)


def take_random_segment(data, max_audio_length):
    # Take random segment of audio
    rand_start_idx = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=tf.shape(data)[0] - max_audio_length - 1,
        dtype=tf.int32)

    return data[rand_start_idx:(rand_start_idx + max_audio_length)]


def preprocess(data, label, max_audio_length):
    processed = convert_to_float32(data)
    processed = pad_audio(processed, max_audio_length)
    processed = take_random_segment(processed, max_audio_length)
    # processed = convert_to_log_mel_spec(
    #     processed,
    #     sr=SAMPLE_RATE,
    #     num_mel_bins=NUM_MEL_BINS,
    #     window_size=FRAME_LENGTH,
    #     step_size=FRAME_STEP,
    #     low_hertz=LOWER_HERTZ,
    #     upper_hertz=UPPER_HERTZ)
    # processed = normalize_log_mel(processed)
    # processed = group_and_downsample_spec_v2(
    #     processed,
    #     n=DOWNSAMPLE_FACTOR,
    #     stack_size=STACK_SIZE)

    return processed, label
