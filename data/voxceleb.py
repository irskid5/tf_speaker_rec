import numpy as np
import tensorflow as tf

# Import datasets
import tensorflow_datasets as tfds

# Audio processing functions
import config
from utils.audio_processing import *

# Import configuration
from config import *

# -------------------------- Initialize constants -------------------------------

MAX_NUM_FRAMES = HP_MAX_NUM_FRAMES.domain.values[0]
SAMPLE_RATE = HP_SAMPLE_RATE.domain.values[0]

# Convert seconds to samples
FRAME_LENGTH = int(HP_FRAME_LENGTH.domain.values[0] * SAMPLE_RATE)
FRAME_STEP = int(HP_FRAME_STEP.domain.values[0] * SAMPLE_RATE)

# Get max audio in samples
MAX_AUDIO_LENGTH = MAX_NUM_FRAMES * FRAME_STEP + FRAME_LENGTH - FRAME_STEP

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

def take_random_segment(data):
    # Take random segment of audio
    rand_start_idx = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=tf.shape(data)[0] - MAX_AUDIO_LENGTH - 1,
        dtype=tf.int32)

    return data[rand_start_idx:(rand_start_idx + MAX_AUDIO_LENGTH)]

def preprocess(data, label):
    processed = convert_to_float32(data)
    processed = pad_audio(processed, MAX_AUDIO_LENGTH)
    processed = take_random_segment(processed)
    processed = convert_to_log_mel_spec(
        processed,
        sr=SAMPLE_RATE,
        num_mel_bins=NUM_MEL_BINS,
        window_size=FRAME_LENGTH,
        step_size=FRAME_STEP,
        low_hertz=LOWER_HERTZ,
        upper_hertz=UPPER_HERTZ)
    processed = normalize_log_mel(processed)
    processed = group_and_downsample_spec(
        processed,
        n=DOWNSAMPLE_FACTOR,
        stack_size=STACK_SIZE)

    return processed, label