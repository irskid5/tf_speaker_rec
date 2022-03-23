import tensorflow as tf
from config import *


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
    audio_size = tf.shape(audio)[0]
    if audio_size < min_audio_size:
        audio = tf.pad(audio, [[0, min_audio_size - audio_size + 2]])
    return audio


def convert_to_log_mel_spec(sample, hparams=None,
                            sr=16000.0,
                            num_mel_bins=80,
                            window_size=400,
                            step_size=160,
                            upper_hertz=7600.0,
                            low_hertz=80.0,
                            fft_length=1024,
                            ret_mfcc=False):
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

    if hparams:
        sr=hparams[HP_SAMPLE_RATE.name]
        num_mel_bins=hparams[HP_NUM_MEL_BINS.name]
        window_size=int(hparams[HP_FRAME_LENGTH.name] * sr)
        step_size=int(hparams[HP_FRAME_STEP.name] * sr)
        upper_hertz=hparams[HP_UPPER_HERTZ.name]
        low_hertz=hparams[HP_LOWER_HERTZ.name]
        fft_length=hparams[HP_FFT_LENGTH.name]

    # Perform short-time discrete fourier transform to get spectrograms
    stfts = tf.signal.stft(sample,
                           frame_length=window_size,
                           frame_step=step_size,
                           fft_length=fft_length)
    spec = tf.abs(stfts)

    # Get mel spectrograms
    num_spec_bins = tf.shape(spec)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins, num_spectrogram_bins=num_spec_bins,
        sample_rate=sr,
        lower_edge_hertz=low_hertz,
        upper_edge_hertz=upper_hertz)
    mel_spec = tf.tensordot(spec, linear_to_mel_weight_matrix, 1)
    mel_spec.set_shape(spec.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Get log-mel spectrograms
    log_mel_spec = tf.math.log(mel_spec + 1e-8)
    features = log_mel_spec

    if ret_mfcc:
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spec)
        features = mfccs[:,:,:13]

    return features

def convert_to_log_mel_spec_layer(sample, 
                            hparams=None,
                            sr=16000.0,
                            num_mel_bins=80,
                            window_size=400,
                            step_size=160,
                            upper_hertz=7600.0,
                            low_hertz=80.0,
                            fft_length=1024,
                            ret_mfcc=False):
    """Generate log-mel spectrogram of raw audio sample

    Args:
        sample ([tf.Tensor]): [Raw audio sample]
        sr (float, optional): [Sample rate]. Defaults to 16000.0.
        num_mel_bins (int, optional): [Number of mel scale bins]. Defaults to 80.
        window_size (int, optional): [Number of samples in 1 window]. Defaults to 400.
        step_size (int, optional): [Number of samples in a stride]. Defaults to 160.
        upper_hertz (float, optional): [Upper hertz energy]. Defaults to 7600.0.
        low_hertz (float, optional): [Lower hertz energy]. Defaults to 80.0.

    Returns:
        [tf.Tensor]: [Normalized log-mel spectrogram]
    """

    if hparams:
        sr=hparams[HP_SAMPLE_RATE.name]
        num_mel_bins=hparams[HP_NUM_MEL_BINS.name]
        window_size=int(hparams[HP_FRAME_LENGTH.name] * sr)
        step_size=int(hparams[HP_FRAME_STEP.name] * sr)
        upper_hertz=hparams[HP_UPPER_HERTZ.name]
        low_hertz=hparams[HP_LOWER_HERTZ.name]
        fft_length=hparams[HP_FFT_LENGTH.name]

    # Perform short-time discrete fourier transform to get spectrograms
    stfts = tf.signal.stft(sample,
                           frame_length=window_size,
                           frame_step=step_size,
                           fft_length=fft_length)
    spec = tf.abs(stfts)

    # Get mel spectrograms
    num_spec_bins = tf.shape(spec)[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins, num_spectrogram_bins=num_spec_bins,
        sample_rate=sr,
        lower_edge_hertz=low_hertz,
        upper_edge_hertz=upper_hertz)
    mel_spec = tf.matmul(spec, linear_to_mel_weight_matrix)

    # Get log-mel spectrograms
    log_mel_spec = tf.math.log(mel_spec + 1e-8)

    # Normalize
    # means = tf.math.reduce_mean(log_mel_spec, 1, keepdims=True)
    # stddevs = tf.math.reduce_std(log_mel_spec, 1, keepdims=True)
    # log_mel_spec = (log_mel_spec - means) / (stddevs + 1e-10)

    features = log_mel_spec

    if ret_mfcc:
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spec)
        features = mfccs[:,:,:13]

    return features


def normalize_log_mel(log_mel_spec):
    # Normalize every feature bin, across timesteps
    normalized = log_mel_spec - tf.reduce_mean(log_mel_spec, axis=0)
    normalized /= (tf.math.reduce_std(log_mel_spec, axis=0) + 1e-8)

    return normalized

def normalize_log_mel_layer(log_mel_spec):
    # Normalize every feature bin, across timesteps
    normalized = tf.map_fn(normalize_log_mel, log_mel_spec)
    return normalized


def group_and_downsample_spec(mel_spec, hparams=None, n=3, stack_size=4):
    """Downsamples log-mel spectrogram by n with a stride.

    Group features into stack_size every n timesteps.

    Args:
        mel_spec ([tf.Tensor]): [Log-Mel Spectrogram]
        n (int, optional): [Downsample factor or stride]. Defaults to 3.
        stack_size (int, optional): [Number of features to group]. Defaults to 4.

    Returns:
        [tf.Tensor]: [Downsampled log-mel spectrogram]
    """
    spec_length = tf.shape(mel_spec)[0]

    if hparams:
        n=hparams[HP_DOWNSAMPLE_FACTOR.name]
        stack_size=hparams[HP_STACK_SIZE.name]

    # Trim input to get equal sized groups for every n timesteps
    trimmed_length = spec_length - tf.math.mod(spec_length - stack_size, n)
    trimmed_spec = mel_spec[:trimmed_length]

    # Get indicies in trimmed input of beginning elements of each group
    stack_idxs = tf.range(0, trimmed_length -
                          stack_size + 1, n, dtype=tf.float32)

    # Group and downsample
    downsampled = tf.map_fn(
        lambda i: tf.reshape(trimmed_spec[int(i):int(i) + stack_size], [-1]), stack_idxs)

    return downsampled


def group_and_downsample_spec_v2(mel_spec, hparams=None, n=3, stack_size=4):
    """Downsamples log-mel spectrogram by n with a stride.

    Group features into stack_size every n timesteps.

    Args:
        mel_spec ([tf.Tensor]): [Log-Mel Spectrogram]
        n (int, optional): [Downsample factor or stride]. Defaults to 3.
        stack_size (int, optional): [Number of features to group]. Defaults to 4.

    Returns:
        [tf.Tensor]: [Downsampled log-mel spectrogram]
    """

    if hparams:
        n=hparams[HP_DOWNSAMPLE_FACTOR.name]
        stack_size=hparams[HP_STACK_SIZE.name]

    feat_length = tf.shape(mel_spec)[-1]
    downsampled = tf.signal.frame(mel_spec, stack_size, n, pad_end=False, axis=0)  # group
    downsampled = tf.reshape(downsampled, [tf.shape(downsampled)[0], feat_length * stack_size])  # downsample

    return downsampled

def group_and_downsample_spec_v2_layer(mel_spec, hparams=None, n=3, stack_size=4):
    """Downsamples log-mel spectrogram by n with a stride.

    Group features into stack_size every n timesteps.

    Args:
        mel_spec ([tf.Tensor]): [Log-Mel Spectrogram]
        n (int, optional): [Downsample factor or stride]. Defaults to 3.
        stack_size (int, optional): [Number of features to group]. Defaults to 4.

    Returns:
        [tf.Tensor]: [Downsampled log-mel spectrogram]
    """

    if hparams:
        n=hparams[HP_DOWNSAMPLE_FACTOR.name]
        stack_size=hparams[HP_STACK_SIZE.name]

    feat_length = tf.shape(mel_spec)[-1]
    downsampled = tf.signal.frame(mel_spec, stack_size, n, pad_end=False, axis=1)  # group
    downsampled = tf.reshape(downsampled, [tf.shape(downsampled)[0], tf.shape(downsampled)[1], feat_length * stack_size])  # downsample

    return downsampled
