import tensorflow as tf

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
        audio = tf.pad(audio, [[0, min_audio_size-audio_size]])
    return audio


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
    log_mel_spec = tf.math.log(mel_spec + 1e-6)

    return log_mel_spec

def normalize_log_mel(log_mel_spec):
    # Normalize
    normalized = log_mel_spec - tf.reduce_mean(log_mel_spec + 1e-8)
    normalized /= (tf.math.reduce_std(log_mel_spec) + 1e-8)

    return normalized

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
    spec_length = tf.shape(mel_spec)[0]

    # Trim input to get equal sized groups for every n timesteps
    trimmed_length = spec_length - tf.math.mod(spec_length - stack_size, n)
    trimmed_spec = mel_spec[:trimmed_length]

    # Get indicies in trimmed input of beginning elements of each group
    stack_idxs = tf.range(0, trimmed_length -
                          stack_size + 1, n, dtype=tf.float32)

    # Group and downsample
    downsampled = tf.map_fn(
        lambda i: tf.reshape(trimmed_spec[int(i):int(i)+stack_size], [-1]), stack_idxs)

    return downsampled