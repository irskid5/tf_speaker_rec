# Import datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Audio processing functions
from utils.audio_processing import *

# Import configuration
from config import *

# Init constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
# AUTOTUNE = 1


def import_dataset(dataset_name, dataset_dir):
    """Imports a dataset and splits into train, validation, and test

    Args:
        dataset_name (String): Name of the dataset as in TFDS
        dataset_dir (String): Directory where dataset is

    Returns:
        Tuple: (Training ds, Validation ds, Test ds, Dataset info)
    """
    (ds_train, ds_val, ds_test), ds_info = tfds.load(dataset_name,
                                                     split=[
                                                         'train', 'validation', 'test'],
                                                     shuffle_files=True,
                                                     as_supervised=True,
                                                     with_info=True,
                                                     data_dir=dataset_dir)
    # Sanity checks
    assert isinstance(ds_train, tf.data.Dataset)
    assert isinstance(ds_val, tf.data.Dataset)
    assert isinstance(ds_test, tf.data.Dataset)

    return ds_train, ds_val, ds_test, ds_info


def preprocess_and_load(ds_train, ds_val, ds_test, ds_info, preprocessor, num_workers, strategy, hparams, eval_full, dtype):
    # Parameters
    BATCH_SIZE = hparams[HP_BATCH_SIZE.name]
    SHUFFLE_BUFFER_SIZE = hparams[HP_SHUFFLE_BUFFER_SIZE.name]
    SAMPLE_RATE = hparams[HP_SAMPLE_RATE.name]
    FRAME_LENGTH = int(hparams[HP_FRAME_LENGTH.name] * SAMPLE_RATE)
    FRAME_STEP = int(hparams[HP_FRAME_STEP.name] * SAMPLE_RATE)
    MAX_AUDIO_LENGTH = hparams[HP_MAX_NUM_FRAMES.name] * FRAME_STEP + FRAME_LENGTH - FRAME_STEP

    # Setup for train dataset
    ds_train = ds_train.map(lambda x, y: preprocessor.preprocess_cast(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.map(lambda x, y: preprocessor.preprocess(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.shuffle(SHUFFLE_BUFFER_SIZE)  # ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(BATCH_SIZE*num_workers, drop_remainder=True)
    ds_train = ds_train.prefetch(AUTOTUNE)

    # Setup for validation dataset
    ds_val = ds_val.map(lambda x, y: preprocessor.preprocess_cast(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.map(lambda x, y: preprocessor.preprocess(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    # ds_val = ds_val.shuffle(ds_info.splits["validation"].num_examples)
    ds_val = ds_val.batch(BATCH_SIZE*num_workers, drop_remainder=True)
    ds_val = ds_val.prefetch(AUTOTUNE)

    # Setup for test dataset
    ds_test = ds_test.map(lambda x, y: preprocessor.preprocess_cast(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    if not eval_full:
        ds_test = ds_test.map(lambda x, y: preprocessor.preprocess(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.shuffle(ds_info.splits["test"].num_examples)
    ds_test = ds_test.batch(BATCH_SIZE*num_workers, drop_remainder=True)
    ds_test = ds_test.prefetch(AUTOTUNE)

    # ds_train = strategy.experimental_distribute_dataset(ds_train)
    # ds_val = strategy.experimental_distribute_dataset(ds_val)
    # ds_test = strategy.experimental_distribute_dataset(ds_test)

    return ds_train, ds_val, ds_test


def preprocess_and_load_hparam_search(ds_train, ds_val, ds_test, ds_info, preprocessor, num_workers, strategy, hparams, eval_full, dtype):
    # Parameters
    BATCH_SIZE = hparams[HP_BATCH_SIZE.name]
    SHUFFLE_BUFFER_SIZE = hparams[HP_SHUFFLE_BUFFER_SIZE.name]
    SAMPLE_RATE = hparams[HP_SAMPLE_RATE.name]
    FRAME_LENGTH = int(hparams[HP_FRAME_LENGTH.name] * SAMPLE_RATE)
    FRAME_STEP = int(hparams[HP_FRAME_STEP.name] * SAMPLE_RATE)
    MAX_AUDIO_LENGTH = hparams[HP_MAX_NUM_FRAMES.name] * FRAME_STEP + FRAME_LENGTH - FRAME_STEP

    # Setup for train dataset
    ds_train = ds_train.map(lambda x, y: preprocessor.preprocess_cast(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.map(lambda x, y: preprocessor.preprocess(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.shuffle(SHUFFLE_BUFFER_SIZE)  # ds_info.splits["train"].num_examples)
    # ds_train = ds_train.cache()
    ds_train = ds_train.batch(BATCH_SIZE*num_workers, drop_remainder=True)
    # ds_train = ds_train.cache()
    ds_train = ds_train.prefetch(AUTOTUNE)

    # Setup for validation dataset
    ds_val = ds_val.map(lambda x, y: preprocessor.preprocess_cast(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.map(lambda x, y: preprocessor.preprocess(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    # ds_val = ds_val.shuffle(ds_info.splits["validation"].num_examples)
    ds_val = ds_val.batch(BATCH_SIZE*num_workers, drop_remainder=True)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(AUTOTUNE)

    # Setup for test dataset
    ds_test = ds_test.map(lambda x, y: preprocessor.preprocess_cast(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    if not eval_full:
        ds_test = ds_test.map(lambda x, y: preprocessor.preprocess(x, y, MAX_AUDIO_LENGTH, dtype), num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.shuffle(ds_info.splits["test"].num_examples)
    ds_test = ds_test.cache()
    # ds_test = ds_test.batch(BATCH_SIZE*num_workers, drop_remainder=True)
    ds_test = ds_test.prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test


def get_dataset(dataset_name, dataset_dir, preprocessor, num_workers, strategy, hparams, is_hparam_search, eval_full, dtype):
    ds_train, ds_val, ds_test, ds_info = import_dataset(dataset_name, dataset_dir)
    if not is_hparam_search:
        ds_train, ds_val, ds_test = preprocess_and_load(ds_train, ds_val, ds_test, ds_info,
                                                    preprocessor, num_workers, strategy,
                                                    hparams, eval_full, dtype)
    else:
        ds_train, ds_val, ds_test = preprocess_and_load_hparam_search(ds_train, ds_val, ds_test, ds_info,
                                                    preprocessor, num_workers, strategy,
                                                    hparams, eval_full, dtype)
    return ds_train, ds_val, ds_test, ds_info
