# Import datasets
import tensorflow_datasets as tfds

# Audio processing functions
from utils.audio_processing import *

# Import configuration
from config import *

# Init constants
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = HP_BATCH_SIZE.domain.values[0]


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


def preprocess_and_load(ds_train, ds_val, ds_test, ds_info, preprocessor):
    # Setup for train dataset
    ds_train = ds_train.map(preprocessor.preprocess, num_parallel_calls=AUTOTUNE)
    # ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.shuffle(64000)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(AUTOTUNE)

    # Setup for validation dataset
    ds_val = ds_val.map(preprocessor.preprocess, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.shuffle(ds_info.splits["validation"].num_examples)
    ds_val = ds_val.batch(BATCH_SIZE)
    ds_val = ds_val.prefetch(AUTOTUNE)

    # Setup for test dataset
    ds_test = ds_test.map(preprocessor.preprocess, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.shuffle(ds_info.splits["test"].num_examples)
    # ds_test = ds_test.batch(BATCH_SIZE)
    ds_test = ds_test.prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test


def get_dataset(dataset_name, dataset_dir, preprocessor):
    ds_train, ds_val, ds_test, ds_info = import_dataset(dataset_name, dataset_dir)
    ds_train, ds_val, ds_test = preprocess_and_load(ds_train, ds_val, ds_test, ds_info, preprocessor)
    return ds_train, ds_val, ds_test
