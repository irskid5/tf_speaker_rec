import tensorflow as tf

import numpy as np

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


def text8_creator():
    """Creates the text8 dataset by downloading the most recent version. splitting,
    assigning ids, and packing into tf.data.Dataset object

    Returns:
        tuple: [tf.data.Dataset "train", tf.data.Dataset "validation", tf.data.Dataset "test"]
    """
    # DOWNLOAD AND OPEN -------------------------------------------------

    resp = urlopen("http://mattmahoney.net/dc/text8.zip")
    myzip = ZipFile(BytesIO(resp.read()))
    file = "text8"
    text = myzip.open(file, 'r').read().decode(encoding='utf-8')

    # length of text is the number of characters in it
    print(f'Length of text: {len(text)} characters')

    # Take a look at the first 250 characters in text
    print(text[:250])

    # PREPROCESSING -----------------------------------------------------

    # The unique characters in the file
    vocab = sorted(set(text))
    print(f'{len(vocab)} unique characters')

    # Vectorize the text
    # Before training, you need to convert the strings to a numerical representation.

    # The tf.keras.layers.StringLookup layer can convert each character into a numeric ID. It just needs the text to be split into tokens first.
    example_texts = [' abcdefghijklmnopqrstuvwxyz']

    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
    # print(chars)

    # Now create the tf.keras.layers.StringLookup layer
    # It converts from tokens to character IDs
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None, num_oov_indices=0, output_mode="int")
    # ids = ids_from_chars(tf.reshape(chars.to_tensor(), shape=[-1]))
    # print(ids)

    # This converts from character IDs to tokens
    chars_from_ids = None
    # chars_from_ids = tf.keras.layers.StringLookup(
    #     vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
    # chars = chars_from_ids(ids)
    # print(chars)

    # Use this function to convert a list of chars to a string

    # def text_from_ids(ids):
    #     return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

    # Now split up the text8 file and convert to ids
    print("Split and convert text to ints.")
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    # print(all_ids)

    # Split the dataset into train, val and test
    print("Splitting dataset.")
    ids_dataset_train = tf.data.Dataset.from_tensor_slices(all_ids[:90000000])
    ids_dataset_val = tf.data.Dataset.from_tensor_slices(
        all_ids[90000000:95000000])
    ids_dataset_test = tf.data.Dataset.from_tensor_slices(all_ids[95000000:])
    print("Completed creation on text8 dataset.")

    return ids_dataset_train, ids_dataset_val, ids_dataset_test, ids_from_chars, chars_from_ids

# Save the dataset
# path_to_dataset_folder = "/home/vele/Documents/dev/text8_fun/text8_created"
# print("Saving datasets to " + path_to_dataset_folder)
# print("Saving train dataset...")
# ids_dataset_train.save(path_to_dataset_folder+"/train")
# print("Saving validation dataset...")
# ids_dataset_val.save(path_to_dataset_folder+"/validation")
# print("Saving test dataset...")
# ids_dataset_test.save(path_to_dataset_folder+"/test")
# print("Done!")

# ids_dataset_train = tf.data.Dataset.load(path=path_to_dataset_folder+"/train")
# ids_dataset_val = tf.data.Dataset.load(
#     path=path_to_dataset_folder+"/validation")
# ids_dataset_test = tf.data.Dataset.load(path=path_to_dataset_folder+"/test")

# x = list(ids_dataset_val.as_numpy_iterator())

# print("done")
